// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! `run_shell` tool — executes commands through a pluggable `ShellBackend`.
//!
//! On desktop the default backend is the real OS shell (`NativeShell`); on
//! mobile it is the fastshell sandbox engine (`FastshellBackend`). See
//! `backend.rs`. Returns a JSON string with `{success, returncode, stdout,
//! stderr, command}`. Oversized output is archived under `.aacode/extracts/`.

use super::backend::ShellBackend;
use super::registry::Tool;
use super::schema::{ParamType, ToolParameter, ToolSchema};
use crate::config::{DangerAction, SafetyConfig, UnknownAction};
use crate::error::{AacodeError, Result};
use serde_json::{json, Value};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

// Re-export the fastshell handle type for the rest of the crate.
pub use super::backend::SharedFastshell as SharedShell;

/// Default output caps for `run_shell` (aligned with the Python AACode
/// defaults). `full_output=true` (or `max_output=0`) disables them.
pub const DEFAULT_MAX_OUTPUT_CHARS: usize = 20_000;
pub const DEFAULT_MAX_OUTPUT_LINES: usize = 500;

pub struct ShellTool {
    backend: Arc<dyn ShellBackend>,
    /// Working directory / archive root for command execution.
    cwd: std::path::PathBuf,
    /// Configured char cap. 0 = unset → [`DEFAULT_MAX_OUTPUT_CHARS`].
    max_output_chars: usize,
    /// Configured line cap. 0 = unset → [`DEFAULT_MAX_OUTPUT_LINES`].
    max_output_lines: usize,
    default_timeout_secs: u64,
    default_idle_timeout_secs: u64,
    safety: SafetyConfig,
}

impl ShellTool {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        backend: Arc<dyn ShellBackend>,
        cwd: std::path::PathBuf,
        max_output_chars: usize,
        max_output_lines: usize,
        default_timeout_secs: u64,
        default_idle_timeout_secs: u64,
        safety: SafetyConfig,
    ) -> Self {
        ShellTool {
            // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0
            backend,
            cwd,
            max_output_chars,
            max_output_lines,
            default_timeout_secs,
            default_idle_timeout_secs,
            safety,
        }
    }

    /// Archive a large string to `.aacode/extracts/` and return the path.
    fn archive(&self, content: &str, prefix: &str) -> Option<String> {
        let dir = self.cwd.join(".aacode").join("extracts");
        std::fs::create_dir_all(&dir).ok()?;
        let name = format!("tool_{}_{}.txt", prefix, uuid::Uuid::new_v4().simple());
        let path = dir.join(&name);
        std::fs::write(&path, content).ok()?;
        Some(path.to_string_lossy().to_string())
    }

    /// Apply the line cap first, then the char cap. When either truncates, the
    /// **full** output is archived once so the model can grep/read it later.
    fn maybe_truncate(
        &self,
        text: &str,
        prefix: &str,
        char_limit: usize,
        line_limit: usize,
    ) -> String {
        if text.is_empty() {
            return String::new();
        }
        let mut truncated = false;
        let mut body = text.to_string();

        if line_limit > 0 {
            let mut newlines = 0usize;
            let mut cut = None;
            for (i, ch) in body.char_indices() {
                if ch == '\n' {
                    newlines += 1;
                    if newlines == line_limit {
                        cut = Some(i + 1);
                        break;
                    }
                }
            }
            if let Some(cut) = cut {
                if cut < body.len() {
                    body.truncate(cut);
                    truncated = true;
                }
            }
        }

        if char_limit > 0 && body.chars().count() > char_limit {
            body = body.chars().take(char_limit).collect();
            truncated = true;
        }

        if !truncated {
            return body;
        }

        let total = text.chars().count();
        match self.archive(text, prefix) {
            Some(path) => format!(
                "{body}\n\n[output truncated: max_output={char_limit} chars / {line_limit} lines ({total} chars total). Full output archived at {path}; raise max_output, pass full_output=true, or read it in chunks with head/tail/sed -n.]"
            ),
            None => format!(
                "{body}\n\n[output truncated: max_output={char_limit} chars / {line_limit} lines ({total} chars total); raise max_output or pass full_output=true.]"
            ),
        }
    }

    /// First executable word, skipping leading `VAR=value` assignments.
    fn first_command(command: &str) -> String {
        for raw in command.split_whitespace() {
            let token = raw.trim_matches(|c| {
                matches!(
                    c,
                    '"' | '\'' | '(' | ')' | '{' | '}' | '[' | ']' | ';' | '&' | '|'
                )
            });
            if token.is_empty() {
                continue;
            }
            if let Some((key, _)) = token.split_once('=') {
                if !key.is_empty() && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                    continue;
                }
            }
            return token.to_string();
        }
        String::new()
    }

    /// Commands whose arguments are data, not commands (avoid false positives
    /// like `grep "rm -rf /" file`). `find` only counts when it doesn't run or
    /// delete anything.
    fn is_data_command(name: &str, command: &str) -> bool {
        const DATA: &[&str] = &[
            "echo", "printf", "rg", "ripgrep", "grep", "egrep", "fgrep", "git", "sed", "awk",
            "cat", "head", "tail", "ls", "find",
        ];
        if !DATA.contains(&name) {
            return false;
        }
        if name == "find"
            && ["-exec", "-execdir", "-delete"]
                .iter()
                .any(|f| command.contains(f))
        {
            return false;
        }
        true
    }

    /// Shell wrappers execute their arguments, so quoted/heredoc content must
    /// still be scanned.
    fn is_wrapper(name: &str) -> bool {
        const WRAPPERS: &[&str] = &[
            "bash",
            "sh",
            "zsh",
            "dash",
            "ksh",
            "fish",
            "csh",
            "tcsh",
            "pwsh",
            "powershell",
            "cmd",
            "eval",
            "exec",
            "env",
            "nohup",
            "time",
            "command",
            "xargs",
            "sudo",
        ];
        WRAPPERS.contains(&name)
    }

    /// Drop heredoc bodies so script text is not mistaken for commands.
    fn strip_heredoc_bodies(command: &str) -> String {
        let mut out = Vec::new();
        let mut delimiter: Option<String> = None;
        for line in command.lines() {
            if let Some(delim) = &delimiter {
                if line.trim() == delim {
                    delimiter = None;
                }
                continue;
            }
            out.push(line);
            if let Some(pos) = line.find("<<") {
                let rest = line[pos + 2..].trim_start_matches('-').trim();
                let token = rest.split_whitespace().next().unwrap_or("");
                let token = token.trim_matches(|c| c == '"' || c == '\'');
                if !token.is_empty() && token.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                {
                    delimiter = Some(token.to_string());
                }
            }
        }
        out.join("\n")
    }

    /// Text the dangerous patterns should inspect.
    fn dangerous_scan_text(command: &str) -> String {
        let name = Self::first_command(command);
        if name.is_empty() {
            return String::new();
        }
        if Self::is_data_command(&name, command) {
            return String::new();
        }
        if Self::is_wrapper(&name) {
            return command.to_string();
        }
        Self::strip_heredoc_bodies(command)
    }

    /// Detect blatantly dangerous commands (used only in `reject` mode).
    fn is_dangerous(command: &str) -> bool {
        let c = Self::dangerous_scan_text(command).to_lowercase();
        if c.is_empty() {
            return false;
        }
        let patterns = [
            "rm -rf /",
            "rm -rf /*",
            ":(){:|:&};:", // fork bomb
            "mkfs",
            "dd if=/dev/zero",
            "> /dev/sda",
            "chmod -r 000 /",
        ];
        patterns.iter().any(|p| c.contains(p))
    }

    /// Resolve a command name on PATH / as a shell builtin (native backend).
    fn command_exists(name: &str) -> bool {
        if name.is_empty() {
            return true;
        }
        #[cfg(target_os = "windows")]
        {
            matches!(
                std::process::Command::new("where").arg(name).output(),
                Ok(out) if out.status.success()
            )
        }
        #[cfg(not(target_os = "windows"))]
        {
            matches!(
                std::process::Command::new("sh")
                    .arg("-c")
                    .arg("command -v \"$1\" >/dev/null 2>&1")
                    .arg("sh")
                    .arg(name)
                    .status(),
                Ok(status) if status.success()
            )
        }
    }
}

#[async_trait::async_trait]
impl Tool for ShellTool {
    fn lane(&self) -> Option<&'static str> {
        Some("shell")
    }
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "run_shell",
            "Execute shell commands — the universal Swiss Army knife. This is the ONLY tool for ALL file operations. Use shell commands for: locating (grep -n pattern file, find), reading full content (cat/tail/head file), writing (echo text > file, cat > file << 'EOF'), editing (sed/awk), running code (python/node/go/rustc/gcc), testing (pytest/cargo test), git, and more. Supports pipes (|), redirection (>), heredocs (<< 'EOF'), chaining (&& / || / ;), command substitution ($(...)), and variable expansion ($VAR). Always returns a result object with stdout, stderr, and returncode — check returncode for success. A returncode of 0 means success — treat the action as done and do not re-run it to verify. Opening actions (`open` / `open_settings`) are idempotent and already echo their result in stdout.",
            vec![
                ToolParameter::new(
                    "command",
                    ParamType::String,
                    true,
                    "The shell command to execute. Supports pipes (|), redirection (>), chaining (&& / ;). Always quote filenames with spaces/special chars.",
                    &["cmd", "shell", "script", "exec"],
                ),
                ToolParameter::new(
                    "timeout",
                    ParamType::Integer,
                    false,
                    "Total command timeout in seconds.",
                    &["time_limit", "max_time", "wait"],
                ),
                ToolParameter::new(
                    "idle_timeout",
                    ParamType::Integer,
                    false,
                    "Max seconds with no output before killing (idle timeout). Default: 30.",
                    &["idle_time", "no_output_timeout"],
                ),
                ToolParameter::new(
                    "stdin_input",
                    ParamType::String,
                    false,
                    "Standard input piped to the program (for input()). Separate lines with \\n.",
                    &["input", "stdin"],
                ),
                ToolParameter::new(
                    "max_output",
                    ParamType::Integer,
                    false,
                    "Cap returned output characters (default 20000). Pass 0 to disable the cap, or a larger number for more. Truncated output is archived under .aacode/extracts.",
                    &["max_chars", "limit", "output_limit"],
                ),
                ToolParameter::new(
                    "max_output_lines",
                    ParamType::Integer,
                    false,
                    "Cap returned output lines (default 500). Use full_output=true to disable.",
                    &["output_lines", "max_result_lines"],
                ),
                ToolParameter::new(
                    "full_output",
                    ParamType::Boolean,
                    false,
                    "Disable the default output caps and return the full stdout/stderr. Only use when you truly need everything — large outputs cost many tokens.",
                    &["full", "unlimited"],
                ),
            ],
        )
    }

    async fn call(&self, args: &Value, _cancel: &AtomicBool) -> Result<String> {
        // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0
        let command = args
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Danger policy.
        if self.safety.dangerous_command_action == DangerAction::Reject
            && Self::is_dangerous(&command)
        {
            return Ok(json!({
                "success": false,
                "error": "Command rejected by safety guard (dangerous pattern).",
                "command": command,
            })
            .to_string());
        }

        // Unknown-command policy. Default is `allow` (no-op); when set to
        // `reject`, only a native shell can be checked reliably (fastshell's
        // builtins are not on PATH).
        if self.safety.unknown_command_action == UnknownAction::Reject
            && self.backend.kind() == "native"
        {
            let name = Self::first_command(&command);
            let looks_like_command = !name.is_empty()
                && name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/'));
            if looks_like_command && !Self::command_exists(&name) {
                return Ok(json!({
                    "success": false,
                    "error": format!("Command rejected by safety guard (unknown command '{name}')."),
                    "command": command,
                })
                .to_string());
            }
        }

        let stdin_input = args.get("stdin_input").and_then(|v| v.as_str());
        let timeout = args
            .get("timeout")
            .and_then(|v| v.as_u64())
            .unwrap_or(self.default_timeout_secs);
        let idle_timeout = args
            .get("idle_timeout")
            .and_then(|v| v.as_u64())
            .unwrap_or(self.default_idle_timeout_secs);

        let stdin_owned = stdin_input.map(|s| s.to_string());
        let backend = self.backend.clone();
        let cwd = self.cwd.clone();
        let command2 = command.clone();
        let result = tokio::task::spawn_blocking(move || {
            backend.run(
                &command2,
                stdin_owned.as_deref(),
                timeout,
                idle_timeout,
                &cwd,
            )
        })
        .await
        .map_err(|e| AacodeError::Other(format!("shell spawn: {e}")))?;

        // Output budget: default caps keep huge outputs out of the context; the
        // full text is archived. `full_output=true` / `max_output=0` opts out.
        let wants_full = args
            .get("full_output")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
            || match args.get("max_output") {
                Some(Value::String(s)) => {
                    matches!(
                        s.to_ascii_lowercase().as_str(),
                        "full" | "none" | "null" | ""
                    )
                }
                Some(Value::Number(n)) => n.as_u64() == Some(0),
                _ => false,
            };
        let per_call_chars = args
            .get("max_output")
            .and_then(|v| v.as_u64())
            .filter(|n| *n > 0)
            .map(|n| n as usize);

        let (char_limit, line_limit) = if wants_full {
            (0usize, 0usize)
        } else {
            let chars = per_call_chars.unwrap_or(if self.max_output_chars == 0 {
                DEFAULT_MAX_OUTPUT_CHARS
            } else {
                self.max_output_chars
            });
            let lines = args
                .get("max_output_lines")
                .and_then(|v| v.as_u64())
                .filter(|n| *n > 0)
                .map(|n| n as usize)
                .unwrap_or(if self.max_output_lines == 0 {
                    DEFAULT_MAX_OUTPUT_LINES
                } else {
                    self.max_output_lines
                });
            (chars, lines)
        };

        // Binary stdout must never be lossily pasted into the context. Detect
        // it, persist it as an artifact, and return only a bounded preview.
        let stdout_value = if is_binary_text(&result.stdout) {
            let store = crate::artifacts::ArtifactStore::new(&self.cwd);
            binary_output_json(&store, "stdout", &result.stdout)
        } else {
            json!(self.maybe_truncate(&result.stdout, "stdout", char_limit, line_limit))
        };
        let stderr = self.maybe_truncate(&result.stderr, "stderr", char_limit, line_limit);

        // `success` must reflect the exit code: a timed-out (124) or failing
        // command is not a success (the model otherwise treats 124 as OK).
        let mut out = json!({
            "success": result.exit_code == 0,
            "returncode": result.exit_code,
            "stdout": stdout_value,
            "stderr": stderr,
            "backend": self.backend.kind(),
        });
        if result.exit_code == 124 {
            out["hint"] = json!(
                "command timed out — split it into smaller steps or add explicit per-command timeouts (e.g. `curl -m 20`)"
            );
        }
        Ok(out.to_string())
    }
}

/// Does this decoded stdout look like binary (NUL bytes / replacement chars /
/// control chars)? Used to route it to an artifact instead of the context.
pub(crate) fn is_binary_text(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    if s.contains('\0') {
        return true;
    }
    let total = s.chars().count();
    let weird = s
        .chars()
        .filter(|c| *c == '\u{FFFD}' || (*c < ' ' && *c != '\n' && *c != '\r' && *c != '\t'))
        .count();
    weird * 100 / total.max(1) >= 2
}

/// Persist a binary stdout blob and return a bounded, model-facing JSON value.
pub(crate) fn binary_output_json(
    store: &crate::artifacts::ArtifactStore,
    label: &str,
    text: &str,
) -> Value {
    let bytes = text.as_bytes();
    let preview = crate::artifacts::hex_preview(bytes, 64);
    match store.put_bytes(bytes, None, label) {
        Ok(r) => json!({
            "binary": true,
            "bytes": r.bytes,
            "mime": r.mime,
            "artifact": r.path,
            "preview_hex": preview,
            "truncated": r.bytes > 64,
            "hint": "Output is binary; saved as an artifact. Inspect a bounded sample with run_shell (e.g. `xxd -l 64 <path>` / `od -N 64 -tx1 <path>` / `head -c 64 <path>`), or read the source file directly.",
        }),
        Err(_) => json!({
            "binary": true,
            "bytes": bytes.len(),
            "preview_hex": preview,
            "truncated": bytes.len() > 64,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::super::backend::{FastshellBackend, NativeShell};

    #[test]
    fn binary_stdout_detected_and_artifacted() {
        assert!(is_binary_text("a\0b"));
        assert!(is_binary_text(&"\u{FFFD}".repeat(50)));
        assert!(!is_binary_text("normal text\nwith lines\n"));
        let dir = std::env::temp_dir().join(format!("aacode_shellbin_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let store = crate::artifacts::ArtifactStore::new(&dir);
        let v = binary_output_json(&store, "stdout", "PNG\u{FFFD}\u{FFFD}\u{FFFD}data");
        assert_eq!(v["binary"], true);
        assert!(v.get("artifact").and_then(|x| x.as_str()).is_some());
        assert!(v["preview_hex"].as_str().unwrap().len() > 4);
        assert!(v["truncated"].is_boolean());
    }
    use super::*;
    use fastshell::{Config, Fastshell};
    use std::sync::Mutex;

    fn native_tool(cwd: std::path::PathBuf) -> ShellTool {
        ShellTool::new(
            Arc::new(NativeShell::new()),
            cwd,
            24000,
            500,
            30,
            30,
            SafetyConfig::default(),
        )
    }

    fn tmp() -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "aacode_shelltool_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[tokio::test]
    async fn native_echo_roundtrip() {
        let dir = tmp();
        let t = native_tool(dir);
        let cancel = AtomicBool::new(false);
        let out = t
            .call(&json!({"command": "echo hello"}), &cancel)
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["success"], true);
        assert_eq!(v["returncode"], 0);
        assert_eq!(v["backend"], "native");
        assert!(v["stdout"].as_str().unwrap().contains("hello"));
    }

    #[tokio::test]
    async fn native_writes_to_real_cwd() {
        let dir = tmp();
        let t = native_tool(dir.clone());
        let cancel = AtomicBool::new(false);
        t.call(&json!({"command": "echo content > note.txt"}), &cancel)
            .await
            .unwrap();
        assert!(dir.join("note.txt").exists());
    }

    #[tokio::test]
    async fn native_heredoc_via_tool() {
        let dir = tmp();
        let t = native_tool(dir);
        let cancel = AtomicBool::new(false);
        let out = t
            .call(
                &json!({"command": "cat > x.txt <<'EOF'\nhi\nEOF\ncat x.txt"}),
                &cancel,
            )
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["stdout"].as_str().unwrap().contains("hi"));
    }

    #[tokio::test]
    async fn max_output_truncation() {
        let dir = tmp();
        let t = native_tool(dir);
        let cancel = AtomicBool::new(false);
        let out = t.call(&json!({"command": "for i in $(seq 1 500); do echo linelineline; done", "max_output": 100}), &cancel).await.unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["stdout"].as_str().unwrap().len() < 5000);
    }

    #[tokio::test]
    async fn max_output_lines_caps_lines() {
        let dir = tmp();
        let t = native_tool(dir);
        let cancel = AtomicBool::new(false);
        let out = t
            .call(
                &json!({
                    "command": "for i in $(seq 1 200); do echo l$i; done",
                    "max_output": 100000,
                    "max_output_lines": 10
                }),
                &cancel,
            )
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        let s = v["stdout"].as_str().unwrap();
        let content = s.split("\n\n[output truncated").next().unwrap();
        assert!(
            content.lines().count() <= 10,
            "expected <=10 lines, got {}: {content}",
            content.lines().count()
        );
        assert!(s.contains("truncated"));
    }

    /// With no configured cap (0 = unset) `run_shell` now applies the default
    /// 20k-char / 500-line budget and archives the full output; `full_output`
    /// opts out.
    #[tokio::test]
    async fn default_output_is_capped_and_full_output_opts_out() {
        let dir = tmp();
        let t = ShellTool::new(
            Arc::new(NativeShell::new()),
            dir,
            0,
            0,
            30,
            30,
            SafetyConfig::default(),
        );
        let cancel = AtomicBool::new(false);

        let capped = t
            .call(
                &json!({"command": "for i in $(seq 1 5000); do echo linelineline; done"}),
                &cancel,
            )
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&capped).unwrap();
        let s = v["stdout"].as_str().unwrap();
        assert!(s.contains("truncated"), "default cap should truncate: {s}");
        assert!(s.len() <= 20_000 + 500, "unexpected size {}", s.len());

        let full = t
            .call(
                &json!({
                    "command": "for i in $(seq 1 5000); do echo linelineline; done",
                    "full_output": true
                }),
                &cancel,
            )
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&full).unwrap();
        let s = v["stdout"].as_str().unwrap();
        assert!(
            s.len() > 50_000,
            "full_output should keep everything, len={}",
            s.len()
        );
        assert!(!s.contains("truncated"));
    }

    #[tokio::test]
    async fn dangerous_reject_mode() {
        let dir = tmp();
        let mut safety = SafetyConfig::default();
        safety.dangerous_command_action = DangerAction::Reject;
        let t = ShellTool::new(
            Arc::new(NativeShell::new()),
            dir,
            24000,
            500,
            30,
            30,
            safety,
        );
        let cancel = AtomicBool::new(false);
        let out = t
            .call(&json!({"command": "rm -rf /"}), &cancel)
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["success"], false);
    }

    #[tokio::test]
    async fn unknown_command_reject_mode() {
        let dir = tmp();
        let mut safety = SafetyConfig::default();
        safety.unknown_command_action = crate::config::UnknownAction::Reject;
        let t = ShellTool::new(
            Arc::new(NativeShell::new()),
            dir,
            24000,
            500,
            30,
            30,
            safety,
        );
        let cancel = AtomicBool::new(false);
        let out = t
            .call(
                &json!({"command": "definitely_not_a_command_zzz --flag"}),
                &cancel,
            )
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["success"], false);
        assert!(v["error"].as_str().unwrap().contains("unknown command"));
    }

    #[tokio::test]
    async fn fastshell_backend_still_works() {
        let dir = tmp();
        let mut fs = Fastshell::new();
        let mut cfg = Config::default();
        cfg.sandbox_path = dir.to_string_lossy().to_string();
        cfg.python_enabled = false;
        fs.init(cfg).unwrap();
        let backend = Arc::new(FastshellBackend::new(Arc::new(Mutex::new(fs))));
        let t = ShellTool::new(backend, dir, 24000, 500, 30, 30, SafetyConfig::default());
        let cancel = AtomicBool::new(false);
        let out = t
            .call(&json!({"command": "echo sandboxed"}), &cancel)
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["backend"], "fastshell");
        assert!(v["stdout"].as_str().unwrap().contains("sandboxed"));
    }

    #[test]
    fn is_dangerous_detects() {
        assert!(ShellTool::is_dangerous("rm -rf /"));
        assert!(ShellTool::is_dangerous("sudo rm -rf /"));
        assert!(ShellTool::is_dangerous("bash -c \"rm -rf /\""));
        assert!(ShellTool::is_dangerous("bash <<'EOF'\nrm -rf /\nEOF"));
        assert!(ShellTool::is_dangerous("find . -exec rm -rf / \\;"));
        assert!(!ShellTool::is_dangerous("ls -la"));
        // Data commands: quoted arguments are not commands.
        assert!(!ShellTool::is_dangerous("echo \"rm -rf /\""));
        assert!(!ShellTool::is_dangerous("grep \"rm -rf /\" file.txt"));
        // A heredoc body written to a file is data, not an executed command.
        assert!(!ShellTool::is_dangerous(
            "cat > x.sh <<'EOF'\nrm -rf /\nEOF"
        ));
        assert!(!ShellTool::is_dangerous("find . -name 'rm -rf /'"));
    }
}
