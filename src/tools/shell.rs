// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! `run_shell` tool — executes commands through a pluggable `ShellBackend`.
//!
//! On desktop the default backend is the real OS shell (`NativeShell`); on
//! mobile it is the fastshell sandbox engine (`FastshellBackend`). See
//! `backend.rs`. Returns a JSON string with `{success, returncode, stdout,
//! stderr, command}`. Oversized output is archived under `.aacode/extracts/`.

use super::backend::ShellBackend;
use super::jobs::{JobSnapshot, JobStore};
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
    jobs: Option<Arc<JobStore>>,
    background_timeout_secs: u64,
    foreground_max_secs: u64,
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
            jobs: None,
            background_timeout_secs: 1_800,
            foreground_max_secs: 600,
        }
    }

    /// Share the registry used by `job_*` tools so `run_shell` can return a
    /// job id on foreground timeout and support explicit background runs.
    pub fn with_jobs(
        mut self,
        jobs: Arc<JobStore>,
        background_timeout_secs: u64,
        foreground_max_secs: u64,
    ) -> Self {
        self.jobs = Some(jobs);
        self.background_timeout_secs = background_timeout_secs.max(1);
        self.foreground_max_secs = foreground_max_secs.max(1);
        self
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

fn snapshot_to_cmd_output(s: JobSnapshot) -> super::backend::CmdOutput {
    super::backend::CmdOutput {
        stdout: s.stdout,
        stderr: s.stderr,
        exit_code: s.exit_code,
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
                    "Max seconds with no output before escalating (idle timeout). Default: 120.",
                    &["idle_time", "no_output_timeout"],
                ),
                ToolParameter::new(
                    "background",
                    ParamType::Boolean,
                    false,
                    "Run in the background and return a job id immediately. Use for builds, tests, downloads, or servers.",
                    &["run_in_background", "async"],
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

        let background = args
            .get("background")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let foreground_timeout = timeout.min(self.foreground_max_secs).max(1);
        let stdin_owned = stdin_input.map(|s| s.to_string());
        let backend = self.backend.clone();
        let cwd = self.cwd.clone();
        let command2 = command.clone();
        let use_jobs = self.jobs.is_some() && stdin_owned.is_none();
        if background && stdin_owned.is_some() {
            return Ok(json!({
                "success": false,
                "error": "background mode does not support stdin_input; use a foreground command or provide input from a file.",
            })
            .to_string());
        }
        let result = if use_jobs {
            let store = self.jobs.as_ref().expect("job store checked");
            let job_id = store
                .start(
                    backend,
                    command2,
                    cwd,
                    self.background_timeout_secs,
                    idle_timeout,
                )
                .map_err(AacodeError::Tool)?;
            if background {
                return Ok(json!({
                    "success": true,
                    "status": "running",
                    "job_id": job_id,
                    "hint": "Background job started. Use job_output/job_status; do not start a duplicate command.",
                })
                .to_string());
            }
            let snapshot = store.wait_for(&job_id, foreground_timeout).await;
            match snapshot {
                Some(s) if s.status != "running" => {
                    store.remove(&job_id);
                    snapshot_to_cmd_output(s)
                }
                _ => {
                    return Ok(json!({
                        "success": true,
                        "status": "backgrounded",
                        "job_id": job_id,
                        "elapsed_ms": foreground_timeout * 1000,
                        "hint": "The command exceeded its foreground wait and is still running. Continue other work and collect it with job_output/job_status; do not start a duplicate command.",
                    })
                    .to_string());
                }
            }
        } else {
            tokio::task::spawn_blocking(move || {
                backend.run(
                    &command2,
                    stdin_owned.as_deref(),
                    timeout,
                    idle_timeout,
                    &cwd,
                )
            })
            .await
            .map_err(|e| AacodeError::Other(format!("shell spawn: {e}")))?
        };

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
            "cwd": self
                .backend
                .logical_cwd()
                .unwrap_or_else(|| self.cwd.display().to_string()),
            "physical_cwd": self.cwd.display().to_string(),
        });
        if result.exit_code != 0 {
            let (kind, hint) = classify_shell_exit(result.exit_code, &command, &result.stderr);
            out["error_kind"] = json!(kind);
            out["hint"] = json!(hint);
            if matches!(kind, "no_match" | "differences" | "condition_false") {
                // These are useful results, not tool failures. Keep the raw
                // returncode but let the model continue instead of retrying.
                out["success"] = json!(true);
                out["expected_nonzero"] = json!(true);
            }
        } else if result.stdout.trim().is_empty() && result.stderr.trim().is_empty() {
            if let Some(hint) = empty_output_hint(&command) {
                out["empty_output"] = json!(true);
                out["hint"] = json!(hint);
            }
        }
        Ok(out.to_string())
    }
}

/// Classify a non-zero shell exit without pretending every non-zero code is a
/// tool bug (`grep`/`diff`/`test` use exit 1 as a normal result).
pub(crate) fn classify_shell_exit(
    exit_code: i32,
    command: &str,
    stderr: &str,
) -> (&'static str, String) {
    let lower = command.to_lowercase();
    let stderr_lower = stderr.to_lowercase();
    let has = |name: &str| {
        lower.split_whitespace().any(|w| {
            w.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-') == name
        })
    };
    if stderr_lower.contains("traceback (most recent call last)") {
        let hint = if stderr_lower.contains("keyerror") {
            "Python KeyError: the requested field/key is missing; inspect the available keys first"
        } else if stderr_lower.contains("modulenotfounderror")
            || stderr_lower.contains("importerror")
        {
            "Python import failed; the module is unavailable in this sandbox"
        } else if stderr_lower.contains("filenotfounderror") {
            "Python file not found; verify the path or download result"
        } else if stderr_lower.contains("syntaxerror") {
            "Python syntax error; fix the script before retrying"
        } else {
            "Python raised an exception; inspect the traceback before retrying"
        };
        return ("python_exception", hint.to_string());
    }
    if has("curl") && !lower.contains("-ss") && lower.contains("-s") && stderr.trim().is_empty() {
        return (
            "network_connect",
            "curl used -s and suppressed the error; retry with -sS to see the real network error"
                .into(),
        );
    }
    if (lower.contains(";") || lower.contains("&&") || lower.contains("||"))
        && (stderr_lower.contains("not found") || stderr_lower.contains("no such file"))
    {
        return (
            "command_chain",
            "a later command failed; an earlier command in the chain may have failed first. Use && for dependent steps and check the earlier exit code.".into(),
        );
    }
    match exit_code {
        124 => ("timeout", "command timed out; split it into smaller steps or increase timeout".into()),
        130 => ("cancelled", "command was cancelled".into()),
        126 => ("permission_denied", "command is not executable or permission was denied".into()),
        127 => ("command_not_found", "command was not found; check the command name/path".into()),
        6 if has("curl") => ("dns", "curl could not resolve the host; check DNS/network".into()),
        7 if has("curl") => ("network_connect", "curl could not connect to the host; check network/reachability".into()),
        28 if has("curl") || has("wget") => ("timeout", "network request timed out".into()),
        22 if has("curl") => ("http_status", "curl received an HTTP error status; inspect the URL/server response".into()),
        1 if has("grep") || has("egrep") || has("fgrep") || has("rg") => (
            "no_match",
            "no matching lines were found (grep/rg exit 1 is a normal result)".into(),
        ),
        1 if has("diff") => ("differences", "files differ (diff exit 1 is a normal result)".into()),
        1 if has("test") || has("[") => ("condition_false", "test condition evaluated to false".into()),
        _ if stderr.trim().is_empty() => (
            "exit_nonzero",
            format!("command exited with code {exit_code} but wrote no stderr; inspect stdout and the command"),
        ),
        _ => ("command_failed", "command failed; inspect stderr and exit code".into()),
    }
}

/// Explain a successful command that printed nothing because Windows cmd.exe
/// mangles a multi-line `python -c` / `node -e` snippet. The model should write
/// a temporary script instead of guessing what happened.
pub(crate) fn empty_output_hint(command: &str) -> Option<&'static str> {
    if !cfg!(target_os = "windows") || !command.contains('\n') {
        return None;
    }
    let lower = command.to_lowercase();
    let interpreters = [
        "python -c",
        "python3 -c",
        "py -c",
        "node -e",
        "perl -e",
        "ruby -e",
    ];
    if !interpreters.iter().any(|token| lower.contains(token)) {
        return None;
    }
    Some(
        "Command exited 0 but printed nothing. Multi-line code passed to python -c / node -e is unreliable in cmd.exe; write a temporary script file and run it instead.",
    )
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
    let total = s.chars().count().max(1);
    let replacement = s.chars().filter(|c| *c == '\u{FFFD}').count();
    let controls = s
        .chars()
        .filter(|c| c.is_control() && *c != '\n' && *c != '\r' && *c != '\t')
        .count();
    // `cat -A` and similar diagnostics contain a few control glyphs but are
    // still useful text. Only archive when bad characters dominate.
    replacement * 100 / total >= 5 || controls * 100 / total >= 10
}

/// Persist a binary stdout blob and return a bounded, model-facing JSON value.
pub(crate) fn binary_output_json(
    store: &crate::artifacts::ArtifactStore,
    label: &str,
    text: &str,
) -> Value {
    let bytes = text.as_bytes();
    let preview = crate::artifacts::hex_preview(bytes, 64);
    let preview_text: String = text
        .chars()
        .take(240)
        .map(|c| {
            if c.is_control() && c != '\n' && c != '\t' {
                ' '
            } else {
                c
            }
        })
        .collect();
    match store.put_bytes(bytes, None, label) {
        Ok(r) => json!({
            "binary": true,
            "bytes": r.bytes,
            "mime": r.mime,
            "artifact": r.path,
            "preview_hex": preview,
            "preview_text": preview_text,
            "truncated": r.bytes > 64,
            "hint": "Output is binary; saved as an artifact. Inspect a bounded sample with run_shell (e.g. `xxd -l 64 <path>` / `od -N 64 -tx1 <path>` / `head -c 64 <path>`), or read the source file directly.",
        }),
        Err(_) => json!({
            "binary": true,
            "bytes": bytes.len(),
            "preview_hex": preview,
            "preview_text": preview_text,
            "truncated": bytes.len() > 64,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::super::backend::{FastshellBackend, NativeShell};

    #[test]
    fn a_few_control_chars_are_still_text() {
        let mut s = "normal text\n".repeat(100);
        s.push('\u{000e}');
        assert!(!is_binary_text(&s));
        let b = "text\u{000e}".repeat(20);
        assert!(is_binary_text(&b));
    }

    #[test]
    fn shell_exit_classification_is_tool_aware() {
        assert_eq!(
            classify_shell_exit(7, "curl -s https://x", "").0,
            "network_connect"
        );
        assert_eq!(classify_shell_exit(1, "grep -q foo f", "").0, "no_match");
        assert_eq!(classify_shell_exit(1, "diff a b", "").0, "differences");
        assert_eq!(
            classify_shell_exit(1, "test -f nope", "").0,
            "condition_false"
        );
        assert_eq!(
            classify_shell_exit(127, "nope-cmd", "").0,
            "command_not_found"
        );
        assert_eq!(classify_shell_exit(9, "echo", "").0, "exit_nonzero");
        assert_eq!(
            classify_shell_exit(
                1,
                "python3 script.py",
                "Traceback (most recent call last):\nKeyError: 'x'"
            )
            .0,
            "python_exception"
        );
        assert_eq!(
            classify_shell_exit(1, "curl -s https://x", "").0,
            "network_connect"
        );
        assert_eq!(
            classify_shell_exit(
                1,
                "curl -s https://x -o /tmp/a; head /tmp/a",
                "head: /tmp/a: Not found"
            )
            .0,
            "command_chain"
        );
    }

    #[test]
    fn empty_multiline_python_on_windows_has_hint() {
        let command = "python -c \"\nprint(1)\n\"";
        if cfg!(target_os = "windows") {
            assert!(super::empty_output_hint(command)
                .unwrap()
                .contains("temporary script"));
        } else {
            assert!(super::empty_output_hint(command).is_none());
        }
        assert!(super::empty_output_hint("python -c \"print(1)\"").is_none());
        assert!(super::empty_output_hint("mkdir foo").is_none());
    }

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
    async fn background_job_returns_id_and_quick_commands_stay_foreground() {
        let dir = tmp();
        let store = Arc::new(crate::tools::jobs::JobStore::new(4));
        let tool = ShellTool::new(
            Arc::new(NativeShell::new()),
            dir,
            24000,
            500,
            1,
            0,
            SafetyConfig::default(),
        )
        .with_jobs(store.clone(), 5, 2);

        let quick = tool
            .call(&json!({"command": "printf ok"}), &AtomicBool::new(false))
            .await
            .unwrap();
        assert!(quick.contains("ok"));
        assert!(
            store.list().is_empty(),
            "completed foreground job must be removed"
        );

        let started = tool
            .call(
                &json!({"command": "sleep 2", "background": true}),
                &AtomicBool::new(false),
            )
            .await
            .unwrap();
        let value: Value = serde_json::from_str(&started).unwrap();
        assert_eq!(value["status"], "running");
        assert!(value["job_id"].as_str().unwrap().starts_with("job_"));
    }

    #[tokio::test]
    async fn foreground_timeout_promotes_to_background_job() {
        let dir = tmp();
        let store = Arc::new(crate::tools::jobs::JobStore::new(4));
        let tool = ShellTool::new(
            Arc::new(NativeShell::new()),
            dir,
            24000,
            500,
            1,
            0,
            SafetyConfig::default(),
        )
        .with_jobs(store.clone(), 5, 1);

        let result = tool
            .call(&json!({"command": "sleep 2"}), &AtomicBool::new(false))
            .await
            .unwrap();
        let value: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(value["status"], "backgrounded");
        let id = value["job_id"].as_str().unwrap();
        let done = store.wait_for(id, 5).await.unwrap();
        assert_ne!(done.status, "running");
    }

    #[tokio::test]
    async fn background_rejects_stdin_input() {
        let dir = tmp();
        let store = Arc::new(crate::tools::jobs::JobStore::new(2));
        let tool = ShellTool::new(
            Arc::new(NativeShell::new()),
            dir,
            24000,
            500,
            1,
            0,
            SafetyConfig::default(),
        )
        .with_jobs(store, 5, 2);
        let out = tool
            .call(
                &json!({"command": "cat", "background": true, "stdin_input": "x"}),
                &AtomicBool::new(false),
            )
            .await
            .unwrap();
        let value: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(value["success"], false);
        assert!(value["error"].as_str().unwrap().contains("stdin_input"));
    }

    #[tokio::test]
    async fn native_echo_roundtrip() {
        let dir = tmp();
        let t = native_tool(dir.clone());
        let cancel = AtomicBool::new(false);
        let out = t
            .call(&json!({"command": "echo hello"}), &cancel)
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["success"], true);
        assert_eq!(v["returncode"], 0);
        assert_eq!(v["backend"], "native");
        assert_eq!(v["cwd"].as_str().unwrap(), dir.to_string_lossy());
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
    async fn fastshell_error_semantics_and_utf8_file() {
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
            .call(
                &json!({"command": "printf 'nothing' > f.txt && grep -q needle f.txt"}),
                &cancel,
            )
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["error_kind"], "no_match", "{v}");
        let out = t
            .call(
                &json!({"command": "printf a > a.txt && printf b > b.txt && diff a.txt b.txt"}),
                &cancel,
            )
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["error_kind"], "differences", "{v}");
        let out = t
            .call(&json!({"command": "test -f missing.txt"}), &cancel)
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["error_kind"], "condition_false", "{v}");
        let out = t
            .call(
                &json!({"command": "printf '中文😀' > cn.txt && file cn.txt"}),
                &cancel,
            )
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(
            v["stdout"].as_str().unwrap().contains("Unicode text"),
            "{v}"
        );
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
