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
use crate::config::{DangerAction, SafetyConfig};
use crate::error::Result;
use serde_json::{json, Value};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

// Re-export the fastshell handle type for the rest of the crate.
pub use super::backend::SharedFastshell as SharedShell;

pub struct ShellTool {
    backend: Arc<dyn ShellBackend>,
    /// Working directory / archive root for command execution.
    cwd: std::path::PathBuf,
    max_output_chars: usize,
    default_timeout_secs: u64,
    safety: SafetyConfig,
}

impl ShellTool {
    pub fn new(
        backend: Arc<dyn ShellBackend>,
        cwd: std::path::PathBuf,
        max_output_chars: usize,
        default_timeout_secs: u64,
        safety: SafetyConfig,
    ) -> Self {
        ShellTool {
        // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0
            backend,
            cwd,
            max_output_chars,
            default_timeout_secs,
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

    /// Truncate a field, archiving the full content when too long.
    fn maybe_truncate(&self, text: String, prefix: &str, limit: usize) -> String {
        if limit == 0 || text.chars().count() <= limit {
            return text;
        }
        let head: String = text.chars().take(limit).collect();
        match self.archive(&text, prefix) {
            Some(path) => format!(
                "{head}\n\n[Full output ({} chars) saved to {path}. Use run_shell to grep/cat it.]",
                text.chars().count()
            ),
            None => format!("{head}\n\n[truncated, {} chars total]", text.chars().count()),
        }
    }

    /// Detect blatantly dangerous commands (used only in `reject` mode).
    fn is_dangerous(command: &str) -> bool {
        let c = command.to_lowercase();
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
}

impl Tool for ShellTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "run_shell",
            "Execute shell commands — the universal Swiss Army knife. Use for ALL file read/write/search operations. There is no write_file, read_file or edit_file tool — use run_shell + shell commands (cat/tail for reading, echo/sed/awk for writing/editing, grep/rg/find for searching, git/python/pytest/gcc/go etc). Supports pipes (|), redirection (>), heredocs (<< 'EOF'), chaining (&& / || / ;), command substitution ($(...)), and variable expansion ($VAR). Always returns a result object with stdout, stderr, and returncode — check returncode for success.",
            vec![
                ToolParameter::new(
                    "command",
                    ParamType::String,
                    true,
                    "The shell command to execute. For multi-line files, use heredoc: cat > file << 'EOF'\\n...\\nEOF. Supports pipes (|), redirection (>), chaining (&& / ;). Always quote filenames with spaces/special chars.",
                    &["cmd", "shell", "script", "exec"],
                ),
                ToolParameter::new(
                    "timeout",
                    ParamType::Integer,
                    false,
                    "Command timeout in seconds.",
                    &["time_limit", "max_time", "wait"],
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
                    "Limit returned output characters. Default: no limit.",
                    &["max_chars", "limit", "output_limit"],
                ),
            ],
        )
    }

    fn call(&self, args: &Value, _cancel: &AtomicBool) -> Result<String> {
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

        let stdin_input = args.get("stdin_input").and_then(|v| v.as_str());
        let timeout = args
            .get("timeout")
            .and_then(|v| v.as_u64())
            .unwrap_or(self.default_timeout_secs);

        let result = self
            .backend
            .run(&command, stdin_input, timeout, &self.cwd);

        // Per-call max_output override; else the tool's configured cap.
        let limit = args
            .get("max_output")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(self.max_output_chars);

        let stdout = self.maybe_truncate(result.stdout, "stdout", limit);
        let stderr = self.maybe_truncate(result.stderr, "stderr", limit);

        Ok(json!({
            "success": true,
            "returncode": result.exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "command": command,
            "backend": self.backend.kind(),
        })
        .to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::super::backend::{FastshellBackend, NativeShell};
    use super::*;
    use fastshell::{Config, Fastshell};
    use std::sync::Mutex;

    fn native_tool(cwd: std::path::PathBuf) -> ShellTool {
        ShellTool::new(
            Arc::new(NativeShell::new()),
            cwd,
            24000,
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

    #[test]
    fn native_echo_roundtrip() {
        let dir = tmp();
        let t = native_tool(dir);
        let cancel = AtomicBool::new(false);
        let out = t.call(&json!({"command": "echo hello"}), &cancel).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["success"], true);
        assert_eq!(v["returncode"], 0);
        assert_eq!(v["backend"], "native");
        assert!(v["stdout"].as_str().unwrap().contains("hello"));
    }

    #[test]
    fn native_writes_to_real_cwd() {
        let dir = tmp();
        let t = native_tool(dir.clone());
        let cancel = AtomicBool::new(false);
        t.call(&json!({"command": "echo content > note.txt"}), &cancel)
            .unwrap();
        // File exists at the REAL working directory, not a VFS jail.
        assert!(dir.join("note.txt").exists());
    }

    #[test]
    fn native_heredoc_via_tool() {
        let dir = tmp();
        let t = native_tool(dir);
        let cancel = AtomicBool::new(false);
        let out = t
            .call(
                &json!({"command": "cat > x.txt <<'EOF'\nhi\nEOF\ncat x.txt"}),
                &cancel,
            )
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["stdout"].as_str().unwrap().contains("hi"));
    }

    #[test]
    fn max_output_truncation() {
        let dir = tmp();
        let t = native_tool(dir);
        let cancel = AtomicBool::new(false);
        let out = t
            .call(
                &json!({"command": "for i in $(seq 1 500); do echo linelineline; done", "max_output": 100}),
                &cancel,
            )
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["stdout"].as_str().unwrap().len() < 5000);
    }

    #[test]
    fn dangerous_reject_mode() {
        let dir = tmp();
        let mut safety = SafetyConfig::default();
        safety.dangerous_command_action = DangerAction::Reject;
        let t = ShellTool::new(Arc::new(NativeShell::new()), dir, 24000, 30, safety);
        let cancel = AtomicBool::new(false);
        let out = t.call(&json!({"command": "rm -rf /"}), &cancel).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["success"], false);
    }

    #[test]
    fn fastshell_backend_still_works() {
        // The sandbox backend must remain functional (mobile path).
        let dir = tmp();
        let mut fs = Fastshell::new();
        let mut cfg = Config::default();
        cfg.sandbox_path = dir.to_string_lossy().to_string();
        cfg.python_enabled = false;
        fs.init(cfg).unwrap();
        let backend = Arc::new(FastshellBackend::new(Arc::new(Mutex::new(fs))));
        let t = ShellTool::new(backend, dir, 24000, 30, SafetyConfig::default());
        let cancel = AtomicBool::new(false);
        let out = t.call(&json!({"command": "echo sandboxed"}), &cancel).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["backend"], "fastshell");
        assert!(v["stdout"].as_str().unwrap().contains("sandboxed"));
    }

    #[test]
    fn is_dangerous_detects() {
        assert!(ShellTool::is_dangerous("rm -rf /"));
        assert!(!ShellTool::is_dangerous("ls -la"));
    }
}
