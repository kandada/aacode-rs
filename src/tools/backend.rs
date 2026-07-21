// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Shell execution backend abstraction.
//!
//! The agent's `run_shell` tool can run commands two ways:
//!
//!   * **NativeShell** (default on desktop: Linux/macOS/Windows) — spawns the
//!     real OS shell (`sh -c` / `cmd /C`). Full heredoc, pipes, real paths,
//!     real `grep`/`rg`/`git`, and the system `python3` all work exactly as the
//!     user expects. This is the natural behavior on a developer machine.
//!
//!   * **FastshellBackend** (default on mobile: Android/iOS) — routes through
//!     the `fastshell` sandbox engine (180+ built-in commands + embedded
//!     CPython) inside a VFS jail. Used where there is no usable OS shell.
//!
//! The choice is auto-detected by target OS but can be overridden by config.

use fastshell::Fastshell;
use std::sync::{Arc, Mutex};

/// Uniform result of a shell command.
#[derive(Debug, Clone)]
pub struct CmdOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

/// A command execution backend. `cwd` is the working directory; `timeout_secs`
/// of 0 means "no explicit timeout".
pub trait ShellBackend: Send + Sync {
    fn run(
        &self,
        command: &str,
        stdin_input: Option<&str>,
        timeout_secs: u64,
        cwd: &std::path::Path,
    ) -> CmdOutput;

    /// Human-readable backend name (for diagnostics).
    fn kind(&self) -> &'static str;
}

/// Which backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Real OS shell via std::process::Command.
    Native,
    /// fastshell sandbox engine.
    Fastshell,
}

impl BackendKind {
    /// Default backend for the current platform: native on desktop, fastshell
    /// on mobile (where there is no usable system shell).
    pub fn platform_default() -> BackendKind {
        if cfg!(any(target_os = "android", target_os = "ios")) {
            BackendKind::Fastshell
        } else {
            BackendKind::Native
        }
    }
}

// ───────────────────────────── Native backend ─────────────────────────────

/// Runs commands through the real operating-system shell.
pub struct NativeShell;

impl NativeShell {
    pub fn new() -> Self {
        NativeShell
    }
}

impl Default for NativeShell {
    fn default() -> Self {
        NativeShell::new()
    }
}

impl ShellBackend for NativeShell {
    fn run(
        &self,
        command: &str,
        stdin_input: Option<&str>,
        timeout_secs: u64,
        cwd: &std::path::Path,
    ) -> CmdOutput {
        use std::io::Write;
        use std::process::{Command, Stdio};

        // Pick the platform shell.
        let mut cmd = if cfg!(target_os = "windows") {
            let mut c = Command::new("cmd");
            c.arg("/C").arg(command);
            c
        } else {
            let mut c = Command::new("sh");
            c.arg("-c").arg(command);
            c
        };
        cmd.current_dir(cwd)
            .stdin(if stdin_input.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                return CmdOutput {
                    stdout: String::new(),
                    stderr: format!("failed to spawn shell: {e}"),
                    exit_code: 127,
                }
            }
        };

        // Feed stdin if provided.
        if let Some(input) = stdin_input {
            if let Some(mut stdin) = child.stdin.take() {
                let _ = stdin.write_all(input.as_bytes());
                // dropping stdin closes it (EOF)
            }
        }

        // Timeout handling: wait in a thread and enforce a deadline.
        let output = if timeout_secs > 0 {
            wait_with_timeout(child, timeout_secs)
        } else {
            child.wait_with_output().map(|o| CmdOutput {
                stdout: String::from_utf8_lossy(&o.stdout).to_string(),
                stderr: String::from_utf8_lossy(&o.stderr).to_string(),
                exit_code: o.status.code().unwrap_or(-1),
            })
        };

        output.unwrap_or_else(|e| CmdOutput {
            stdout: String::new(),
            stderr: format!("shell execution error: {e}"),
            exit_code: -1,
        })
    }

    fn kind(&self) -> &'static str {
        "native"
    }
}

/// Wait for a child with a timeout. On timeout, kill the process group and
/// report exit 124 (POSIX convention). Reader threads are abandoned — the
/// agent loop serialises commands so a new run will use a fresh process set.
fn wait_with_timeout(
    mut child: std::process::Child,
    timeout_secs: u64,
) -> std::io::Result<CmdOutput> {
    use std::sync::mpsc;
    let mut stdout_pipe = child.stdout.take();
    let mut stderr_pipe = child.stderr.take();

    let out_handle = std::thread::spawn(move || {
        let mut buf = String::new();
        if let Some(ref mut p) = stdout_pipe {
            use std::io::Read;
            let _ = p.read_to_string(&mut buf);
        }
        buf
    });
    let err_handle = std::thread::spawn(move || {
        let mut buf = String::new();
        if let Some(ref mut p) = stderr_pipe {
            use std::io::Read;
            let _ = p.read_to_string(&mut buf);
        }
        buf
    });

    let pid = child.id();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let status = child.wait();
        let _ = tx.send(status);
    });

    match rx.recv_timeout(std::time::Duration::from_secs(timeout_secs)) {
        Ok(Ok(status)) => {
            let stdout = out_handle.join().unwrap_or_default();
            let stderr = err_handle.join().unwrap_or_default();
            Ok(CmdOutput {
                stdout,
                stderr,
                exit_code: status.code().unwrap_or(-1),
            })
        }
        Ok(Err(e)) => Err(e),
        Err(_) => {
            // Kill the process group so reader threads unblock.
            #[cfg(unix)]
            {
                unsafe {
                    libc::kill(pid as i32, libc::SIGKILL);
                }
            }
            #[cfg(windows)]
            {
                // On Windows, PID can be used to terminate via Command with taskkill.
                let _ = std::process::Command::new("taskkill")
                    .args(["/F", "/T", "/PID", &pid.to_string()])
                    .output();
            }
            let stdout = out_handle.join().unwrap_or_default();
            let stderr = err_handle.join().unwrap_or_default();
            Ok(CmdOutput {
                stdout,
                stderr: format!("{stderr}\ncommand timed out after {timeout_secs}s"),
                exit_code: 124,
            })
        }
    }
}

// ─────────────────────────── Fastshell backend ────────────────────────────

/// Shared, thread-safe handle to the fastshell SDK.
pub type SharedFastshell = Arc<Mutex<Fastshell>>;

/// Routes commands through the fastshell sandbox engine.
pub struct FastshellBackend {
    fs: SharedFastshell,
}

impl FastshellBackend {
    pub fn new(fs: SharedFastshell) -> Self {
        FastshellBackend { fs }
    }
}

impl ShellBackend for FastshellBackend {
    fn run(
        &self,
        command: &str,
        stdin_input: Option<&str>,
        _timeout_secs: u64,
        _cwd: &std::path::Path,
    ) -> CmdOutput {
        // fastshell::execute takes only the command; stdin is emulated by
        // piping via printf (fastshell has no stdin param). Timeout is governed
        // by the fastshell Config (command_timeout_ms), and cwd by the sandbox.
        let effective = match stdin_input {
            Some(s) if !s.is_empty() => {
                let escaped = s.replace('\\', "\\\\").replace('\'', "'\\''");
                format!("printf '%s' '{escaped}' | {command}")
            }
            _ => command.to_string(),
        };
        let r = {
            let fs = self.fs.lock().unwrap_or_else(|e| e.into_inner());
            fs.execute(&effective)
        };
        CmdOutput {
            stdout: r.stdout,
            stderr: r.stderr,
            exit_code: r.exit_code,
        }
    }

    fn kind(&self) -> &'static str {
        "fastshell"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_default_is_native_on_desktop() {
        // On the CI/dev desktop target this must be Native.
        #[cfg(not(any(target_os = "android", target_os = "ios")))]
        assert_eq!(BackendKind::platform_default(), BackendKind::Native);
    }

    #[test]
    fn native_echo() {
        let sh = NativeShell::new();
        let out = sh.run("echo hello", None, 10, std::path::Path::new("."));
        assert_eq!(out.exit_code, 0);
        assert!(out.stdout.contains("hello"));
    }

    #[test]
    fn native_heredoc_works() {
        // The whole point: real shell features like heredoc must work natively.
        let dir = std::env::temp_dir().join(format!("native_hd_{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let sh = NativeShell::new();
        let out = sh.run(
            "cat > f.txt << 'EOF'\nline1\nline2\nEOF\ncat f.txt",
            None,
            10,
            &dir,
        );
        assert_eq!(out.exit_code, 0, "stderr={}", out.stderr);
        assert!(out.stdout.contains("line1"));
        assert!(out.stdout.contains("line2"));
        // File actually created at the real cwd (not a VFS path).
        assert!(dir.join("f.txt").exists());
    }

    #[test]
    fn native_real_cwd_and_pipes() {
        let dir = std::env::temp_dir().join(format!("native_cwd_{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("data.txt"), "apple\nbanana\napple\n").unwrap();
        let sh = NativeShell::new();
        let out = sh.run("cat data.txt | grep apple | wc -l", None, 10, &dir);
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout.trim(), "2");
    }

    #[test]
    fn native_stdin_input() {
        let sh = NativeShell::new();
        let out = sh.run("cat", Some("piped content"), 10, std::path::Path::new("."));
        assert_eq!(out.exit_code, 0);
        assert!(out.stdout.contains("piped content"));
    }

    #[test]
    fn native_nonzero_exit() {
        let sh = NativeShell::new();
        let out = sh.run("exit 3", None, 10, std::path::Path::new("."));
        assert_eq!(out.exit_code, 3);
    }

    #[test]
    fn native_timeout_kills() {
        let sh = NativeShell::new();
        let start = std::time::Instant::now();
        // On some platforms, the OS shell may absorb SIGKILL to its children
        // differently; we verify that a command that sleeps for 10s with a 1s
        // timeout DOES terminate within a reasonable window (guarded at 8s).
        let out = sh.run("sleep 10", None, 1, std::path::Path::new("."));
        let elapsed = start.elapsed();
        assert!(elapsed.as_secs() < 8, "timeout should have fired; elapsed={elapsed:?}");
        // Exit 124 is the POSIX convention for command-killed-by-timeout.
        assert_eq!(out.exit_code, 124, "unexpected exit code; stdout={}, stderr={}", out.stdout, out.stderr);
    }

    #[test]
    fn native_python_real() {
        // If python3 exists (desktop), it runs via the real interpreter.
        let sh = NativeShell::new();
        let out = sh.run("python3 -c \"print(6*7)\"", None, 10, std::path::Path::new("."));
        if out.exit_code == 0 {
            assert!(out.stdout.contains("42"));
        }
    }
}
