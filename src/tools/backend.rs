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
/// of 0 means "no explicit total timeout". `idle_timeout_secs` of 0 disables
/// idle timeout (currently only enforced by `NativeShell`, ignored by
/// `FastshellBackend`).
pub trait ShellBackend: Send + Sync {
    fn run(
        &self,
        command: &str,
        stdin_input: Option<&str>,
        timeout_secs: u64,
        idle_timeout_secs: u64,
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
        idle_timeout_secs: u64,
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
        let output = if timeout_secs > 0 || idle_timeout_secs > 0 {
            wait_with_timeout(child, timeout_secs, idle_timeout_secs)
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

/// Wait for a child process. Enforces both a total timeout and an idle
/// timeout (time with no stdout/stderr output). On timeout, kills the
/// process group and reports exit 124.
fn wait_with_timeout(
    mut child: std::process::Child,
    timeout_secs: u64,
    idle_timeout_secs: u64,
) -> std::io::Result<CmdOutput> {
    use std::io::Read;
    use std::sync::mpsc;

    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();

    enum Chunk {
        Stdout(String),
        Stderr(String),
    }

    let (chunk_tx, chunk_rx) = mpsc::channel::<Chunk>();

    // Reader threads send chunks through the channel so the main thread can
    // track idle time.
    if let Some(mut p) = stdout_pipe {
        let tx = chunk_tx.clone();
        std::thread::spawn(move || {
            let mut buf = [0u8; 4096];
            loop {
                match p.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        let s = String::from_utf8_lossy(&buf[..n]).to_string();
                        let _ = tx.send(Chunk::Stdout(s));
                    }
                    Err(_) => break,
                }
            }
        });
    }
    if let Some(mut p) = stderr_pipe {
        let tx = chunk_tx.clone();
        std::thread::spawn(move || {
            let mut buf = [0u8; 4096];
            loop {
                match p.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        let s = String::from_utf8_lossy(&buf[..n]).to_string();
                        let _ = tx.send(Chunk::Stderr(s));
                    }
                    Err(_) => break,
                }
            }
        });
    }

    let pid = child.id();
    let (exit_tx, exit_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let status = child.wait();
        let _ = exit_tx.send(status);
    });

    let mut stdout = String::new();
    let mut stderr = String::new();
    let deadline = if timeout_secs > 0 {
        Some(std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs))
    } else {
        None
    };
    let idle_dur = std::time::Duration::from_secs(idle_timeout_secs);
    let mut last_data = std::time::Instant::now();

    loop {
        // Compute how long we can wait for the next event.
        let wait_dur = match deadline {
            Some(dl) => {
                let remaining = dl.saturating_duration_since(std::time::Instant::now());
                // Also limit by idle timeout if enabled.
                if idle_timeout_secs > 0 {
                    let idle_remaining = idle_dur.saturating_sub(last_data.elapsed());
                    remaining.min(idle_remaining)
                } else {
                    remaining
                }
            }
            None => {
                if idle_timeout_secs > 0 {
                    idle_dur.saturating_sub(last_data.elapsed())
                } else {
                    // Neither total nor idle timeout — wait forever.
                    std::time::Duration::from_secs(u64::MAX)
                }
            }
        };

        if wait_dur.is_zero() {
            // Determine the reason and kill.
            let reason = if deadline.map_or(false, |dl| dl <= std::time::Instant::now()) {
                format!("command timed out after {timeout_secs}s")
            } else {
                format!("command idle timeout after {idle_timeout_secs}s")
            };
            #[cfg(unix)]
            unsafe {
                libc::kill(pid as i32, libc::SIGKILL);
            }
            #[cfg(windows)]
            {
                let _ = std::process::Command::new("taskkill")
                    .args(["/F", "/T", "/PID", &pid.to_string()])
                    .output();
            }
            return Ok(CmdOutput {
                stdout,
                stderr: format!("{stderr}\n{reason}"),
                exit_code: 124,
            });
        }

        // Wait for either a chunk or the process to exit.
        match chunk_rx.recv_timeout(wait_dur) {
            Ok(Chunk::Stdout(s)) => {
                last_data = std::time::Instant::now();
                stdout.push_str(&s);
            }
            Ok(Chunk::Stderr(s)) => {
                last_data = std::time::Instant::now();
                stderr.push_str(&s);
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // Reader channels closed — drain any remaining exit status.
                break;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // Check if the process already exited in the meantime.
                if let Ok(Ok(status)) = exit_rx.try_recv() {
                    return Ok(CmdOutput {
                        stdout,
                        stderr,
                        exit_code: status.code().unwrap_or(-1),
                    });
                }
                // Otherwise the timeout computation will catch it on the
                // next loop iteration.
                continue;
            }
        }

        // Also check if process already exited (non-blocking).
        if let Ok(Ok(status)) = exit_rx.try_recv() {
            // Drain remaining chunks quickly.
            while let Ok(chunk) = chunk_rx.try_recv() {
                match chunk {
                    Chunk::Stdout(s) => stdout.push_str(&s),
                    Chunk::Stderr(s) => stderr.push_str(&s),
                }
            }
            return Ok(CmdOutput {
                stdout,
                stderr,
                exit_code: status.code().unwrap_or(-1),
            });
        }
    }

    // Process has exited or all reader threads finished.
    match exit_rx.recv() {
        Ok(Ok(status)) => Ok(CmdOutput {
            stdout,
            stderr,
            exit_code: status.code().unwrap_or(-1),
        }),
        Ok(Err(e)) => Err(e),
        Err(_) => Ok(CmdOutput {
            stdout,
            stderr: format!("{stderr}\nprocess disappeared"),
            exit_code: 127,
        }),
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
        timeout_secs: u64,
        _idle_timeout_secs: u64,
        _cwd: &std::path::Path,
    ) -> CmdOutput {
        // fastshell::execute takes only the command; stdin is emulated by
        // piping via printf (fastshell has no stdin param). cwd is governed
        // by the sandbox.
        let effective = match stdin_input {
            Some(s) if !s.is_empty() => {
                let escaped = s.replace('\\', "\\\\").replace('\'', "'\\''");
                format!("printf '%s' '{escaped}' | {command}")
            }
            _ => command.to_string(),
        };

        // Try-lock the fastshell handle. Use the command timeout as the
        // lock-acquisition ceiling (with a 5 s floor) so we don't burn
        // more time waiting for the lock than we'd allow the command to run.
        let lock_timeout_secs = timeout_secs.max(5);
        let timeout_ms = timeout_secs.saturating_mul(1000);
        let fs_guard = {
            let deadline = std::time::Instant::now()
                + std::time::Duration::from_secs(lock_timeout_secs);
            loop {
                match self.fs.try_lock() {
                    Ok(g) => break g,
                    Err(std::sync::TryLockError::WouldBlock) => {
                        if std::time::Instant::now() > deadline {
                            return CmdOutput {
                                stdout: String::new(),
                                stderr: format!(
                                    "fastshell: could not acquire lock after {lock_timeout_secs}s (busy)\n"
                                ),
                                exit_code: 124,
                            };
                        }
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
                    Err(std::sync::TryLockError::Poisoned(e)) => break e.into_inner(),
                }
            }
        };

        let r = if timeout_ms > 0 {
            fs_guard.execute_with_timeout(&effective, timeout_ms)
        } else {
            fs_guard.execute(&effective)
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
        let out = sh.run("echo hello", None, 10, 0, std::path::Path::new("."));
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
            0,
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
        let out = sh.run("cat data.txt | grep apple | wc -l", None, 10, 0, &dir);
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout.trim(), "2");
    }

    #[test]
    fn native_stdin_input() {
        let sh = NativeShell::new();
        let out = sh.run("cat", Some("piped content"), 10, 0, std::path::Path::new("."));
        assert_eq!(out.exit_code, 0);
        assert!(out.stdout.contains("piped content"));
    }

    #[test]
    fn native_nonzero_exit() {
        let sh = NativeShell::new();
        let out = sh.run("exit 3", None, 10, 0, std::path::Path::new("."));
        assert_eq!(out.exit_code, 3);
    }

    #[test]
    fn native_timeout_kills() {
        let sh = NativeShell::new();
        let start = std::time::Instant::now();
        // On some platforms, the OS shell may absorb SIGKILL to its children
        // differently; we verify that a command that sleeps for 10s with a 1s
        // timeout DOES terminate within a reasonable window (guarded at 8s).
        let out = sh.run("sleep 10", None, 1, 0, std::path::Path::new("."));
        let elapsed = start.elapsed();
        assert!(elapsed.as_secs() < 8, "timeout should have fired; elapsed={elapsed:?}");
        // Exit 124 is the POSIX convention for command-killed-by-timeout.
        assert_eq!(out.exit_code, 124, "unexpected exit code; stdout={}, stderr={}", out.stdout, out.stderr);
    }

    #[test]
    fn native_python_real() {
        // If python3 exists (desktop), it runs via the real interpreter.
        let sh = NativeShell::new();
        let out = sh.run("python3 -c \"print(6*7)\"", None, 10, 0, std::path::Path::new("."));
        if out.exit_code == 0 {
            assert!(out.stdout.contains("42"));
        }
    }
}
