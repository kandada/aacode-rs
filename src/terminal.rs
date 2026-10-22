// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Interactive terminal sessions (real PTY on Unix).
//!
//! This is a **host/user** API, not an agent tool: the model does not get input
//! access by default (see the safety discussion in `docs`). Sessions are
//! bounded, process-local, and never persisted.
//!
//! - Unix: a real PTY (`nix::pty::openpty`) running the requested shell.
//! - Windows / mobile: not supported here (`start` returns an error).
//!
//! Lifecycle: `start → write/read/resize → close`. Output is kept in a bounded
//! ring (256 KiB) with an absolute cursor; `read` reports `truncated` when the
//! cursor fell behind the retained window. Process exit is reported lazily via
//! `try_wait` on `read`.

#![cfg_attr(not(feature = "pty"), allow(dead_code))]

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::Instant;

pub const MAX_SESSIONS: usize = 2;
pub const MAX_BUFFER_BYTES: usize = 256 * 1024;

#[derive(Debug, Clone)]
pub struct ReadResult {
    pub data: Vec<u8>,
    pub cursor: u64,
    pub eof: bool,
    pub exit_code: Option<i32>,
    pub truncated: bool,
}

#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub id: String,
    pub elapsed_ms: u128,
    pub eof: bool,
    pub exit_code: Option<i32>,
    pub buffered: usize,
}

#[derive(Default)]
struct Buffer {
    data: VecDeque<u8>,
    /// Number of bytes dropped from the front of the ring.
    base: u64,
    eof: bool,
    exit: Option<i32>,
}

impl Buffer {
    fn push(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.data.push_back(b);
        }
        while self.data.len() > MAX_BUFFER_BYTES {
            self.data.pop_front();
            self.base += 1;
        }
    }

    fn read(&self, cursor: u64, max: usize) -> (Vec<u8>, u64, bool) {
        let start = cursor.max(self.base);
        let truncated = cursor < self.base;
        let skip = (start - self.base) as usize;
        let take = max.min(self.data.len().saturating_sub(skip));
        let data = self.data.iter().skip(skip).take(take).copied().collect();
        (data, start + take as u64, truncated)
    }
}

struct Session {
    id: String,
    started: Instant,
    buffer: std::sync::Arc<Mutex<Buffer>>,
    #[cfg(unix)]
    writer: std::sync::Arc<Mutex<std::fs::File>>,
    child: std::sync::Arc<Mutex<std::process::Child>>,
}

pub struct TerminalManager {
    sessions: Mutex<HashMap<String, std::sync::Arc<Session>>>,
    max: usize,
}

impl TerminalManager {
    pub fn new(max: usize) -> Self {
        TerminalManager {
            sessions: Mutex::new(HashMap::new()),
            max: max.clamp(1, MAX_SESSIONS),
        }
    }

    /// Start an interactive shell in a PTY. Unix only.
    #[cfg(unix)]
    pub fn start(
        &self,
        shell: &str,
        cwd: &std::path::Path,
        env: &[(String, String)],
        cols: u16,
        rows: u16,
    ) -> std::result::Result<String, String> {
        use std::os::fd::AsRawFd;
        use std::os::unix::process::CommandExt;
        use std::process::{Command, Stdio};

        let mut sessions = self.sessions.lock().unwrap();
        if sessions.len() >= self.max {
            return Err(format!(
                "too many terminal sessions ({} max); close one first",
                self.max
            ));
        }

        let winsize = nix::pty::Winsize {
            ws_row: rows.max(1),
            ws_col: cols.max(1),
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        let pty =
            nix::pty::openpty(Some(&winsize), None).map_err(|e| format!("openpty failed: {e}"))?;
        let master = pty.master;
        let slave = pty.slave;
        let slave_fd = slave.as_raw_fd();

        let mut cmd = Command::new(shell);
        cmd.current_dir(cwd)
            .env("TERM", "xterm-256color")
            .envs(env.iter().map(|(k, v)| (k.as_str(), v.as_str())))
            .stdin(Stdio::from(
                slave.try_clone().map_err(|e| format!("pty stdin: {e}"))?,
            ))
            .stdout(Stdio::from(
                slave.try_clone().map_err(|e| format!("pty stdout: {e}"))?,
            ))
            .stderr(Stdio::from(
                slave.try_clone().map_err(|e| format!("pty stderr: {e}"))?,
            ));

        // New session + controlling terminal so job control works like a shell.
        unsafe {
            cmd.pre_exec(move || {
                if libc::setsid() < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                // Best-effort: attach the controlling terminal.
                let _ = libc::ioctl(slave_fd, libc::TIOCSCTTY as _, 0);
                Ok(())
            });
        }

        let child = cmd.spawn().map_err(|e| format!("spawn '{shell}': {e}"))?;
        drop(slave);

        let master_file = std::fs::File::from(master);
        let mut reader = master_file
            .try_clone()
            .map_err(|e| format!("pty reader: {e}"))?;
        let buffer = std::sync::Arc::new(Mutex::new(Buffer::default()));
        let buffer2 = buffer.clone();
        std::thread::spawn(move || {
            use std::io::Read;
            let mut buf = [0u8; 8192];
            loop {
                match reader.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => buffer2.lock().unwrap().push(&buf[..n]),
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(_) => break,
                }
            }
            buffer2.lock().unwrap().eof = true;
        });

        let id = format!("term_{}", &uuid::Uuid::new_v4().simple().to_string()[..8]);
        let session = std::sync::Arc::new(Session {
            id: id.clone(),
            started: Instant::now(),
            buffer,
            writer: std::sync::Arc::new(Mutex::new(master_file)),
            child: std::sync::Arc::new(Mutex::new(child)),
        });
        sessions.insert(id.clone(), session);
        Ok(id)
    }

    #[cfg(not(unix))]
    pub fn start(
        &self,
        _shell: &str,
        _cwd: &std::path::Path,
        _env: &[(String, String)],
        _cols: u16,
        _rows: u16,
    ) -> std::result::Result<String, String> {
        Err("interactive PTY is only supported on Unix builds".into())
    }

    pub fn write(&self, id: &str, bytes: &[u8]) -> std::result::Result<(), String> {
        #[cfg(unix)]
        {
            use std::io::Write;
            let sessions = self.sessions.lock().unwrap();
            let session = sessions
                .get(id)
                .ok_or_else(|| format!("unknown terminal '{id}'"))?;
            let mut w = session.writer.lock().unwrap();
            w.write_all(bytes)
                .and_then(|_| w.flush())
                .map_err(|e| format!("terminal write: {e}"))
        }
        #[cfg(not(unix))]
        {
            let _ = (id, bytes);
            Err("interactive PTY is only supported on Unix builds".into())
        }
    }

    pub fn read(
        &self,
        id: &str,
        cursor: u64,
        max_bytes: usize,
    ) -> std::result::Result<ReadResult, String> {
        let sessions = self.sessions.lock().unwrap();
        let session = sessions
            .get(id)
            .ok_or_else(|| format!("unknown terminal '{id}'"))?;
        let mut buf = session.buffer.lock().unwrap();
        if buf.eof && buf.exit.is_none() {
            if let Ok(Some(status)) = session.child.lock().unwrap().try_wait() {
                buf.exit = status.code();
            }
        }
        let (data, cursor, truncated) = buf.read(cursor, max_bytes.max(1));
        Ok(ReadResult {
            data,
            cursor,
            eof: buf.eof,
            exit_code: buf.exit,
            truncated,
        })
    }

    #[cfg(unix)]
    pub fn resize(&self, id: &str, cols: u16, rows: u16) -> std::result::Result<(), String> {
        use std::os::fd::AsRawFd;
        let sessions = self.sessions.lock().unwrap();
        let session = sessions
            .get(id)
            .ok_or_else(|| format!("unknown terminal '{id}'"))?;
        let w = session.writer.lock().unwrap();
        let ws = libc::winsize {
            ws_row: rows.max(1),
            ws_col: cols.max(1),
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        let rc = unsafe { libc::ioctl(w.as_raw_fd(), libc::TIOCSWINSZ, &ws) };
        if rc < 0 {
            return Err(format!(
                "terminal resize failed: {}",
                std::io::Error::last_os_error()
            ));
        }
        Ok(())
    }

    #[cfg(not(unix))]
    pub fn resize(&self, _id: &str, _cols: u16, _rows: u16) -> std::result::Result<(), String> {
        Err("interactive PTY is only supported on Unix builds".into())
    }

    pub fn close(&self, id: &str) -> std::result::Result<(), String> {
        let session = {
            let mut sessions = self.sessions.lock().unwrap();
            sessions
                .remove(id)
                .ok_or_else(|| format!("unknown terminal '{id}'"))?
        };
        let mut child = session.child.lock().unwrap();
        let _ = child.kill();
        let _ = child.wait();
        Ok(())
    }

    pub fn list(&self) -> Vec<SessionInfo> {
        let sessions = self.sessions.lock().unwrap();
        sessions
            .values()
            .map(|s| {
                let b = s.buffer.lock().unwrap();
                SessionInfo {
                    id: s.id.clone(),
                    elapsed_ms: s.started.elapsed().as_millis(),
                    eof: b.eof,
                    exit_code: b.exit,
                    buffered: b.data.len(),
                }
            })
            .collect()
    }
}

#[cfg(all(test, feature = "pty", unix))]
mod tests {
    use super::*;

    #[test]
    fn pty_roundtrip_write_read_resize_close() {
        let mgr = TerminalManager::new(MAX_SESSIONS);
        let dir = std::env::temp_dir();
        let id = mgr
            .start(
                "/bin/sh",
                &dir,
                &[("TERM".into(), "xterm-256color".into())],
                80,
                24,
            )
            .expect("start pty");

        mgr.write(&id, b"printf 'PTY_OK\\n'\n").expect("write");

        let mut cursor = 0u64;
        let mut out = Vec::new();
        for _ in 0..200 {
            let r = mgr.read(&id, cursor, 4096).expect("read");
            out.extend_from_slice(&r.data);
            cursor = r.cursor;
            if String::from_utf8_lossy(&out).contains("PTY_OK") {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(
            String::from_utf8_lossy(&out).contains("PTY_OK"),
            "pty output: {:?}",
            String::from_utf8_lossy(&out)
        );

        mgr.resize(&id, 100, 40).expect("resize");
        mgr.write(&id, b"exit\n").expect("exit");
        mgr.close(&id).expect("close");
        assert!(mgr.list().is_empty());
    }

    #[test]
    fn max_sessions_is_enforced() {
        let mgr = TerminalManager::new(1);
        let dir = std::env::temp_dir();
        let id = mgr.start("/bin/sh", &dir, &[], 80, 24).unwrap();
        let second = mgr.start("/bin/sh", &dir, &[], 80, 24);
        assert!(second.is_err(), "second session should be rejected");
        mgr.close(&id).unwrap();
    }
}
