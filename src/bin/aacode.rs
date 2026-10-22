// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Desktop CLI for aacode-rs. Useful for local development/debugging.
//!
//! Usage:
//!   aacode [-p <project>] [--session <id>] [--plan-first] "task description"
//!   aacode [-p <project>] --interactive
//!
//! Config comes from env (LLM_API_KEY / LLM_API_URL / LLM_MODEL_NAME / LLM_GATEWAY).
//! Also auto-loads `.env` from the current directory and the binary's parent dir.

use aacode_rs::config::AgentConfig;
use aacode_rs::runtime::AgentRuntime;
use aacode_rs::stream::StdoutSink;
use std::io::Write;
use std::sync::atomic::AtomicBool;

/// Load a `.env` file into the process environment (simple format, no quoting support).
fn load_dotenv(path: &std::path::Path) {
    if !path.exists() {
        return;
    }
    let content = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => return,
    };
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some(eq) = trimmed.find('=') {
            let key = trimmed[..eq].trim();
            let val = trimmed[eq + 1..].trim();
            if !key.is_empty() {
                // Only set if not already in env (env > .env).
                if std::env::var(key).is_err() {
                    std::env::set_var(key, val);
                }
            }
        }
    }
}

#[cfg(all(feature = "pty", unix))]
fn run_terminal(args: &[String]) {
    use std::io::{Read, Write};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    let shell = args.first().map(String::as_str).unwrap_or("/bin/sh");
    let (cols, rows) = term_size();
    let mgr = Arc::new(Mutex::new(aacode_rs::terminal::TerminalManager::new(2)));
    let id = match mgr
        .lock()
        .unwrap()
        .start(shell, std::path::Path::new("."), &[], cols, rows)
    {
        Ok(id) => id,
        Err(e) => {
            eprintln!("terminal: {e}");
            return;
        }
    };
    eprintln!("aacode terminal — {shell} ({cols}x{rows}). Type `exit` to leave.");
    let _raw = RawMode::enable();

    // stdin -> PTY
    let writer = {
        let mgr = mgr.clone();
        let id = id.clone();
        std::thread::spawn(move || {
            let mut stdin = std::io::stdin();
            let mut buf = [0u8; 1024];
            loop {
                match stdin.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        if mgr.lock().unwrap().write(&id, &buf[..n]).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        })
    };

    // PTY -> stdout
    let mut cursor = 0u64;
    loop {
        let read = mgr.lock().unwrap().read(&id, cursor, 8192);
        match read {
            Ok(r) => {
                if !r.data.is_empty() {
                    let mut out = std::io::stdout();
                    let _ = out.write_all(&r.data);
                    let _ = out.flush();
                    cursor = r.cursor;
                }
                if r.eof {
                    break;
                }
            }
            Err(_) => break,
        }
        std::thread::sleep(Duration::from_millis(15));
    }
    let _ = mgr.lock().unwrap().close(&id);
    // Do not join the stdin reader: it may be blocked on read(0) and the
    // process exits right after this function returns.
    let _ = writer;
}

#[cfg(all(feature = "pty", unix))]
struct RawMode {
    orig: libc::termios,
}

#[cfg(all(feature = "pty", unix))]
impl RawMode {
    fn enable() -> Option<Self> {
        unsafe {
            let mut t: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(0, &mut t) != 0 {
                return None;
            }
            let orig = t;
            libc::cfmakeraw(&mut t);
            if libc::tcsetattr(0, libc::TCSANOW, &t) != 0 {
                return None;
            }
            Some(RawMode { orig })
        }
    }
}

#[cfg(all(feature = "pty", unix))]
impl Drop for RawMode {
    fn drop(&mut self) {
        unsafe {
            libc::tcsetattr(0, libc::TCSANOW, &self.orig);
        }
    }
}

#[cfg(all(feature = "pty", unix))]
fn term_size() -> (u16, u16) {
    unsafe {
        let mut ws: libc::winsize = std::mem::zeroed();
        if libc::ioctl(0, libc::TIOCGWINSZ, &mut ws) == 0 && ws.ws_col > 0 && ws.ws_row > 0 {
            (ws.ws_col, ws.ws_row)
        } else {
            (80, 24)
        }
    }
}

fn main() {
    let tokio_rt = tokio::runtime::Runtime::new().expect("tokio runtime");

    let _args: Vec<String> = std::env::args().collect();
    let args: Vec<String> = std::env::args().collect();

    // `aacode terminal [shell]` — interactive PTY (Unix, feature `pty`).
    if args.get(1).map(String::as_str) == Some("terminal") {
        #[cfg(all(feature = "pty", unix))]
        {
            run_terminal(&args[2..]);
            return;
        }
        #[cfg(not(all(feature = "pty", unix)))]
        {
            eprintln!("terminal requires building aacode-rs with --features pty on Unix");
            return;
        }
    }

    let mut project = ".".to_string();
    let mut session_id: Option<String> = None;
    let mut plan_first = false;
    let mut interactive = false;
    let mut task_parts: Vec<String> = Vec::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-p" | "--project" => {
                i += 1;
                if i < args.len() {
                    project = args[i].clone();
                }
            }
            "--session" => {
                i += 1;
                if i < args.len() {
                    session_id = Some(args[i].clone());
                }
            }
            "--plan-first" => plan_first = true,
            "--interactive" | "-i" => interactive = true,
            "-h" | "--help" => {
                print_help();
                return;
            }
            other => task_parts.push(other.to_string()),
        }
        i += 1;
    }

    // Auto-load .env from cwd and from aacode-rs directory (env already set wins).
    load_dotenv(&std::env::current_dir().unwrap_or_default().join(".env"));
    // Also try the project dir's .env.
    load_dotenv(&std::path::PathBuf::from(&project).join(".env"));

    let mut config = AgentConfig::default();
    config.apply_env();
    config.plan_first = plan_first;

    let errs = config.validate();
    if !errs.is_empty() {
        eprintln!("Configuration error:");
        for e in errs {
            eprintln!("  - {e}");
        }
        eprintln!("\nSet LLM_API_KEY (and optionally LLM_API_URL / LLM_MODEL_NAME / LLM_GATEWAY).");
        std::process::exit(1);
    }

    let project_path = std::path::PathBuf::from(&project);
    let agent_rt = match AgentRuntime::init(config, project_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to initialize: {e}");
            std::process::exit(1);
        }
    };

    let sink = StdoutSink::new(true); // CLI is always TTY (human-readable)
    let cancel = AtomicBool::new(false);

    if interactive {
        run_interactive(&tokio_rt, &agent_rt, &sink, &cancel);
        return;
    }

    let task = task_parts.join(" ");
    if task.trim().is_empty() {
        run_interactive(&tokio_rt, &agent_rt, &sink, &cancel);
        return;
    }

    match tokio_rt.block_on(agent_rt.run_task(&task, session_id.as_deref(), &sink, &cancel)) {
        Ok(res) => {
            println!(
                "\n[status: {:?}, iterations: {}]",
                res.status, res.iterations
            );
        }
        Err(e) => {
            eprintln!("Task failed: {e}");
            std::process::exit(1);
        }
    }
}

fn run_interactive(
    tokio_rt: &tokio::runtime::Runtime,
    agent_rt: &AgentRuntime,
    sink: &StdoutSink,
    cancel: &AtomicBool,
) {
    println!("aacode-rs interactive mode. Type a task, or 'exit' to quit.");
    loop {
        print!("\n> ");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line).is_err() {
            break;
        }
        let task = line.trim();
        if task.is_empty() {
            continue;
        }
        if matches!(task, "exit" | "quit" | "q") {
            println!("bye");
            break;
        }
        match tokio_rt.block_on(agent_rt.run_task(task, None, sink, cancel)) {
            Ok(res) => println!(
                "\n[status: {:?}, iterations: {}]",
                res.status, res.iterations
            ),
            Err(e) => eprintln!("error: {e}"),
        }
    }
}

fn print_help() {
    println!(
        "aacode-rs — Rust coding agent\n\n\
        USAGE:\n  \
        aacode [-p <project>] [--session <id>] [--plan-first] \"task\"\n  \
        aacode [-p <project>] --interactive\n\n\
        OPTIONS:\n  \
        -p, --project <dir>   Project sandbox directory (default: .)\n  \
        --session <id>        Continue an existing session\n  \
        --plan-first          Plan before executing\n  \
        -i, --interactive     Interactive session mode\n  \
        -h, --help            Show this help\n\n\
        ENV:\n  \
        LLM_API_KEY, LLM_API_URL, LLM_MODEL_NAME, LLM_GATEWAY, SEARCHXNG_URL"
    );
}
