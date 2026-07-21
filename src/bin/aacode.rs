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

fn main() {
    let args: Vec<String> = std::env::args().collect();
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
    let rt = match AgentRuntime::init(config, project_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to initialize: {e}");
            std::process::exit(1);
        }
    };

    let sink = StdoutSink::new(true); // CLI is always TTY (human-readable)
    let cancel = AtomicBool::new(false);

    if interactive {
        run_interactive(&rt, &sink, &cancel);
        return;
    }

    let task = task_parts.join(" ");
    if task.trim().is_empty() {
        run_interactive(&rt, &sink, &cancel);
        return;
    }

    match rt.run_task(&task, session_id.as_deref(), &sink, &cancel) {
        Ok(res) => {
            println!("\n[status: {:?}, iterations: {}]", res.status, res.iterations);
        }
        Err(e) => {
            eprintln!("Task failed: {e}");
            std::process::exit(1);
        }
    }
}

fn run_interactive(rt: &AgentRuntime, sink: &StdoutSink, cancel: &AtomicBool) {
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
        match rt.run_task(task, None, sink, cancel) {
            Ok(res) => println!("\n[status: {:?}, iterations: {}]", res.status, res.iterations),
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
