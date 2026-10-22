// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Prompt-injection regression matrix.
//!
//! Every payload arrives as a **tool observation** (the realistic path: web
//! pages, files, MCP results, filenames). The invariant under test is
//! structural: injected text must never become a system message, create a tool
//! call, change the tool registry, or change the permission mode. Model refusal
//! is not relied upon.

mod common;

use aacode_rs::agent::react_loop::{ReactLoop, RunStatus};
use aacode_rs::config::{AgentConfig, PermissionMode};
use aacode_rs::llm::ChatMessage;
use aacode_rs::session::SessionManager;
use aacode_rs::stream::CollectingSink;
use aacode_rs::tools::registry::ToolRegistry;
use common::{final_text, new_log, temp_project, tool_response, LogTool, ScriptedLlm};
use std::sync::atomic::AtomicBool;

const PAYLOADS: &str = include_str!("evals/prompt_injection.tsv");

#[tokio::test]
async fn injected_observation_cannot_create_calls_or_change_policy() {
    for line in PAYLOADS
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
    {
        let (id, payload) = line
            .split_once('\t')
            .expect("payload TSV needs id<TAB>payload");

        let log = new_log();
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(LogTool {
            name: "read_file",
            log: log.clone(),
            output: payload.to_string(),
        }));
        reg.register(Box::new(LogTool {
            name: "run_shell",
            log: log.clone(),
            output: "should never run".into(),
        }));
        let registry_before = reg.names();

        let llm = ScriptedLlm::new(vec![
            tool_response("c1", "read_file", serde_json::json!({"path": id})),
            final_text("done"),
        ]);
        let mut cfg = AgentConfig::default();
        cfg.max_iterations = 3;
        cfg.safety.permission_mode = PermissionMode::ReadOnly;
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);
        let proj = temp_project("inject");
        let mut sm = SessionManager::new(&proj);
        sm.create_session("read the file", None).unwrap();
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let res = loop_
            .run(
                vec![ChatMessage::system("s"), ChatMessage::user("read the file")],
                &mut sm,
                &sink,
                &cancel,
            )
            .await
            .unwrap();
        assert_eq!(res.status, RunStatus::Completed, "payload {id}");

        // 1) Only the intended read ran; injected text created no tool call.
        let calls = log.lock().unwrap().clone();
        assert_eq!(
            calls,
            vec!["read_file".to_string()],
            "payload {id}: injected text caused extra tool execution"
        );

        // 2) Registry and permission policy are unchanged.
        assert_eq!(
            reg.names(),
            registry_before,
            "payload {id}: registry changed"
        );
        assert_eq!(
            cfg.safety.permission_mode,
            PermissionMode::ReadOnly,
            "payload {id}: permission mode changed"
        );

        // 3) The payload exists only as an untrusted tool observation.
        let seen = llm.seen.lock().unwrap();
        assert!(
            seen.len() >= 2,
            "payload {id}: expected a second model request"
        );
        let second = &seen[1];
        assert!(
            second
                .iter()
                .any(|m| m.role == "tool" && m.content.contains(payload)),
            "payload {id}: tool observation missing"
        );
        assert!(
            !second
                .iter()
                .any(|m| m.role == "system" && m.content.contains(payload)),
            "payload {id}: injected text was promoted to a system message"
        );
        assert!(
            !second.iter().any(|m| {
                m.tool_calls
                    .as_ref()
                    .map(|t| t.iter().any(|c| c.name == "run_shell"))
                    .unwrap_or(false)
            }),
            "payload {id}: injected text created a run_shell tool call"
        );

        let _ = std::fs::remove_dir_all(&proj);
    }
}
