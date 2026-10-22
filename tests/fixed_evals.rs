// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Fixed, hash-pinned eval set (fixture provider).
//!
//! This is a *deterministic* Agent-Loop eval: it does not measure model
//! quality (that needs a real provider). It locks the harness contract —
//! tool dispatch, read-only gating, completion — against a frozen dataset.
//! Changes to the dataset must be intentional: update `CASES_SHA256`.

mod common;

use aacode_rs::agent::react_loop::{ReactLoop, RunStatus};
use aacode_rs::config::{AgentConfig, PermissionMode};
use aacode_rs::llm::ChatMessage;
use aacode_rs::session::SessionManager;
use aacode_rs::stream::CollectingSink;
use aacode_rs::tools::registry::ToolRegistry;
use common::{final_text, new_log, temp_project, tool_response, LogTool, ScriptedLlm};
use sha2::{Digest, Sha256};
use std::sync::atomic::AtomicBool;

const CASES: &str = include_str!("evals/fixed_cases.tsv");
/// Pinned SHA-256 of `tests/evals/fixed_cases.tsv` (UTF-8 bytes).
const CASES_SHA256: &str = "effd1050de9af22da1f0c27f389186a23f18e8ff57b021400b2625a545c3ae02";

#[test]
fn fixed_eval_dataset_digest_is_pinned() {
    let digest = format!("{:x}", Sha256::digest(CASES.as_bytes()));
    assert_eq!(
        digest, CASES_SHA256,
        "fixed eval dataset changed; update CASES_SHA256 intentionally"
    );
}

#[tokio::test]
async fn fixed_cases_dispatch_and_gate_correctly() {
    for line in CASES
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
    {
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 6, "malformed case: {line}");
        let (id, prompt, tool, args, mode, expect) =
            (cols[0], cols[1], cols[2], cols[3], cols[4], cols[5]);
        let args: serde_json::Value = serde_json::from_str(args).unwrap();

        let log = new_log();
        let mut reg = ToolRegistry::new();
        for name in ["run_shell", "get_page_text", "search_web", "read_file"] {
            reg.register(Box::new(LogTool {
                name,
                log: log.clone(),
                output: "ok".into(),
            }));
        }

        let llm = ScriptedLlm::new(vec![tool_response("c1", tool, args), final_text("done")]);
        let mut cfg = AgentConfig::default();
        cfg.max_iterations = 3;
        if mode == "read_only" {
            cfg.safety.permission_mode = PermissionMode::ReadOnly;
        }
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);
        let proj = temp_project("eval");
        let mut sm = SessionManager::new(&proj);
        sm.create_session(prompt, None).unwrap();
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let res = loop_
            .run(
                vec![ChatMessage::system("s"), ChatMessage::user(prompt)],
                &mut sm,
                &sink,
                &cancel,
            )
            .await
            .unwrap();

        assert_eq!(
            res.status,
            RunStatus::Completed,
            "case {id}: {:?}",
            res.status
        );
        let calls = log.lock().unwrap().clone();
        match expect {
            "invoked" => assert!(
                calls.iter().any(|c| c == tool),
                "case {id}: expected {tool} to run, got {calls:?}"
            ),
            "blocked" => assert!(
                !calls.iter().any(|c| c == tool),
                "case {id}: {tool} must be blocked in read-only mode"
            ),
            other => panic!("case {id}: unknown expectation {other}"),
        }
        if expect == "blocked" {
            let blocked = sm
                .messages
                .iter()
                .find(|m| m.role == "tool" && m.tool_call_id.as_deref() == Some("c1"))
                .expect("blocked tool result");
            assert!(
                blocked.content.contains("read-only mode"),
                "case {id}: unexpected observation {}",
                blocked.content
            );
        }
        let _ = std::fs::remove_dir_all(&proj);
    }
}
