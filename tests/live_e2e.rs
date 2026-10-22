// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Live end-to-end tests against a real OpenAI-compatible gateway.
//!
//! The API key is read **only** from the environment and is never written to
//! the repository or printed. Run explicitly (never in CI):
//!
//! ```bash
//! export LLM_API_KEY=... LLM_API_URL=https://api.deepseek.com/v1 LLM_MODEL_NAME=deepseek-flash
//! cargo test -p aacode-rs --test live_e2e -- --ignored --nocapture
//! ```

use aacode_rs::config::{AgentConfig, Gateway, ModelConfig};
use aacode_rs::runtime::AgentRuntime;
use aacode_rs::session::SessionManager;
use aacode_rs::stream::CollectingSink;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

fn model_from_env() -> ModelConfig {
    let api_key = std::env::var("LLM_API_KEY")
        .expect("LLM_API_KEY must be set (never hardcode it in the repo)");
    assert!(
        api_key.trim_start().starts_with("sk-"),
        "LLM_API_KEY does not look like an API key"
    );
    ModelConfig {
        name: std::env::var("LLM_MODEL_NAME").unwrap_or_else(|_| "deepseek-chat".into()),
        api_key: Some(api_key),
        base_url: std::env::var("LLM_API_URL").ok(),
        gateway: Gateway::Openai,
        temperature: 0.0,
        max_tokens: 512,
        ..Default::default()
    }
}

fn tmp_project(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "aacode_{tag}_{}_{}",
        std::process::id(),
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn read_usage(project: &PathBuf) -> Vec<serde_json::Value> {
    let path = project.join(".aacode").join("usage").join("usage.jsonl");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

fn print_usage(entries: &[serde_json::Value]) {
    for e in entries {
        eprintln!(
            ">>> usage run={} iter={} input={} output={} cache_read={} miss={} hit_rate={} source={}",
            e["run_id"],
            e["iteration"],
            e["input_tokens"],
            e["output_tokens"],
            e["cache_read_tokens"],
            e["cache_miss_tokens"],
            e["cache_hit_rate"],
            e["usage_source"],
        );
    }
}

/// Full agent loop: tool call → observation → final answer, and verify the new
/// usage log + structured `tool_status` event.
#[tokio::test]
#[ignore = "requires a real LLM_API_KEY (network)"]
async fn live_e2e_tool_call_usage_and_status() {
    let project = tmp_project("live_e2e");
    let mut cfg = AgentConfig::default();
    cfg.model = model_from_env();
    cfg.max_iterations = 6;
    cfg.limits.max_retries = 1;

    let rt = AgentRuntime::init(cfg, project.clone()).unwrap();
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);

    let task = "Use the run_shell tool exactly once to execute `echo E2E_OK`, then reply with the command output.";
    let res = rt.run_task(task, None, &sink, &cancel).await.unwrap();
    eprintln!(
        ">>> status={:?} iterations={} final_text={}",
        res.status, res.iterations, res.final_text
    );
    assert_eq!(
        format!("{:?}", res.status),
        "Completed",
        "{}",
        res.final_text
    );

    let lines = sink.lines();
    assert!(
        res.final_text.contains("E2E_OK") || lines.iter().any(|l| l.contains("E2E_OK")),
        "model neither reported nor observed E2E_OK"
    );
    assert!(
        lines.iter().any(|l| l.contains("\"type\":\"tool_status\"")),
        "no tool_status event was emitted"
    );
    assert!(
        lines
            .iter()
            .any(|l| l.contains("\"type\":\"tool_status\"") && l.contains("\"status\":\"success\"")),
        "tool_status did not report success"
    );

    let entries = read_usage(&project);
    print_usage(&entries);
    assert!(!entries.is_empty(), "usage.jsonl has no entries");
    assert!(
        entries.iter().any(|e| e["usage_source"] == "provider"),
        "gateway did not report provider usage"
    );
    assert!(
        entries
            .iter()
            .any(|e| e["input_tokens"].as_u64().unwrap_or(0) > 0),
        "provider usage has no input_tokens"
    );

    let _ = std::fs::remove_dir_all(&project);
}

/// Cross-task prefix cache: after task 1, task 2 in the same session must
/// replay task 1's user message **verbatim** (no "Task:" rewrite) so the
/// provider prefix cache hits.
#[tokio::test]
#[ignore = "requires a real LLM_API_KEY (network)"]
async fn live_e2e_cross_task_cache() {
    let project = tmp_project("live_cache");
    let mut cfg = AgentConfig::default();
    cfg.model = model_from_env();
    cfg.max_iterations = 4;
    cfg.limits.max_retries = 1;

    let rt = AgentRuntime::init(cfg, project.clone()).unwrap();
    let cancel = AtomicBool::new(false);

    let sink1 = CollectingSink::new(false);
    let r1 = rt
        .run_task("Reply with exactly: FIRST_OK", None, &sink1, &cancel)
        .await
        .unwrap();
    assert_eq!(format!("{:?}", r1.status), "Completed", "{}", r1.final_text);
    eprintln!(">>> task1 final_text={}", r1.final_text);

    let sid = SessionManager::new(&project)
        .list_sessions()
        .first()
        .map(|s| s.session_id.clone())
        .expect("session not persisted");

    let sink2 = CollectingSink::new(false);
    let r2 = rt
        .run_task("Reply with exactly: SECOND_OK", Some(&sid), &sink2, &cancel)
        .await
        .unwrap();
    assert_eq!(format!("{:?}", r2.status), "Completed", "{}", r2.final_text);
    eprintln!(">>> task2 final_text={}", r2.final_text);

    let entries = read_usage(&project);
    print_usage(&entries);
    let run_ids: Vec<String> = {
        let mut ids = Vec::new();
        for e in &entries {
            let id = e["run_id"].as_str().unwrap_or("").to_string();
            if !id.is_empty() && !ids.contains(&id) {
                ids.push(id);
            }
        }
        ids
    };
    assert!(run_ids.len() >= 2, "expected >=2 runs in usage log");

    // The provider reports cache fields only for DeepSeek-style gateways. If it
    // does, the second run's first call must hit the cache (the fix under test).
    let any_cache_field = entries.iter().any(|e| {
        e["cache_read_tokens"].as_u64().unwrap_or(0) > 0
            || e["cache_miss_tokens"].as_u64().unwrap_or(0) > 0
    });
    if any_cache_field {
        let second_run = &run_ids[1];
        let first_of_second = entries
            .iter()
            .filter(|e| e["run_id"].as_str() == Some(second_run.as_str()))
            .min_by_key(|e| e["iteration"].as_u64().unwrap_or(0))
            .expect("second run has no usage entry");
        let hit = first_of_second["cache_read_tokens"].as_u64().unwrap_or(0);
        eprintln!(">>> second run first call cache_read={hit}");
        assert!(
            hit > 0,
            "cross-task cache miss: second run's first call read 0 cached tokens"
        );
    } else {
        eprintln!(">>> gateway reported no cache fields; skipping cache assertion");
    }

    let _ = std::fs::remove_dir_all(&project);
}
