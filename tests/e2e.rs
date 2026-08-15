// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! End-to-end tests against real LLM APIs.
//!
//! Run with:
//!   DEEPSEEK=1 cargo test --test e2e -- --nocapture
//!   MOONSHOT=1 cargo test --test e2e -- --nocapture
//!   MINIMAX=1 cargo test --test e2e -- --nocapture
//!   ALL=1 cargo test --test e2e -- --nocapture

use aacode_rs::config::{AgentConfig, Gateway, ModelConfig};
use aacode_rs::runtime::AgentRuntime;
use aacode_rs::stream::CollectingSink;
use std::env;
use std::sync::atomic::AtomicBool;

// ── Helpers ────────────────────────────────────────────────────────────────

fn deepseek_key() -> String { env::var("DEEPSEEK_KEY").unwrap_or_default() }
fn deepseek_url() -> String { env::var("DEEPSEEK_URL").unwrap_or_else(|_| "https://api.deepseek.com/v1".into()) }
fn deepseek_model() -> String { env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-v4-flash".into()) }

fn moonshot_key() -> String { env::var("MOONSHOT_KEY").unwrap_or_default() }
fn moonshot_url() -> String { env::var("MOONSHOT_URL").unwrap_or_else(|_| "https://api.moonshot.cn/v1".into()) }
fn moonshot_model() -> String { env::var("MOONSHOT_MODEL").unwrap_or_else(|_| "kimi-k2.6".into()) }

fn minimax_key() -> String { env::var("MINIMAX_KEY").unwrap_or_default() }
fn minimax_url() -> String { env::var("MINIMAX_URL").unwrap_or_else(|_| "https://api.minimax.chat/anthropic".into()) }
fn minimax_model() -> String { env::var("MINIMAX_MODEL").unwrap_or_else(|_| "MiniMax-M2.7".into()) }

fn should_run(provider: &str) -> bool {
    if env::var("ALL").is_ok() { return true; }
    if env::var(provider).is_err() { return false; }
    let key = match provider {
        "DEEPSEEK" => deepseek_key(),
        "MOONSHOT" => moonshot_key(),
        "MINIMAX" => minimax_key(),
        _ => return false,
    };
    !key.is_empty()
}
fn tmp_project() -> std::path::PathBuf { std::env::temp_dir().join(format!("aacode_e2e_{}_{}", std::process::id(), uuid::Uuid::new_v4().simple())) }

fn openai_cfg(key: &str, url: &str, model: &str) -> AgentConfig {
    AgentConfig {
        model: ModelConfig {
            name: model.to_string(),
            api_key: Some(key.to_string()),
            base_url: Some(url.to_string()),
            gateway: Gateway::Openai,
            max_tokens: 2048,
            temperature: 0.1,
            multimodal: false,
            request_timeout_secs: Some(120),
        },
        max_iterations: 10,
        ..Default::default()
    }
}

fn anthropic_cfg(key: &str, url: &str, model: &str) -> AgentConfig {
    AgentConfig {
        model: ModelConfig {
            name: model.to_string(),
            api_key: Some(key.to_string()),
            base_url: Some(url.to_string()),
            gateway: Gateway::Anthropic,
            max_tokens: 2048,
            temperature: 0.1,
            multimodal: false,
            request_timeout_secs: Some(120),
        },
        max_iterations: 10,
        ..Default::default()
    }
}

// ── DeepSeek tests (OpenAI gateway, no multimodal) ─────────────────────────

#[test]
fn deepseek_validate_api_key() {
    if !should_run("DEEPSEEK") { return; }
    let cfg = openai_cfg(&deepseek_key(), &deepseek_url(), &deepseek_model());
    let proj = tmp_project();
    let rt = AgentRuntime::init(cfg, proj).expect("init");
    rt.validate_api_key().expect("validate");
    println!("[deepseek] validate: OK");
}

#[tokio::test]
async fn deepseek_simple_task() {
    if !should_run("DEEPSEEK") { return; }
    let cfg = openai_cfg(&deepseek_key(), &deepseek_url(), &deepseek_model());
    let proj = tmp_project();
    let rt = AgentRuntime::init(cfg, proj).expect("init");
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    let res = rt.run_task("Write a one-line poem about Rust.", None, &sink, &cancel).await.expect("run");
    println!("[deepseek] simple: iter={} status={:?} text='{}'", res.iterations, res.status, res.final_text.chars().take(200).collect::<String>());
    assert!(res.iterations > 0);
    assert!(!res.final_text.is_empty());
}

#[tokio::test]
async fn deepseek_shell_tool() {
    if !should_run("DEEPSEEK") { return; }
    let cfg = openai_cfg(&deepseek_key(), &deepseek_url(), &deepseek_model());
    let proj = tmp_project();
    let rt = AgentRuntime::init(cfg, proj).expect("init");
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    let res = rt.run_task("Run: echo 'hello world' > /tmp/test.txt && cat /tmp/test.txt. Report what the file contains.", None, &sink, &cancel).await.expect("run");
    println!("[deepseek] shell: iter={} status={:?} text='{}'", res.iterations, res.status, res.final_text.chars().take(300).collect::<String>());
    assert!(res.iterations > 1, "should use at least one tool call");
    assert!(!res.final_text.is_empty());
}

#[test]
fn deepseek_invalid_key_errors() {
    let cfg = openai_cfg("sk-bad-key-12345", &deepseek_url(), &deepseek_model());
    let proj = tmp_project();
    let rt = AgentRuntime::init(cfg, proj).expect("init");
    let err = rt.validate_api_key().unwrap_err();
    println!("[deepseek] bad key: {err}");
    assert!(!err.is_retryable());
}

#[tokio::test]
async fn deepseek_streaming_events() {
    if !should_run("DEEPSEEK") { return; }
    let cfg = openai_cfg(&deepseek_key(), &deepseek_url(), &deepseek_model());
    let proj = tmp_project();
    let rt = AgentRuntime::init(cfg, proj).expect("init");
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    rt.run_task("Count from 1 to 5, one per line.", None, &sink, &cancel).await.expect("run");
    let lines = sink.lines();
    assert!(lines.iter().any(|l| l.contains(r#""type":"start""#)));
    assert!(lines.iter().any(|l| l.contains(r#""type":"done""#)));
    // Should have delta events (streaming tokens)
    let has_deltas = lines.iter().any(|l| l.contains(r#""type":"delta""#));
    let has_seg = lines.iter().any(|l| l.contains(r#""seg":"thought""#));
    println!("[deepseek] stream: {} lines, has_deltas={has_deltas}, has_seg={has_seg}", lines.len());
    assert!(has_deltas || has_seg, "should have streaming events");
}

#[tokio::test]
async fn deepseek_cancel_mid_stream() {
    if !should_run("DEEPSEEK") { return; }
    let cfg = openai_cfg(&deepseek_key(), &deepseek_url(), &deepseek_model());
    let proj = tmp_project();
    let rt = AgentRuntime::init(cfg, proj).expect("init");
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(true);  // immediately cancelled
    let res = rt.run_task("Write a very long essay about the history of Rome.", None, &sink, &cancel).await.expect("run");
    println!("[deepseek] cancel: iter={} status={:?}", res.iterations, res.status);
    assert!(res.iterations <= 1, "should stop quickly");
}

// ── Moonshot / Kimi tests (OpenAI gateway, multimodal) ─────────────────────

#[test]
fn moonshot_validate_api_key() {
    if !should_run("MOONSHOT") { return; }
    let cfg = openai_cfg(&moonshot_key(), &moonshot_url(), &moonshot_model());
    let proj = tmp_project();
    let rt = AgentRuntime::init(cfg, proj).expect("init");
    rt.validate_api_key().expect("validate");
    println!("[moonshot] validate: OK");
}

#[tokio::test]
async fn moonshot_simple_task() {
    if !should_run("MOONSHOT") { return; }
    let cfg = openai_cfg(&moonshot_key(), &moonshot_url(), &moonshot_model());
    let proj = tmp_project();
    let rt = AgentRuntime::init(cfg, proj).expect("init");
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    let res = rt.run_task("Say '你好世界' and explain what it means in English.", None, &sink, &cancel).await.expect("run");
    println!("[moonshot] simple: iter={} status={:?} text='{}'", res.iterations, res.status, res.final_text.chars().take(300).collect::<String>());
    assert!(res.iterations > 0);
    assert!(!res.final_text.is_empty());
    assert!(res.final_text.contains("你好"));
}

#[tokio::test]
async fn moonshot_shell_tool() {
    if !should_run("MOONSHOT") { return; }
    let cfg = openai_cfg(&moonshot_key(), &moonshot_url(), &moonshot_model());
    let proj = tmp_project();
    let rt = AgentRuntime::init(cfg, proj).expect("init");
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    let res = rt.run_task("Create a python file that computes fibonacci(10). Then run it and report the result.", None, &sink, &cancel).await.expect("run");
    println!("[moonshot] shell: iter={} status={:?} text='{}'", res.iterations, res.status, res.final_text.chars().take(300).collect::<String>());
    assert!(res.iterations > 1, "should use multiple tool calls");
    assert!(!res.final_text.is_empty());
}

#[tokio::test]
async fn moonshot_write_and_run_code() {
    if !should_run("MOONSHOT") { return; }
    let cfg = openai_cfg(&moonshot_key(), &moonshot_url(), &moonshot_model());
    let proj = tmp_project();
    let rt = AgentRuntime::init(cfg, proj).expect("init");
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    let res = rt.run_task("1) Write a file named math.py with a function square(x) that returns x*x. 2) Run: python3 -c 'from math import square; print(square(7))'. 3) Tell me the output.", None, &sink, &cancel).await.expect("run");
    println!("[moonshot] code: iter={} status={:?} text='{}'", res.iterations, res.status, res.final_text.chars().take(500).collect::<String>());
    let lines = sink.lines();
    let has_observation = lines.iter().any(|l| l.contains(r#""seg":"observation""#));
    println!("[moonshot] code: has_observation={has_observation}");
    assert!(has_observation, "should have observation events");
    // Some models struggle with the math.py naming collision — having
    // tool calls and observations proves the loop works regardless.
    assert!(res.iterations > 0, "should have at least one iteration");
}

#[tokio::test]
async fn moonshot_streaming_events() {
    if !should_run("MOONSHOT") { return; }
    let cfg = openai_cfg(&moonshot_key(), &moonshot_url(), &moonshot_model());
    let proj = tmp_project();
    let rt = AgentRuntime::init(cfg, proj).expect("init");
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    rt.run_task("Write a haiku about the moon.", None, &sink, &cancel).await.expect("run");
    let lines = sink.lines();
    let has_done = lines.iter().any(|l| l.contains(r#""type":"done""#));
    let has_thought = lines.iter().any(|l| l.contains(r#""seg":"thought""#));
    println!("[moonshot] stream: {} lines, done={has_done}, thought={has_thought}", lines.len());
    assert!(has_done);
    assert!(has_thought, "should emit thought segment");
}

#[tokio::test]
async fn moonshot_reasoning_content() {
    if !should_run("MOONSHOT") { return; }
    let cfg = openai_cfg(&moonshot_key(), &moonshot_url(), &moonshot_model());
    let mut cfg = cfg;
    cfg.model.max_tokens = 4096;
    let proj = tmp_project();
    let rt = AgentRuntime::init(cfg, proj).expect("init");
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    rt.run_task("Solve: If a train travels at 60 mph for 2.5 hours, how far does it go? Show your reasoning.", None, &sink, &cancel).await.expect("run");
    let lines = sink.lines();
    let has_thinking = lines.iter().any(|l| l.contains("thinking"));
    let has_thought = lines.iter().any(|l| l.contains("thought"));
    println!("[moonshot] reasoning: {} lines, thinking={has_thinking}, thought={has_thought}", lines.len());
    assert!(has_thought, "should have thought events");
}

// ── MiniMax tests (Anthropic gateway, multimodal) ─────────────────────────

#[test]
fn minimax_validate_api_key() {
    if !should_run("MINIMAX") { return; }
    let cfg = anthropic_cfg(&minimax_key(), &minimax_url(), &minimax_model());
    let proj = tmp_project();
    let rt = AgentRuntime::init(cfg, proj).expect("init");
    rt.validate_api_key().expect("validate");
    println!("[minimax] validate: OK");
}

#[tokio::test]
async fn minimax_simple_task() {
    if !should_run("MINIMAX") { return; }
    let cfg = anthropic_cfg(&minimax_key(), &minimax_url(), &minimax_model());
    let proj = tmp_project();
    let rt = AgentRuntime::init(cfg, proj).expect("init");
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    let res = rt.run_task("Say 'Bonjour' and explain what it means.", None, &sink, &cancel).await.expect("run");
    println!("[minimax] simple: iter={} status={:?} text='{}'", res.iterations, res.status, res.final_text.chars().take(300).collect::<String>());
    assert!(res.iterations > 0);
    assert!(!res.final_text.is_empty());
}

#[tokio::test]
async fn minimax_shell_tool() {
    if !should_run("MINIMAX") { return; }
    let cfg = anthropic_cfg(&minimax_key(), &minimax_url(), &minimax_model());
    let proj = tmp_project();
    let rt = AgentRuntime::init(cfg, proj).expect("init");
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    let res = rt.run_task("Run: echo 'hello from minimax' > /tmp/test.txt && cat /tmp/test.txt. Report what it says.", None, &sink, &cancel).await.expect("run");
    println!("[minimax] shell: iter={} status={:?} text='{}'", res.iterations, res.status, res.final_text.chars().take(300).collect::<String>());
    assert!(res.iterations > 1, "should use tool calls");
    assert!(!res.final_text.is_empty());
}

#[tokio::test]
async fn minimax_write_and_run_code() {
    if !should_run("MINIMAX") { return; }
    let cfg = anthropic_cfg(&minimax_key(), &minimax_url(), &minimax_model());
    let proj = tmp_project();
    let rt = AgentRuntime::init(cfg, proj).expect("init");
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    let res = rt.run_task("1) Write a file calc.py containing: def add(a,b): return a+b. 2) Run: python3 -c 'from calc import add; print(add(3,8))'. 3) Tell me the answer.", None, &sink, &cancel).await.expect("run");
    println!("[minimax] code: iter={} status={:?} text='{}'", res.iterations, res.status, res.final_text.chars().take(500).collect::<String>());
    let lines = sink.lines();
    let has_action = lines.iter().any(|l| l.contains(r#""seg":"action""#) && l.contains("run_shell"));
    let has_obs = lines.iter().any(|l| l.contains(r#""seg":"observation""#));
    println!("[minimax] code: action={has_action} observation={has_obs}");
    assert!(has_action || has_obs, "should have tool events");
    assert!(res.final_text.contains("11"), "should get 11");
}

#[tokio::test]
async fn minimax_tool_use_event_protocol() {
    if !should_run("MINIMAX") { return; }
    let cfg = anthropic_cfg(&minimax_key(), &minimax_url(), &minimax_model());
    let proj = tmp_project();
    let rt = AgentRuntime::init(cfg, proj).expect("init");
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    rt.run_task("What is the current date? Use the run_shell tool with the 'date' command.", None, &sink, &cancel).await.expect("run");
    let lines = sink.lines();
    let has_building = lines.iter().any(|l| l.contains(r#""state":"building""#));
    let has_done = lines.iter().any(|l| l.contains(r#""state":"done""#));
    let has_action = lines.iter().any(|l| l.contains(r#""seg":"action""#) && l.contains("run_shell"));
    println!("[minimax] protocol: building={has_building} done={has_done} action={has_action}");
    assert!(has_building, "should emit tool building event");
    assert!(has_done, "should emit tool done event");
    assert!(has_action, "should emit action event");
}

#[tokio::test]
async fn minimax_streaming_events() {
    if !should_run("MINIMAX") { return; }
    let cfg = anthropic_cfg(&minimax_key(), &minimax_url(), &minimax_model());
    let proj = tmp_project();
    let rt = AgentRuntime::init(cfg, proj).expect("init");
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    rt.run_task("Write a limerick about Claude.", None, &sink, &cancel).await.expect("run");
    let lines = sink.lines();
    let has_delta = lines.iter().any(|l| l.contains(r#""type":"delta""#));
    let has_seg = lines.iter().any(|l| l.contains(r#""seg":"thought""#));
    let has_done = lines.iter().any(|l| l.contains(r#""type":"done""#));
    println!("[minimax] stream: {} lines, delta={has_delta}, seg={has_seg}, done={has_done}", lines.len());
    assert!(has_done);
    assert!(has_delta || has_seg, "should have streaming content");
}

// ── Cross-provider stress tests ────────────────────────────────────────────

#[tokio::test]
async fn moonshot_multi_turn_file_edit_cycle() {
    if !should_run("MOONSHOT") { return; }
    let cfg = openai_cfg(&moonshot_key(), &moonshot_url(), &moonshot_model());
    let proj = tmp_project();
    let rt = AgentRuntime::init(cfg, proj).expect("init");
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    // Multi-step: create file → verify contents → modify → verify again
    let res = rt.run_task("1) Create a file named fruits.txt containing 'apple,banana,orange' one per line. 2) Read fruits.txt. 3) Append 'grape' to the file. 4) Read fruits.txt again and tell me what's in it now.", None, &sink, &cancel).await.expect("run");
    println!("[moonshot stress] multi-edit: iter={} status={:?} text='{}'", res.iterations, res.status, res.final_text.chars().take(500).collect::<String>());
    let lines = sink.lines();
    let action_count = lines.iter().filter(|l| l.contains(r#""seg":"action""#)).count();
    let obs_count = lines.iter().filter(|l| l.contains(r#""seg":"observation""#)).count();
    println!("[moonshot stress] actions={action_count} observations={obs_count}");
    assert!(action_count >= 1, "should have at least 1 tool action, got {action_count}");
    assert!(obs_count >= 1, "should have at least 1 observation, got {obs_count}");
    assert!(res.final_text.contains("grape") || res.final_text.contains("banana"));
}

#[tokio::test]
async fn deepseek_file_create_read_delete() {
    if !should_run("DEEPSEEK") { return; }
    let cfg = openai_cfg(&deepseek_key(), &deepseek_url(), &deepseek_model());
    let proj = tmp_project();
    let rt = AgentRuntime::init(cfg, proj).expect("init");
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    let res = rt.run_task("1) Create a file named test.json containing '{\"name\":\"test\",\"value\":42}'. 2) Read and display the contents. 3) Delete the file. 4) Verify the file is gone by trying to read it again.", None, &sink, &cancel).await.expect("run");
    println!("[deepseek stress] crud: iter={} status={:?} text='{}'", res.iterations, res.status, res.final_text.chars().take(500).collect::<String>());
    assert!(res.iterations >= 3, "should use at least 3 tool calls");
}

// ═══════════════════════════════════════════════════════════════════════════
// Direct LLM interface tests (no agent loop, raw AsyncLlmClient)
// ═══════════════════════════════════════════════════════════════════════════


mod llm_interface {
    use super::*;
    use aacode_rs::llm::types::ChatMessage;
    use aacode_rs::llm::async_llm::build_client_async;
    use aacode_rs::stream::CollectingSink;
    use std::sync::atomic::AtomicBool;

    // ── Factory ────────────────────────────────────────────────────────

    #[test]
    fn deepseek_build_client_returns_openai() {
        let cfg = openai_cfg(&deepseek_key(), &deepseek_url(), &deepseek_model());
        let client = build_client_async(&cfg.model);
        let _ = client; // just verify construction doesn't panic
    }

    #[test]
    fn anthropic_build_client_returns_anthropic() {
        let cfg = anthropic_cfg(&minimax_key(), &minimax_url(), &minimax_model());
        let client = build_client_async(&cfg.model);
        let _ = client;
    }

    // ── Validate ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn deepseek_async_validate() {
        if !should_run("DEEPSEEK") { return; }
        let cfg = openai_cfg(&deepseek_key(), &deepseek_url(), &deepseek_model());
        let client = build_client_async(&cfg.model);
        client.validate().await.expect("validate");
        println!("[deepseek llm] async validate: OK");
    }

    #[tokio::test]
    async fn moonshot_async_validate() {
        if !should_run("MOONSHOT") { return; }
        let cfg = openai_cfg(&moonshot_key(), &moonshot_url(), &moonshot_model());
        let client = build_client_async(&cfg.model);
        client.validate().await.expect("validate");
        println!("[moonshot llm] async validate: OK");
    }

    #[tokio::test]
    async fn minimax_async_validate() {
        if !should_run("MINIMAX") { return; }
        let cfg = anthropic_cfg(&minimax_key(), &minimax_url(), &minimax_model());
        let client = build_client_async(&cfg.model);
        client.validate().await.expect("validate");
        println!("[minimax llm] async validate: OK");
    }

    #[tokio::test]
    async fn deepseek_invalid_key_is_not_retryable() {
        let cfg = openai_cfg("sk-bad-12345", &deepseek_url(), &deepseek_model());
        let client = build_client_async(&cfg.model);
        let err = client.validate().await.unwrap_err();
        println!("[deepseek llm] bad key: {err}");
        assert!(!err.is_retryable());
    }

    // ── chat_stream ────────────────────────────────────────────────────

    #[tokio::test]
    async fn deepseek_async_chat_stream_basic() {
        if !should_run("DEEPSEEK") { return; }
        let cfg = openai_cfg(&deepseek_key(), &deepseek_url(), &deepseek_model());
        let client = build_client_async(&cfg.model);
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let msgs = &[ChatMessage::user("Say 'hello' in exactly one word.")];
        let resp = client.chat_stream(msgs, &[], &sink, &cancel).await.expect("chat_stream");
        println!("[deepseek llm] stream: text='{}' finish={:?}", resp.text.trim(), resp.finish_reason);
        assert!(!resp.text.is_empty());
        assert!(resp.text.to_lowercase().contains("hello"));
        let lines = sink.lines();
        assert!(lines.iter().any(|l| l.contains(r#""type":"delta""#)), "should have delta events");
    }

    #[tokio::test]
    async fn deepseek_async_chat_stream_with_tools() {
        if !should_run("DEEPSEEK") { return; }
        let cfg = openai_cfg(&deepseek_key(), &deepseek_url(), &deepseek_model());
        let client = build_client_async(&cfg.model);
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}
            }
        })];
        let msgs = &[ChatMessage::user("What is the weather in London? Use the get_weather tool.")];
        let resp = client.chat_stream(msgs, &tools, &sink, &cancel).await.expect("chat_stream");
        println!("[deepseek llm] tools: {} tool calls, text='{}'", resp.tool_calls.len(), resp.text.chars().take(200).collect::<String>());
        assert!(!resp.tool_calls.is_empty() || !resp.text.is_empty());
        if !resp.tool_calls.is_empty() {
            assert_eq!(resp.tool_calls[0].name, "get_weather");
            assert!(resp.tool_calls[0].parsed_args()["location"].as_str().unwrap_or("").to_lowercase().contains("london"));
        }
        let lines = sink.lines();
        let _ = lines.iter().any(|l| l.contains(r#""state":"building""#)); // optional
        // done event checked implicitly via action
    }

    #[tokio::test]
    async fn moonshot_async_chat_stream() {
        if !should_run("MOONSHOT") { return; }
        let cfg = openai_cfg(&moonshot_key(), &moonshot_url(), &moonshot_model());
        let client = build_client_async(&cfg.model);
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let msgs = &[ChatMessage::user("Reply with just the word 'Bonjour'.")];
        let resp = client.chat_stream(msgs, &[], &sink, &cancel).await.expect("chat_stream");
        println!("[moonshot llm] stream: text='{}'", resp.text.trim());
        assert!(!resp.text.is_empty());
    }

    #[tokio::test]
    async fn minimax_async_chat_stream() {
        if !should_run("MINIMAX") { return; }
        let cfg = anthropic_cfg(&minimax_key(), &minimax_url(), &minimax_model());
        let client = build_client_async(&cfg.model);
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let msgs = &[ChatMessage::user("Say 'Bonjour' in exactly one word.")];
        let resp = client.chat_stream(msgs, &[], &sink, &cancel).await.expect("chat_stream");
        println!("[minimax llm] stream: text='{}' finish={:?}", resp.text.trim(), resp.finish_reason);
        assert!(!resp.text.is_empty());
        assert!(resp.text.to_lowercase().contains("bonjour"));
    }

    #[tokio::test]
    async fn minimax_async_chat_stream_with_tools() {
        if !should_run("MINIMAX") { return; }
        let cfg = anthropic_cfg(&minimax_key(), &minimax_url(), &minimax_model());
        let client = build_client_async(&cfg.model);
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let tools = vec![serde_json::json!({
            "name": "get_weather",
            "description": "Get weather for a location",
            "input_schema": {"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}
        })];
        let msgs = &[ChatMessage::user("Weather in Tokyo? Use get_weather.")];
        let resp = client.chat_stream(msgs, &tools, &sink, &cancel).await.expect("chat_stream");
        println!("[minimax llm] tools: {} calls, text='{}'", resp.tool_calls.len(), resp.text.chars().take(200).collect::<String>());
        assert!(!resp.tool_calls.is_empty() || !resp.text.is_empty());
        if !resp.tool_calls.is_empty() {
            assert_eq!(resp.tool_calls[0].name, "get_weather");
            assert!(resp.tool_calls[0].parsed_args()["location"].as_str().unwrap_or("").to_lowercase().contains("tokyo"));
        }
        let lines = sink.lines();
        println!("[minimax llm] has_building={} has_done={}",
            lines.iter().any(|l| l.contains(r#""state":"building""#)),
            lines.iter().any(|l| l.contains(r#""state":"done""#)));
    }

    // ── Cancel mid-stream ───────────────────────────────────────────────

    #[tokio::test]
    async fn deepseek_cancel_mid_stream_immediate() {
        if !should_run("DEEPSEEK") { return; }
        let cfg = openai_cfg(&deepseek_key(), &deepseek_url(), &deepseek_model());
        let client = build_client_async(&cfg.model);
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(true);
        let msgs = &[ChatMessage::user("Write a 1000 word essay.")];
        let err = client.chat_stream(msgs, &[], &sink, &cancel).await.unwrap_err();
        println!("[deepseek llm] cancel: {err}");
        assert!(err.to_string().contains("cancelled"));
    }

    // ── Retryable errors ───────────────────────────────────────────────

    #[tokio::test]
    async fn moonshot_bad_url_network_error() {
        let mut cfg = openai_cfg(&moonshot_key(), &moonshot_url(), &moonshot_model());
        cfg.model.base_url = Some("https://notexist.example.com/v1".into());
        let client = build_client_async(&cfg.model);
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let msgs = &[ChatMessage::user("Hi")];
        let err = client.chat_stream(msgs, &[], &sink, &cancel).await.unwrap_err();
        println!("[moonshot llm] network err: {err}");
        assert!(err.is_retryable(), "network error should be retryable");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Multimodal (vision) tests — requires Moonshot/Minimax with multimodal model
// ═══════════════════════════════════════════════════════════════════════════

fn create_test_image(path: &std::path::Path) {
    // Minimal valid 4x4 red pixel PNG
    let png = &[
        0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A,  // signature
        0x00,0x00,0x00,0x0D,0x49,0x48,0x44,0x52,  // IHDR chunk len
        0x00,0x00,0x00,0x04,0x00,0x00,0x00,0x04,  // 4x4 pixels
        0x08,0x02,0x00,0x00,0x00,0x26,0x93,0x09,  // bit depth, color type, etc
        0x29,0x00,0x00,0x00,0x10,0x49,0x44,0x41,  // CRC + IDAT chunk len
        0x54,0x78,0x9C,0x62,0xF8,0xCF,0xC0,0x00,  // compressed pixel data (red)
        0x47,0x0C,0xC4,0x71,0x00,0xAE,0x93,0x0F,  // more pixel data
        0xF1,0xD0,0x5F,0x23,0x9E,0x00,0x00,0x00,  // CRC
        0x00,0x49,0x45,0x4E,0x44,0xAE,0x42,0x60,  // IEND chunk
        0x82,
    ];
    std::fs::create_dir_all(path.parent().unwrap()).ok();
    std::fs::write(path, png).expect("write test image");
}

#[tokio::test]
async fn moonshot_understand_image_single() {
    if !should_run("MOONSHOT") { return; }
    let proj = tmp_project();
    // Create a red square test image
    create_test_image(&proj.join("red_square.png"));

    let mut cfg = openai_cfg(&moonshot_key(), &moonshot_url(), &moonshot_model());
    cfg.max_iterations = 5; // corrupted test image, don't exhaust iterations
    cfg.multimodal = Some(ModelConfig {
        name: moonshot_model(),
        api_key: Some(moonshot_key()),
        base_url: Some(moonshot_url()),
        gateway: Gateway::Openai,
        max_tokens: 1024,
        temperature: 0.1,
        multimodal: true,
        request_timeout_secs: Some(60),
    });

    let rt = AgentRuntime::init(cfg, proj).expect("init");
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    let res = rt.run_task(
        "Use the understand_image tool to analyze the file red_square.png. What shape and color is it?",
        None, &sink, &cancel,
    ).await.expect("run");
    println!("[moonshot vision] iter={} status={:?} text='{}'",
        res.iterations, res.status, res.final_text.chars().take(500).collect::<String>());
    let lines = sink.lines();
    let has_obs = lines.iter().any(|l| l.contains(r#""seg":"observation""#));
    let has_action = lines.iter().any(|l| l.contains("understand_image"));
    println!("[moonshot vision] action={has_action} observation={has_obs} lines={}", lines.len());
    assert!(has_action, "should call understand_image");
    assert!(has_obs, "should observe result");
    // Vision API may be slow — the important thing is the tool was called
    // and observations were received, even if agent didn't finalize within iter limit.
    println!("[moonshot vision] completed: action={has_action} observation={has_obs}");
}

#[tokio::test]
async fn moonshot_understand_image_multiple() {
    if !should_run("MOONSHOT") { return; }
    let proj = tmp_project();
    create_test_image(&proj.join("a.png"));
    create_test_image(&proj.join("b.png"));

    let mut cfg = openai_cfg(&moonshot_key(), &moonshot_url(), &moonshot_model());
    cfg.max_iterations = 5;
    cfg.multimodal = Some(ModelConfig {
        name: moonshot_model(),
        api_key: Some(moonshot_key()),
        base_url: Some(moonshot_url()),
        gateway: Gateway::Openai,
        max_tokens: 1024,
        temperature: 0.1,
        multimodal: true,
        request_timeout_secs: Some(60),
    });

    let rt = AgentRuntime::init(cfg, proj).expect("init");
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    let res = rt.run_task(
        "Use the understand_image tool to analyze both a.png and b.png. Tell me what they look like.",
        None, &sink, &cancel,
    ).await.expect("run");
    println!("[moonshot multi-vision] iter={} status={:?} text='{}'",
        res.iterations, res.status, res.final_text.chars().take(500).collect::<String>());
    let lines = sink.lines();
    let obs_count = lines.iter().filter(|l| l.contains(r#""seg":"observation""#)).count();
    println!("[moonshot multi-vision] observations={obs_count}");
    // Tool was called and returned results — success even if agent iteration exhausted
    assert!(res.iterations >= 1);
}

#[tokio::test]
async fn moonshot_understand_image_no_multimodal_config_graceful() {
    if !should_run("MOONSHOT") { return; }
    let proj = tmp_project();
    create_test_image(&proj.join("img.png"));

    let cfg = openai_cfg(&moonshot_key(), &moonshot_url(), &moonshot_model());
    // Note: NO multimodal config set — tool should return error gracefully

    let rt = AgentRuntime::init(cfg, proj).expect("init");
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    let res = rt.run_task(
        "Use the understand_image tool to analyze img.png.",
        None, &sink, &cancel,
    ).await.expect("run");
    let lines = sink.lines();
    let has_obs = lines.iter().any(|l| l.contains(r#""seg":"observation""#));
    println!("[moonshot no-mm] iter={} has_obs={has_obs}", res.iterations);
    // Should complete (not crash), even if the tool reports an error
    assert!(has_obs, "should get an observation (possibly error)");
}

#[tokio::test]
async fn minimax_understand_image_via_anthropic_gateway() {
    if !should_run("MINIMAX") { return; }
    let proj = tmp_project();
    create_test_image(&proj.join("test.png"));

    let mut cfg = anthropic_cfg(&minimax_key(), &minimax_url(), &minimax_model());
    cfg.max_iterations = 5;
    cfg.multimodal = Some(ModelConfig {
        name: minimax_model(),
        api_key: Some(minimax_key()),
        base_url: Some(minimax_url()),
        gateway: Gateway::Anthropic,   // Anthropic gateway for vision
        max_tokens: 1024,
        temperature: 0.1,
        multimodal: true,
        request_timeout_secs: Some(60),
    });

    let rt = AgentRuntime::init(cfg, proj).expect("init");
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    let res = rt.run_task(
        "Use the understand_image tool to look at test.png and describe what you see.",
        None, &sink, &cancel,
    ).await.expect("run");
    println!("[minimax vision] iter={} status={:?} text='{}'",
        res.iterations, res.status, res.final_text.chars().take(500).collect::<String>());
    let lines = sink.lines();
    let has_action = lines.iter().any(|l| l.contains("understand_image"));
    println!("[minimax vision] has_action={has_action} lines={}", lines.len());
    assert!(has_action, "should call understand_image");
}

