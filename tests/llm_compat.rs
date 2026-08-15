// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Deep compatibility tests for the LLM stream parsers.
//!
//! Unlike the inline unit tests (which use hand-crafted minimal SSE), these
//! fixtures reproduce the *actual wire formats* observed from real providers:
//!   - OpenAI GPT-4o (role in first delta, `finish_reason:null` chunks, usage)
//!   - DeepSeek deepseek-chat + deepseek-reasoner (reasoning_content)
//!   - Kimi/Moonshot (tool_calls split mid-argument across many chunks)
//!   - Anthropic Claude (message_start/ping/content_block_*, message_delta usage)
//!   - MiniMax (Anthropic-compatible)
//!
//! Goal: prove `parse_openai_stream` / `parse_anthropic_stream` handle the real
//! quirks — multibyte content, null fields, keep-alive pings, multi-chunk
//! argument fragments, interleaved reasoning+content, empty finish chunks.

use aacode_rs::llm::openai::parse_openai_stream;
use aacode_rs::llm::anthropic::parse_anthropic_stream;
use aacode_rs::stream::CollectingSink;
use std::io::Cursor;
use std::sync::atomic::AtomicBool;

fn run_openai(raw: &str) -> (aacode_rs::llm::LlmResponse, Vec<String>) {
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    let resp = parse_openai_stream(Cursor::new(raw.as_bytes().to_vec()), &sink, &cancel).unwrap();
    (resp, sink.lines())
}

fn run_anthropic(raw: &str) -> (aacode_rs::llm::LlmResponse, Vec<String>) {
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    let resp =
        parse_anthropic_stream(Cursor::new(raw.as_bytes().to_vec()), &sink, &cancel).unwrap();
    (resp, sink.lines())
}

// ─────────────────────────── OpenAI / GPT-4o ───────────────────────────

/// Real GPT-4o pattern: first chunk carries `role:"assistant"` with empty
/// content, subsequent chunks carry content deltas, a chunk with
/// `finish_reason:null`, a final chunk with `finish_reason:"stop"` and empty
/// delta, then a usage-only chunk, then [DONE].
#[test]
fn openai_gpt4o_realistic_text() {
    let raw = concat!(
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\", world\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":3,\"total_tokens\":13}}\n\n",
        "data: [DONE]\n\n"
    );
    let (resp, _) = run_openai(raw);
    assert_eq!(resp.text, "Hello, world");
    assert!(resp.tool_calls.is_empty());
    assert_eq!(resp.finish_reason.as_deref(), Some("stop"));
}

/// Multibyte (Chinese + emoji) content must be reassembled correctly, even when
/// a provider splits a multi-byte character across... no — providers never
/// split inside a UTF-8 char in JSON strings, but the assembled text must be
/// preserved exactly.
#[test]
fn openai_multibyte_content() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"你好\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"，世界 \"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"🚀\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    let (resp, _) = run_openai(raw);
    assert_eq!(resp.text, "你好，世界 🚀");
}

/// DeepSeek reasoner: reasoning_content arrives first (possibly many chunks),
/// then content. reasoning and text must be separated.
#[test]
fn openai_deepseek_reasoner_split() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"Let me\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\" think step\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\" by step.\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"The answer\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\" is 42.\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    let (resp, lines) = run_openai(raw);
    assert_eq!(resp.reasoning_content.as_deref(), Some("Let me think step by step."));
    assert_eq!(resp.text, "The answer is 42.");
    // thinking seg emitted before thought seg
    let ti = lines.iter().position(|l| l.contains(r#""seg":"thinking""#)).unwrap();
    let to = lines.iter().position(|l| l.contains(r#""seg":"thought""#)).unwrap();
    assert!(ti < to);
}

/// Kimi/Moonshot tool call where arguments are split across MANY chunks and the
/// name arrives only in the first fragment. id only in first fragment.
#[test]
fn openai_kimi_tool_call_heavy_fragmentation() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"index\":0,\"id\":\"call_abc\",\"type\":\"function\",\"function\":{\"name\":\"run_shell\",\"arguments\":\"\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"comm\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"and\\\":\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"ls -la\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"}\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    let (resp, _) = run_openai(raw);
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].id, "call_abc");
    assert_eq!(resp.tool_calls[0].name, "run_shell");
    assert_eq!(resp.tool_calls[0].arguments, "{\"command\":\"ls -la\"}");
    // args must parse as valid JSON
    assert_eq!(resp.tool_calls[0].parsed_args()["command"], "ls -la");
}

/// Multiple parallel tool calls in one response (index 0 and 1).
#[test]
fn openai_parallel_tool_calls() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c0\",\"function\":{\"name\":\"run_shell\",\"arguments\":\"{\\\"command\\\":\\\"ls\\\"}\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"c1\",\"function\":{\"name\":\"get_todo_summary\",\"arguments\":\"{}\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    let (resp, _) = run_openai(raw);
    assert_eq!(resp.tool_calls.len(), 2);
    assert_eq!(resp.tool_calls[0].name, "run_shell");
    assert_eq!(resp.tool_calls[1].name, "get_todo_summary");
    assert_eq!(resp.tool_calls[0].id, "c0");
    assert_eq!(resp.tool_calls[1].id, "c1");
}

/// Text content THEN a tool call in the same response (model narrates then acts).
#[test]
fn openai_text_then_tool_call() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"Let me check the files.\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"run_shell\",\"arguments\":\"{\\\"command\\\":\\\"ls\\\"}\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    let (resp, _) = run_openai(raw);
    assert_eq!(resp.text, "Let me check the files.");
    assert_eq!(resp.tool_calls.len(), 1);
}

/// Keep-alive comment lines (`: OPENROUTER PROCESSING`) and blank lines between
/// events must be ignored gracefully.
#[test]
fn openai_keepalive_and_blank_lines() {
    let raw = concat!(
        ": OPENROUTER PROCESSING\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n",
        ": keep-alive\n",
        "\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    let (resp, _) = run_openai(raw);
    assert_eq!(resp.text, "ok");
}

/// Malformed/partial JSON chunk in the middle must be skipped, not abort.
#[test]
fn openai_malformed_chunk_skipped() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
        "data: {this is not valid json\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    let (resp, _) = run_openai(raw);
    assert_eq!(resp.text, "ab");
}

/// Truncation: finish_reason "length" must set the flag and append a warning.
#[test]
fn openai_length_truncation_warning() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"partial answer\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    let (resp, _) = run_openai(raw);
    assert!(resp.is_truncated());
    assert!(resp.text.contains("truncated"));
}

/// Stream that ends WITHOUT [DONE] (connection closed / EOF) must still return
/// the accumulated content.
#[test]
fn openai_eof_without_done() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"incomplete\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\" but usable\"}}]}\n"
    );
    let (resp, _) = run_openai(raw);
    assert_eq!(resp.text, "incomplete but usable");
}

/// Empty stream (immediate [DONE]) → empty response, no panic.
#[test]
fn openai_empty_stream() {
    let (resp, lines) = run_openai("data: [DONE]\n\n");
    assert_eq!(resp.text, "");
    assert!(resp.tool_calls.is_empty());
    // thought seg still emitted (empty) so UI knows the segment ended
    assert!(lines.iter().any(|l| l.contains(r#""seg":"thought""#)));
}

/// In-stream error object (HTTP 200 but a streamed failure) must surface as an
/// error, not be silently dropped. (Learned from rust-genai.)
#[test]
fn openai_in_stream_error() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n",
        "data: {\"error\":{\"message\":\"Error in input stream\",\"type\":\"server_error\"}}\n\n",
        "data: [DONE]\n\n"
    );
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    let r = parse_openai_stream(Cursor::new(raw.as_bytes().to_vec()), &sink, &cancel);
    assert!(r.is_err(), "expected an error result");
    let msg = format!("{}", r.err().unwrap());
    assert!(msg.contains("Error in input stream"), "got: {msg}");
}

/// Some providers (Ollama, proxies) emit reasoning under `delta.reasoning`
/// instead of `delta.reasoning_content`. Both must populate reasoning_content.
#[test]
fn openai_reasoning_alt_field() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"reasoning\":\"pondering\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n",
        "data: [DONE]\n\n"
    );
    let (resp, _) = run_openai(raw);
    assert_eq!(resp.reasoning_content.as_deref(), Some("pondering"));
    assert_eq!(resp.text, "answer");
}

/// Provider that sends content AND finish_reason in the SAME chunk (Mistral-style)
/// must not lose that final content.
#[test]
fn openai_content_with_finish_reason_same_chunk() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"first \"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"last\"},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    let (resp, _) = run_openai(raw);
    assert_eq!(resp.text, "first last");
    assert_eq!(resp.finish_reason.as_deref(), Some("stop"));
}


// ─────────────────────────── Anthropic / Claude ───────────────────────────

/// Realistic Claude stream: message_start, content_block_start(text), ping,
/// content_block_delta(text_delta)×N, content_block_stop, message_delta(stop),
/// message_stop.
#[test]
fn anthropic_claude_realistic_text() {
    let raw = concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"role\":\"assistant\",\"content\":[]}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: ping\ndata: {\"type\":\"ping\"}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" 世界\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
    );
    let (resp, _) = run_anthropic(raw);
    assert_eq!(resp.text, "Hello 世界");
    assert_eq!(resp.finish_reason.as_deref(), Some("end_turn"));
}

/// Claude with extended thinking: content_block(thinking) then content_block(text).
#[test]
fn anthropic_extended_thinking() {
    let raw = concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Considering options\"}}\n\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Final answer\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n"
    );
    let (resp, _) = run_anthropic(raw);
    assert_eq!(resp.reasoning_content.as_deref(), Some("Considering options"));
    assert_eq!(resp.text, "Final answer");
}

/// Claude tool_use: content_block_start(tool_use with name+id) then
/// input_json_delta fragments assembling the JSON input.
#[test]
fn anthropic_tool_use_fragmented() {
    let raw = concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"run_shell\",\"input\":{}}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"command\\\"\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\": \\\"pytest -q\\\"}\"}}\n\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n"
    );
    let (resp, _) = run_anthropic(raw);
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].id, "toolu_1");
    assert_eq!(resp.tool_calls[0].name, "run_shell");
    assert_eq!(resp.tool_calls[0].parsed_args()["command"], "pytest -q");
}

/// Claude text block AND tool_use block in the same message (narrate + act).
#[test]
fn anthropic_text_and_tool_use() {
    let raw = concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Running tests.\"}}\n\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"run_shell\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"command\\\":\\\"ls\\\"}\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n"
    );
    let (resp, _) = run_anthropic(raw);
    assert_eq!(resp.text, "Running tests.");
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].name, "run_shell");
}

/// Claude tool_use with empty input (no input_json_delta) → arguments "{}".
#[test]
fn anthropic_tool_use_empty_input() {
    let raw = concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"get_todo_summary\"}}\n\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n"
    );
    let (resp, _) = run_anthropic(raw);
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].arguments, "{}");
    assert!(resp.tool_calls[0].parsed_args().is_object());
}

/// Claude max_tokens truncation.
#[test]
fn anthropic_max_tokens_truncation() {
    let raw = concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\"}}\n\n"
    );
    let (resp, _) = run_anthropic(raw);
    assert!(resp.is_truncated());
    assert!(resp.text.contains("truncated"));
}

/// MiniMax (Anthropic-compatible) — same event shapes as Claude.
#[test]
fn anthropic_minimax_compatible() {
    let raw = concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"minimax reply\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n"
    );
    let (resp, _) = run_anthropic(raw);
    assert_eq!(resp.text, "minimax reply");
}

/// Empty anthropic stream.
#[test]
fn anthropic_empty_stream() {
    let (resp, _) = run_anthropic("data: [DONE]\n\n");
    assert_eq!(resp.text, "");
    assert!(resp.tool_calls.is_empty());
}

/// Anthropic in-stream error event must surface as an error.
#[test]
fn anthropic_in_stream_error() {
    let raw = concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"content\":[]}}\n\n",
        "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n"
    );
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    let r = parse_anthropic_stream(Cursor::new(raw.as_bytes().to_vec()), &sink, &cancel);
    assert!(r.is_err());
    assert!(format!("{}", r.err().unwrap()).contains("Overloaded"));
}


// ─────────────── Live API compatibility (opt-in, requires key) ───────────────
//
// Run with a real key to prove wire-format compatibility against the actual
// provider, e.g.:
//   LLM_API_KEY=sk-xxx LLM_MODEL_NAME=deepseek-chat \
//   LLM_API_URL=https://api.deepseek.com/v1 \
//   cargo test -p aacode-rs --test llm_compat -- --ignored --nocapture
//
// These are #[ignore] so CI/offline runs skip them.

use aacode_rs::config::{Gateway, ModelConfig};
use aacode_rs::llm::{build_client, ChatMessage};

#[tokio::test]
#[ignore = "requires a real LLM_API_KEY"]
async fn live_openai_streams_text() {
    let mut model = ModelConfig {
        name: std::env::var("LLM_MODEL_NAME").unwrap_or_else(|_| "deepseek-chat".into()),
        api_key: std::env::var("LLM_API_KEY").ok(),
        base_url: std::env::var("LLM_API_URL").ok(),
        gateway: Gateway::Openai,
        ..Default::default()
    };
    model.apply_env();
    let client = build_client(&model);
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    let msgs = vec![
        ChatMessage::system("Answer in one short word."),
        ChatMessage::user("Say 'pong'."),
    ];
    let resp = client
        .chat_stream(&msgs, &[], &sink, &cancel)
        .await
        .expect("live call failed");
    eprintln!("LIVE text = {:?}", resp.text);
    assert!(!resp.text.trim().is_empty(), "expected non-empty response");
}

#[tokio::test]
#[ignore = "requires a real LLM_API_KEY"]
async fn live_openai_tool_call() {
    let mut model = ModelConfig {
        name: std::env::var("LLM_MODEL_NAME").unwrap_or_else(|_| "deepseek-chat".into()),
        api_key: std::env::var("LLM_API_KEY").ok(),
        base_url: std::env::var("LLM_API_URL").ok(),
        gateway: Gateway::Openai,
        ..Default::default()
    };
    model.apply_env();
    let client = build_client(&model);
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    // Provide a run_shell tool and ask the model to use it.
    let tools = vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Execute a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            }
        }
    })];
    let msgs = vec![
        ChatMessage::system("You must use the run_shell tool."),
        ChatMessage::user("Run `ls` in the current directory using the tool."),
    ];
    let resp = client
        .chat_stream(&msgs, &tools, &sink, &cancel)
        .await
        .expect("live call failed");
    eprintln!("LIVE tool_calls = {:?}", resp.tool_calls);
    assert!(!resp.tool_calls.is_empty(), "expected a tool call");
    assert_eq!(resp.tool_calls[0].name, "run_shell");
    // arguments must be valid JSON containing a command
    assert!(resp.tool_calls[0].parsed_args().get("command").is_some());
}

#[tokio::test]
#[ignore = "requires a real Anthropic-compatible LLM_API_KEY"]
async fn live_anthropic_streams_text() {
    let mut model = ModelConfig {
        name: std::env::var("LLM_MODEL_NAME").unwrap_or_else(|_| "claude-3-5-sonnet-20241022".into()),
        api_key: std::env::var("LLM_API_KEY").ok(),
        base_url: std::env::var("LLM_API_URL").ok(),
        gateway: Gateway::Anthropic,
        ..Default::default()
    };
    model.apply_env();
    model.gateway = Gateway::Anthropic;
    let client = build_client(&model);
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    let msgs = vec![
        ChatMessage::system("Answer in one short word."),
        ChatMessage::user("Say 'pong'."),
    ];
    let resp = client
        .chat_stream(&msgs, &[], &sink, &cancel)
        .await
        .expect("live call failed");
    eprintln!("LIVE anthropic text = {:?}", resp.text);
    assert!(!resp.text.trim().is_empty());
}

#[tokio::test]
#[ignore = "requires a real Anthropic-compatible LLM_API_KEY"]
async fn live_anthropic_tool_call() {
    let mut model = ModelConfig {
        name: std::env::var("LLM_MODEL_NAME").unwrap_or_else(|_| "claude-3-5-sonnet-20241022".into()),
        api_key: std::env::var("LLM_API_KEY").ok(),
        base_url: std::env::var("LLM_API_URL").ok(),
        gateway: Gateway::Anthropic,
        ..Default::default()
    };
    model.apply_env();
    model.gateway = Gateway::Anthropic;
    let client = build_client(&model);
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    let tools = vec![serde_json::json!({
        "name": "run_shell",
        "description": "Execute a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"]
        }
    })];
    let msgs = vec![
        ChatMessage::system("You must use the run_shell tool to answer."),
        ChatMessage::user("Run `ls` in the current directory using the run_shell tool."),
    ];
    let resp = client
        .chat_stream(&msgs, &tools, &sink, &cancel)
        .await
        .expect("live call failed");
    eprintln!("LIVE anthropic tool_calls = {:?}", resp.tool_calls);
    eprintln!("LIVE anthropic finish_reason = {:?}", resp.finish_reason);
    assert!(!resp.tool_calls.is_empty(), "expected a tool call");
    assert_eq!(resp.tool_calls[0].name, "run_shell");
    // input_json_delta must have assembled into valid JSON with a command
    assert!(resp.tool_calls[0].parsed_args().get("command").is_some());
}



// ──────── regression / cross-cutting ────────

fn parse_oai(raw: &str) -> aacode_rs::llm::types::LlmResponse {
    use aacode_rs::llm::openai::parse_openai_stream;
    use aacode_rs::stream::CollectingSink;
    use std::io::Cursor;
    let sink = CollectingSink::new(false);
    let cancel = std::sync::atomic::AtomicBool::new(false);
    parse_openai_stream(Cursor::new(raw.as_bytes().to_vec()), &sink, &cancel).unwrap()
}

fn parse_anth(raw: &str) -> aacode_rs::llm::types::LlmResponse {
    use aacode_rs::llm::anthropic::parse_anthropic_stream;
    use aacode_rs::stream::CollectingSink;
    use std::io::Cursor;
    let sink = CollectingSink::new(false);
    let cancel = std::sync::atomic::AtomicBool::new(false);
    parse_anthropic_stream(Cursor::new(raw.as_bytes().to_vec()), &sink, &cancel).unwrap()
}

#[test]
fn oai_tool_call_defaults_index_to_zero_when_missing() {
    // Some local models / proxies omit `index` entirely.
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"id\":\"t1\",\"type\":\"function\",\"function\":{\"name\":\"run_shell\",\"arguments\":\"{\\\"cmd\\\":\\\"ls\\\"}\"}}]}}]}\n\n",
        "data: [DONE]\n\n"
    );
    let r = parse_oai(raw);
    assert_eq!(r.tool_calls.len(), 1);
    assert_eq!(r.tool_calls[0].name, "run_shell");
}

#[test]
fn oai_reasoning_fallback_to_reasoning_field() {
    // Ollama / proxy-style `reasoning` (not reasoning_content).
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"reasoning\":\"think hard\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"done\"}}]}\n\n",
        "data: [DONE]\n\n"
    );
    let r = parse_oai(raw);
    assert_eq!(r.reasoning_content.as_deref(), Some("think hard"));
    assert_eq!(r.text, "done");
}

#[test]
fn oai_empty_content_skipped() {
    let raw = "data: {\"choices\":[{\"delta\":{\"content\":\"\"}}]}\n\ndata: [DONE]\n\n";
    let r = parse_oai(raw);
    assert_eq!(r.text, "");
}

// (anthropic wire-format tests are in aacode-rs/src/llm/anthropic.rs)

#[test]
fn anth_stream_tool_use_partial_json_accumulates() {
    let raw = concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"run_shell\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"cmd\\\"\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\": \\\"ls\\\"}\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n",
        "data: [DONE]\n\n"
    );
    let r = parse_anth(raw);
    assert_eq!(r.tool_calls.len(), 1);
    assert_eq!(r.tool_calls[0].parsed_args()["cmd"], "ls");
}
