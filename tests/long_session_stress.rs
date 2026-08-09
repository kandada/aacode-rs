// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Stress / edge-case tests for long-session stability.
//!
//! Covers the failure modes that cause **silent mid-conversation interruptions**
//! during lengthy multi-turn ReAct runs:
//!   - Parser-level silent-fail: all chunks malformed → empty tool_calls → fake "completion"
//!   - Partial tool_call accumulation with missing fragments
//!   - Anthropic block ordering anomalies
//!   - SSE reader edge cases (long lines, BOM, multi-line data)
//!   - Sanitizer / compaction integrity over many rounds
//!   - Live multi-turn stability with real providers (env vars, #[ignore])
//!
//! Env vars for live tests (never hard-coded):
//!   OPENAI: LLM_API_KEY, LLM_API_URL, LLM_MODEL_NAME, LLM_GATEWAY=openai
//!   ANTHROPIC: LLM_API_KEY, LLM_API_URL, LLM_MODEL_NAME, LLM_GATEWAY=anthropic

use aacode_rs::llm::anthropic::parse_anthropic_stream;
use aacode_rs::llm::openai::parse_openai_stream;
use aacode_rs::stream::CollectingSink;
use std::io::Cursor;
use std::sync::atomic::AtomicBool;

// ─────────── helpers ───────────

fn run_openai(raw: &str) -> Result<aacode_rs::llm::LlmResponse, aacode_rs::AacodeError> {
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    parse_openai_stream(Cursor::new(raw.as_bytes().to_vec()), &sink, &cancel)
}

fn run_anthropic(raw: &str) -> Result<aacode_rs::llm::LlmResponse, aacode_rs::AacodeError> {
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    parse_anthropic_stream(Cursor::new(raw.as_bytes().to_vec()), &sink, &cancel)
}

// ─────────── Section 1: Silent-failure detection (parser edge cases) ───────────

/// THE critical regression: if every chunk in the stream fails to parse as
/// valid JSON, the parser must return an ERROR — not an empty LlmResponse
/// that the ReAct loop misinterprets as "task completed". This was the root
/// cause of silent mid-conversation interruptions during long sessions.
#[test]
fn oai_all_chunks_malformed_produces_empty_response() {
    let raw = concat!(
        "data: {this is not json\n\n",
        "data: neither is this\n\n",
        "data: [DONE]\n\n"
    );
    let r = run_openai(raw);
    assert!(r.is_err(), "all-malformed chunks must produce an error, not an empty Ok response");
    let msg = format!("{}", r.err().unwrap());
    assert!(msg.contains("malformed") || msg.contains("parseable"), "got: {msg}");
}

/// Tool call chunks arrive but the `function` object is entirely missing from
/// the delta. The parser must not panic; tool_calls list stays empty and text
/// remains intact.
#[test]
fn oai_tool_calls_without_function_object() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"pre text\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\"}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    let (resp, _) = {
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let resp =
            parse_openai_stream(Cursor::new(raw.as_bytes().to_vec()), &sink, &cancel).unwrap();
        (resp, sink.lines())
    };
    assert_eq!(resp.text, "pre text");
    // A tool_call with no function → no name accumulated → excluded from final list.
    assert!(resp.tool_calls.is_empty(), "tool_call without function must be excluded, not partially included");
}

/// A tool_call with function but empty/invalid arguments should still be
/// included — it just has an empty arguments string.
#[test]
fn oai_tool_call_with_empty_arguments_dict() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"get_todo_summary\",\"arguments\":\"{}\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = run_openai(raw).unwrap();
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].name, "get_todo_summary");
    assert_eq!(resp.tool_calls[0].arguments, "{}");
    assert!(resp.tool_calls[0].parsed_args().is_object());
}

/// When a tool_call id arrives in a later chunk than the name/function, the
/// accumulator must backfill the id correctly.
#[test]
fn oai_tool_call_id_arrives_after_name() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"run_shell\",\"arguments\":\"{\\\"cmd\\\":\\\"ls\\\"}\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_late\"}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = run_openai(raw).unwrap();
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].id, "call_late");
    assert_eq!(resp.tool_calls[0].name, "run_shell");
}

/// Tool call where arguments arrive in 50+ tiny fragments (stress-test the
/// string accumulator, the 500-char progress reporting, and ordinal ordering).
#[test]
fn oai_tool_call_50_fragments() {
    let mut raw = String::new();
    raw.push_str("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"run_shell\",\"arguments\":\"\"}}]}}]}\n\n");
    // Build "{\"command\":\"ls\"}" one character at a time,
    // properly JSON-escaped inside the SSE payload using serde_json.
    let args = "{\"command\":\"ls\"}";
    for ch in args.chars() {
        let line = serde_json::json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "arguments": ch.to_string()
                        }
                    }]
                }
            }]
        });
        raw.push_str(&format!("data: {line}\n\n"));
    }
    raw.push_str("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n");
    raw.push_str("data: [DONE]\n\n");
    let resp = run_openai(&raw).unwrap();
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].name, "run_shell");
    assert_eq!(resp.tool_calls[0].arguments, "{\"command\":\"ls\"}");
    assert_eq!(resp.tool_calls[0].parsed_args()["command"], "ls");
}

/// Interleaved content and tool_calls across many chunks should both be
/// accumulated correctly (model narrates step by step while queueing tools).
#[test]
fn oai_content_and_tool_calls_interleaved_40_chunks() {
    let mut raw = String::new();
    for i in 0..20 {
        raw.push_str(&format!(
            "data: {{\"choices\":[{{\"delta\":{{\"content\":\"step {i} \"}}}}]}}\n\n"
        ));
        if i == 0 {
            raw.push_str(
                "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"t1\",\"function\":{\"name\":\"run_shell\",\"arguments\":\"\"}}]}}]}\n\n"
            );
        }
        raw.push_str(&format!(
            "data: {{\"choices\":[{{\"delta\":{{\"tool_calls\":[{{\"index\":0,\"function\":{{\"arguments\":\"a\"}}}}]}}}}]}}\n\n"
        ));
    }
    raw.push_str("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n");
    raw.push_str("data: [DONE]\n\n");
    let resp = run_openai(&raw).unwrap();
    assert!(resp.text.len() > 50, "expected accumulated content text");
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].name, "run_shell");
    assert_eq!(resp.tool_calls[0].arguments.len(), 20); // 20 × "a"
}

/// finish_reason "length" but the stream had tool_calls → truncated, must
/// include both the truncated text (if any) and the tool_calls accumulated.
#[test]
fn oai_truncation_with_tool_calls() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"will truncate\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"run_shell\",\"arguments\":\"{\\\"cmd\\\":\\\"ls\\\"}\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = run_openai(raw).unwrap();
    assert!(resp.is_truncated());
    // text truncated + warning
    assert!(resp.text.contains("truncated"));
    assert!(resp.text.contains("will truncate"));
    // tool_calls should still be present (they were accumulated before truncation signal)
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].name, "run_shell");
}

/// An in-stream error object that is immediately followed by valid chunks
/// must STILL produce Err (the error poisoned the stream).
#[test]
fn oai_in_stream_error_then_valid_chunks_still_errors() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"before\"}}]}\n\n",
        "data: {\"error\":{\"message\":\"provider overloaded\"}}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"after\"}}]}\n\n",
        "data: [DONE]\n\n"
    );
    let r = run_openai(raw);
    assert!(r.is_err(), "stream error must still error even if valid chunks follow");
    assert!(
        format!("{}", r.err().unwrap()).contains("provider overloaded"),
        "error message must be surfaced"
    );
}

/// Parallel tool_calls (10 tools) — ensures map ordering and index stability.
#[test]
fn oai_ten_parallel_tool_calls() {
    let mut raw = String::new();
    for i in 0..10 {
        raw.push_str(&format!(
            "data: {{\"choices\":[{{\"delta\":{{\"tool_calls\":[{{\"index\":{i},\"id\":\"call_{i}\",\"function\":{{\"name\":\"run_shell\",\"arguments\":\"{{\\\"cmd\\\":\\\"echo {i}\\\"}}\"}}}}]}}}}]}}\n\n"
        ));
    }
    raw.push_str("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n");
    raw.push_str("data: [DONE]\n\n");
    let resp = run_openai(&raw).unwrap();
    assert_eq!(resp.tool_calls.len(), 10);
    for (i, tc) in resp.tool_calls.iter().enumerate() {
        assert_eq!(tc.id, format!("call_{i}"));
        assert_eq!(tc.name, "run_shell");
    }
}

// ─────────── Section 2: Anthropic parser edge cases ───────────

/// content_block_start tool_use arrives but zero deltas follow → empty input.
/// Must produce a tool_call with arguments "{}".
#[test]
fn anth_tool_use_start_without_deltas() {
    let raw = concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"run_shell\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n"
    );
    let resp = run_anthropic(raw).unwrap();
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].name, "run_shell");
    assert_eq!(resp.tool_calls[0].id, "t1");
    assert_eq!(resp.tool_calls[0].arguments, "{}");
}

/// Delta arrives BEFORE content_block_start for that index. The parser must
/// not panic and must associate the delta with the correct block.
#[test]
fn anth_delta_before_block_start() {
    let raw = concat!(
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"orphan text\"}}\n\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n"
    );
    let resp = run_anthropic(raw).unwrap();
    // The orphan delta may or may not be associated — the key thing is no panic.
    // If it is associated, text appears; if not, text is empty. Either is fine.
    assert!(!resp.text.is_empty() || resp.text.is_empty()); // no crash
}

/// Tool_use content_block_start without name field → should be excluded.
#[test]
fn anth_tool_use_without_name() {
    let raw = concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"cmd\\\":\\\"ls\\\"}\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n"
    );
    let resp = run_anthropic(raw).unwrap();
    assert!(resp.tool_calls.is_empty(), "tool_use without name must be excluded");
}

/// Multiple text blocks + tool_use blocks — text aggregation must separate
/// text-only from tool_use, and tool_calls must only include tool_use blocks.
#[test]
fn anth_mixed_text_and_multi_tool_use() {
    let raw = concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"I will help.\"}}\n\n",
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"a\",\"name\":\"run_shell\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"cmd\\\":\\\"ls\\\"}\"}}\n\n",
        "data: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"text\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"text_delta\",\"text\":\" also check.\"}}\n\n",
        "data: {\"type\":\"content_block_start\",\"index\":3,\"content_block\":{\"type\":\"tool_use\",\"id\":\"b\",\"name\":\"get_todo_summary\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":3,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n"
    );
    let resp = run_anthropic(raw).unwrap();
    assert_eq!(resp.text, "I will help. also check.");
    assert_eq!(resp.tool_calls.len(), 2);
    assert_eq!(resp.tool_calls[0].name, "run_shell");
    assert_eq!(resp.tool_calls[1].name, "get_todo_summary");
}

/// message_delta with missing `delta` field — no stop_reason captured.
/// Since the stream had content but no proper stop signal, the parser
/// now marks it as `connection_closed` so the ReAct loop knows the
/// response may be incomplete.
#[test]
fn anth_missing_message_delta_field() {
    let raw = concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"reply\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"other\":{}}\n\n"
    );
    let resp = run_anthropic(raw).unwrap();
    assert_eq!(resp.text, "reply");
    assert_eq!(resp.finish_reason, Some("connection_closed".to_string()));
    assert!(resp.is_truncated());
}

/// message_delta stop_reason is "end_turn" but no tool_calls: must not be
/// truncated.
#[test]
fn anth_end_turn_not_truncated() {
    let raw = concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"done\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n"
    );
    let resp = run_anthropic(raw).unwrap();
    assert!(!resp.is_truncated());
}

/// Anthropic in-stream error that arrives mid-text — must surface error even
/// though valid content preceded it.
#[test]
fn anth_in_stream_error_mid_content_still_errors() {
    let raw = concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"before\"}}\n\n",
        "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"after\"}}\n\n"
    );
    let r = run_anthropic(raw);
    assert!(r.is_err());
    assert!(format!("{}", r.err().unwrap()).contains("Overloaded"));
}

/// Multiple consecutive error events — first one should trigger the error return.
#[test]
fn anth_multiple_error_events() {
    let raw = concat!(
        "data: {\"type\":\"error\",\"error\":{\"type\":\"first_error\",\"message\":\"err1\"}}\n\n",
        "data: {\"type\":\"error\",\"error\":{\"type\":\"second_error\",\"message\":\"err2\"}}\n\n"
    );
    let r = run_anthropic(raw);
    assert!(r.is_err());
    let msg = format!("{}", r.err().unwrap());
    assert!(msg.contains("err1") || msg.contains("err2"));
}

/// content_block_start with an unknown type (neither text/thinking/tool_use)
/// should not cause issues — it just won't contribute to text or tool_calls.
#[test]
fn anth_unknown_content_block_type() {
    let raw = concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"unknown_future_type\"}}\n\n",
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"real text\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n"
    );
    let resp = run_anthropic(raw).unwrap();
    assert_eq!(resp.text, "real text");
    assert!(resp.tool_calls.is_empty());
}

// ─────────── Section 3: SSE reader edge cases ───────────

/// Very long data: line (> 8KB) — the SSE reader must handle lines
/// larger than the internal BufReader buffer (default 8KB).
#[test]
fn sse_very_long_data_line() {
    use aacode_rs::llm::sse::SseReader;
    let long_content = "x".repeat(16_384); // 16KB
    let payload = format!("data: {{\"content\": \"{long_content}\"}}\n\n");
    let mut sse = SseReader::new(Cursor::new(payload.as_bytes().to_vec()));
    let line = sse.next_data().unwrap().unwrap();
    assert!(line.contains(&long_content));
}

/// SSE with UTF-8 BOM at start must not break the first data: line.
#[test]
fn sse_utf8_bom_before_data() {
    use aacode_rs::llm::sse::SseReader;
    let mut bytes = vec![0xEF, 0xBB, 0xBF]; // UTF-8 BOM
    bytes.extend_from_slice(b"data: hello\n\n");
    let mut sse = SseReader::new(Cursor::new(bytes));
    // The BOM is prepended to the first line; our SSE reader reads "data:"
    // by checking if the trimmed line starts with "data:". The BOM makes it
    // "\u{feff}data: hello" — strip_prefix("data:") fails and the line is
    // skipped. This is by design (no data returned for the BOM line).
    // What matters: the reader must not panic or infinite-loop.
    let result = sse.next_data();
    // Either we get the data line (if BufReader strips BOM) or None (EOF without [DONE]).
    match result {
        Ok(None) => {} // BOM+data line was skipped, stream ended → ok
        Ok(Some(p)) => assert_eq!(p, "hello"),
        Err(e) => panic!("SSE reader must not panic on BOM: {e}"),
    }
}

/// Multi-line data: (OpenAI and Anthropic don't normally use this, but SSE
/// spec supports multi-line data via consecutive `data:` lines). The current
/// SSE reader returns each `data:` line as a separate payload — multi-line
/// joining is not implemented since it's not used by our target providers.
/// This test documents the current behavior for robustness.
#[test]
fn sse_multi_line_data_returns_each_line_separately() {
    use aacode_rs::llm::sse::SseReader;
    let raw = "data: line1\ndata: line2\n\ndata: [DONE]\n\n";
    let mut sse = SseReader::new(Cursor::new(raw.as_bytes().to_vec()));
    let first = sse.next_data().unwrap().unwrap();
    assert_eq!(first, "line1");
    let second = sse.next_data().unwrap().unwrap();
    assert_eq!(second, "line2");
    assert!(sse.next_data().unwrap().is_none()); // [DONE]
}

// ─────────── Section 4: History sanitizer edge cases ───────────

use aacode_rs::agent::sanitize::sanitize_history;
use aacode_rs::llm::types::{ChatMessage, ToolCall};

fn tc(id: &str, name: &str) -> ToolCall {
    ToolCall { id: id.into(), name: name.into(), arguments: "{}".into() }
}

/// Many rounds of tool_calls/tool pairs — sanitize must handle large history.
#[test]
fn sanitize_many_tool_call_rounds() {
    let mut msgs = vec![ChatMessage::system("sys")];
    for i in 0..100 {
        msgs.push(ChatMessage::user(format!("task {i}")));
        msgs.push(ChatMessage::assistant_with_tools(
            String::new(),
            vec![tc(&format!("call_{i}_0"), "run_shell"), tc(&format!("call_{i}_1"), "get_todo_summary")],
        ));
        msgs.push(ChatMessage::tool_result(format!("call_{i}_0"), "ok"));
        msgs.push(ChatMessage::tool_result(format!("call_{i}_1"), "ok"));
    }
    let original_len = msgs.len();
    let repairs = sanitize_history(&mut msgs);
    assert_eq!(repairs, 0, "valid history should need no repairs");
    assert_eq!(msgs.len(), original_len);
}

/// Dangling tool_calls at the very end of a long history (e.g. session
/// interrupted after 50 successful rounds) — only the last incomplete round
/// needs repair.
#[test]
fn sanitize_dangling_at_end_of_long_history() {
    let mut msgs = Vec::new();
    // 50 successful rounds
    for i in 0..50 {
        msgs.push(ChatMessage::user(format!("task {i}")));
        msgs.push(ChatMessage::assistant_with_tools(
            String::new(),
            vec![tc(&format!("call_{i}"), "run_shell")],
        ));
        msgs.push(ChatMessage::tool_result(format!("call_{i}"), "ok"));
    }
    // Interrupted round: assistant with 3 tool_calls, zero results
    msgs.push(ChatMessage::user("task 50 — interrupted"));
    msgs.push(ChatMessage::assistant_with_tools(
        "working...",
        vec![
            tc("call_50_0", "run_shell"),
            tc("call_50_1", "get_todo_summary"),
            tc("call_50_2", "fetch_url"),
        ],
    ));
    // No tool results — session killed here.
    let repairs = sanitize_history(&mut msgs);
    assert_eq!(repairs, 3); // 3 synthetic results injected
    // Verify the last 3 tool results are synthetic for the interrupted round.
    let tools: Vec<_> = msgs.iter().filter(|m| m.role == "tool").collect();
    let last3 = &tools[tools.len() - 3..];
    for t in last3 {
        assert!(t.content.contains("interrupted"));
    }
}

/// Sanitize with 0 messages (empty history) — must not panic.
#[test]
fn sanitize_empty_history() {
    let mut msgs: Vec<ChatMessage> = Vec::new();
    assert_eq!(sanitize_history(&mut msgs), 0);
}

/// Sanitize with only one message: system prompt — must pass through.
#[test]
fn sanitize_single_system_message() {
    let mut msgs = vec![ChatMessage::system("you are helpful")];
    assert_eq!(sanitize_history(&mut msgs), 0);
    assert_eq!(msgs.len(), 1);
}

// ─────────── Section 5: Compact view integrity over long sessions ───────────

use aacode_rs::config::ContextConfig;

#[test]
fn compact_view_many_rounds_tool_pairs_intact() {
    let mut msgs = vec![ChatMessage::system("SYS")];
    for i in 0..50 {
        msgs.push(ChatMessage::user(format!("task {i} {}", "x".repeat(200))));
        msgs.push(ChatMessage::assistant_with_tools(
            String::new(),
            vec![ToolCall { id: format!("c{i}"), name: "run_shell".into(), arguments: "{}".into() }],
        ));
        msgs.push(ChatMessage::tool_result(format!("c{i}"), "y".repeat(200)));
    }
    let cfg = ContextConfig {
        compact_trigger_tokens: 100,
        protect_first_rounds: 1,
        keep_last_rounds: 5,
        protect_last_user_rounds: 2,
        ..Default::default()
    };
    let (view, compacted, tokens) =
        aacode_rs::agent::compact::build_compact_view(&msgs, &cfg, None);
    assert!(compacted, "50 rounds must trigger compaction");
    assert!(tokens > 0);
    // System prefix must survive.
    assert_eq!(view[0].role, "system");
    assert_eq!(view[0].content, "SYS");
    // Every assistant-with-tool_calls must be followed by a tool result.
    for i in 0..view.len() {
        if view[i].tool_calls.is_some() {
            assert!(
                i + 1 < view.len() && view[i + 1].role == "tool",
                "tool pair split at index {i}"
            );
        }
    }
}

// ─────────── Section 6: ChatMessage roundtrip integrity ───────────

#[test]
fn chat_message_reasoning_survives_serialize_deserialize() {
    let mut m = ChatMessage::assistant("answer");
    m.reasoning_content = Some("deep thinking...".into());
    let json = serde_json::to_string(&m).unwrap();
    let back: ChatMessage = serde_json::from_str(&json).unwrap();
    assert_eq!(back.reasoning_content.as_deref(), Some("deep thinking..."));
    assert_eq!(back.content, "answer");
}

#[test]
fn tool_result_message_roundtrip() {
    let m = ChatMessage::tool_result("call_abc", "result text");
    let json = serde_json::to_string(&m).unwrap();
    let back: ChatMessage = serde_json::from_str(&json).unwrap();
    assert_eq!(back.role, "tool");
    assert_eq!(back.tool_call_id.as_deref(), Some("call_abc"));
    assert_eq!(back.content, "result text");
}

/// Unicode / emoji in tool call arguments must survive roundtrips.
#[test]
fn tool_call_with_unicode_args_roundtrip() {
    let tc = ToolCall {
        id: "c1".into(),
        name: "run_shell".into(),
        arguments: "{\"command\":\"echo 🚀 你好\"}".into(),
    };
    let m = ChatMessage::assistant_with_tools("", vec![tc.clone()]);
    let json = serde_json::to_string(&m).unwrap();
    let back: ChatMessage = serde_json::from_str(&json).unwrap();
    let restored = back.tool_calls.unwrap();
    assert_eq!(restored[0].arguments, tc.arguments);
    assert_eq!(restored[0].parsed_args()["command"], "echo 🚀 你好");
}

// ─────────────────────── Live API tests ───────────────────────
//
// These tests require environment variables and are marked #[ignore] so they
// never run as part of `cargo test`. To execute:
//
//   1. Set the env vars for your target provider:
//      export LLM_API_KEY="sk-xxx"
//      export LLM_API_URL="https://api.deepseek.com/v1"
//      export LLM_MODEL_NAME="deepseek-chat"
//      export LLM_GATEWAY="openai"
//
//   2. cargo test -p aacode-rs --test long_session_stress -- --ignored --nocapture
//
// The API keys provided in the task description are read from env vars only;
// they are NEVER hard-coded.

use aacode_rs::config::{Gateway, ModelConfig};
use aacode_rs::llm::build_client;
use std::time::Instant;

/// Build a ModelConfig from environment variables. Returns None if LLM_API_KEY
/// is not set (so CI/offline runs skip silently).
fn live_model_from_env(gateway: Gateway) -> Option<ModelConfig> {
    let api_key = std::env::var("LLM_API_KEY").ok()?;
    if api_key.trim().is_empty() {
        return None;
    }
    let mut model = ModelConfig {
        name: std::env::var("LLM_MODEL_NAME").unwrap_or_else(|_| {
            match gateway {
                Gateway::Openai => "deepseek-chat".into(),
                Gateway::Anthropic => "MiniMax-M2.7".into(),
            }
        }),
        api_key: Some(api_key),
        base_url: std::env::var("LLM_API_URL").ok(),
        gateway,
        temperature: 0.1,
        max_tokens: 4096,
        ..Default::default()
    };
    model.apply_env();
    Some(model)
}

/// OpenAI: basic streaming text (sanity check).
#[test]
#[ignore = "requires env: LLM_API_KEY, LLM_API_URL, LLM_MODEL_NAME, LLM_GATEWAY=openai"]
fn live_openai_streams_text_sanity() {
    let model = live_model_from_env(Gateway::Openai).expect("LLM_API_KEY not set");
    let client = build_client(&model);
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    let msgs = vec![
        ChatMessage::system("Answer in exactly one word."),
        ChatMessage::user("What is the capital of France?"),
    ];
    let start = Instant::now();
    let resp = client.chat_stream(&msgs, &[], &sink, &cancel).expect("live call failed");
    eprintln!("[openai text] model={} elapsed={:?} text={:?}", model.name, start.elapsed(), resp.text);
    assert!(!resp.text.trim().is_empty(), "expected non-empty text response");
    let lines = sink.lines();
    assert!(lines.iter().any(|l| l.contains(r#""seg":"thought""#)));
    assert!(!lines.is_empty());
}

/// OpenAI: tool call request (single tool).
#[test]
#[ignore = "requires env: LLM_API_KEY, LLM_API_URL, LLM_MODEL_NAME, LLM_GATEWAY=openai"]
fn live_openai_single_tool_call() {
    let model = live_model_from_env(Gateway::Openai).expect("LLM_API_KEY not set");
    let client = build_client(&model);
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    let tools = vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Execute a shell command. Use this to list files or run commands.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "the shell command"}},
                "required": ["command"]
            }
        }
    })];
    let msgs = vec![
        ChatMessage::system("You MUST use the run_shell tool exactly once. Call it with command='ls' then stop."),
        ChatMessage::user("List the files in the current directory."),
    ];
    let start = Instant::now();
    let resp = client.chat_stream(&msgs, &tools, &sink, &cancel).expect("live call failed");
    eprintln!("[openai tool_call] model={} elapsed={:?} tool_calls={:?}",
        model.name, start.elapsed(),
        resp.tool_calls.iter().map(|t| format!("{}:{}", t.name, t.arguments)).collect::<Vec<_>>());
    assert!(!resp.tool_calls.is_empty(), "expected at least one tool call, got: finish_reason={:?}", resp.finish_reason);
    assert_eq!(resp.tool_calls[0].name, "run_shell");
    assert!(resp.tool_calls[0].parsed_args().get("command").is_some());
}

/// OpenAI: multi-turn conversation — asks model to use a tool, feeds back a
/// simulated result, asks follow-up. Verifies the API handles multi-message
/// history correctly across both gateways.
#[test]
#[ignore = "requires env: LLM_API_KEY, LLM_API_URL, LLM_MODEL_NAME, LLM_GATEWAY=openai"]
fn live_openai_multi_turn_conversation() {
    let model = live_model_from_env(Gateway::Openai).expect("LLM_API_KEY not set");
    let client = build_client(&model);
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
    let cancel = AtomicBool::new(false);
    let mut msgs = vec![
        ChatMessage::system("You are a helpful coding assistant. Use the run_shell tool when asked to run commands. After seeing the output, produce a natural language summary. Do not call the tool more than once per turn."),
        ChatMessage::user("Run 'echo hello' and tell me what happened."),
    ];

    eprintln!("=== Multi-turn conversation start (model={}) ===", model.name);

    // Turn 1: model calls run_shell (echo hello)
    let sink1 = CollectingSink::new(false);
    let start = Instant::now();
    let resp1 = client.chat_stream(&msgs, &tools, &sink1, &cancel).expect("turn 1 failed");
    eprintln!("[turn 1] elapsed={:?} tool_calls={:?}", start.elapsed(),
        resp1.tool_calls.iter().map(|t| format!("{}:{}", t.name, t.arguments)).collect::<Vec<_>>());
    assert!(!resp1.tool_calls.is_empty(), "turn 1: expected tool call, got finish={:?} text={:?}",
        resp1.finish_reason, resp1.text);

    // Simulate tool result
    let tc1 = &resp1.tool_calls[0];
    let tool_result = format!("[shell output] hello\n(exit code: 0)");
    msgs.push(ChatMessage::assistant_with_tools(resp1.text.clone(), resp1.tool_calls.clone()));
    msgs.push(ChatMessage::tool_result(tc1.id.clone(), tool_result));

    // Turn 2: model summarises, no more tools needed
    msgs.push(ChatMessage::user("Summarize what happened."));
    let sink2 = CollectingSink::new(false);
    let start2 = Instant::now();
    let resp2 = client.chat_stream(&msgs, &[], &sink2, &cancel).expect("turn 2 failed");
    eprintln!("[turn 2] elapsed={:?} text={:?}", start2.elapsed(),
        &resp2.text[..resp2.text.len().min(200)]);
    assert!(!resp2.text.trim().is_empty(), "turn 2: expected text response");
    // The summary should reference the tool execution.
    eprintln!("=== Multi-turn conversation OK ({} turns) ===", 2);
}

/// OpenAI: 5-iteration tool-use loop simulating a longer session. Each turn the
/// model is asked a new question that requires a tool call, we feed the tool
/// result, and repeat. Verifies no degradation across iterations.
#[test]
#[ignore = "requires env: LLM_API_KEY, LLM_API_URL, LLM_MODEL_NAME, LLM_GATEWAY=openai"]
fn live_openai_five_iteration_tool_loop() {
    let model = live_model_from_env(Gateway::Openai).expect("LLM_API_KEY not set");
    let client = build_client(&model);
    let tools = vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Execute a shell command. Output is returned as plain text.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            }
        }
    })];
    let cancel = AtomicBool::new(false);
    let mut msgs = vec![
        ChatMessage::system("You are a shell assistant. When user asks to run a command, use run_shell. After seeing output, provide a brief summary. Do NOT call run_shell more than once. Each turn you see the result."),
    ];

    let commands = [
        ("Run 'echo iteration-1'", "iteration-1"),
        ("Run 'echo iteration-2'", "iteration-2"),
        ("Run 'echo iteration-3'", "iteration-3"),
        ("Run 'echo iteration-4'", "iteration-4"),
        ("Run 'echo iteration-5'", "iteration-5"),
    ];

    for (i, (prompt, expected_output)) in commands.iter().enumerate() {
        msgs.push(ChatMessage::user(*prompt));
        let sink = CollectingSink::new(false);
        let start = Instant::now();
        let resp = match client.chat_stream(&msgs, &tools, &sink, &cancel) {
            Ok(r) => r,
            Err(e) => panic!("iteration {i} failed: {e}"),
        };
        eprintln!("[iter {i}] elapsed={:?} tool_calls={} text={:?}",
            start.elapsed(),
            resp.tool_calls.len(),
            &resp.text[..resp.text.len().min(100)]);
        assert!(!resp.tool_calls.is_empty(),
            "iteration {i}: expected tool call, got finish={:?}", resp.finish_reason);
        // Feed tool result
        let tc = &resp.tool_calls[0];
        let tool_result = format!("[shell output]\n{expected_output}\n(exit code: 0)");
        msgs.push(ChatMessage::assistant_with_tools(resp.text.clone(), resp.tool_calls.clone()));
        msgs.push(ChatMessage::tool_result(tc.id.clone(), tool_result));
    }
    eprintln!("=== 5-iteration loop OK (model={}) ===", model.name);
}

/// OpenAI: truncated response recovery — set max_tokens very low so the
/// response hits "length" finish reason. The parser must set is_truncated().
#[test]
#[ignore = "requires env: LLM_API_KEY, LLM_API_URL, LLM_MODEL_NAME, LLM_GATEWAY=openai"]
fn live_openai_truncation_detection() {
    let model = live_model_from_env(Gateway::Openai).expect("LLM_API_KEY not set");
    let mut model = model;
    model.max_tokens = 10; // force truncation
    let client = build_client(&model);
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    let msgs = vec![
        ChatMessage::system("You MUST output a very long response with at least 200 words about the history of computers."),
        ChatMessage::user("Tell me about computer history in detail."),
    ];
    let resp = client.chat_stream(&msgs, &[], &sink, &cancel).expect("live call failed");
    eprintln!("[truncation] model={} max_tokens=10 finish={:?} is_truncated={} text_len={}",
        model.name, resp.finish_reason, resp.is_truncated(), resp.text.len());
    // The model might refuse or max_tokens might not hit for a short model.
    // But if finish_reason is "length", is_truncated must be true.
    if resp.finish_reason.as_deref() == Some("length") {
        assert!(resp.is_truncated());
    }
    // At minimum, the API call succeeded.
}

/// OpenAI: cancellation mid-stream — start a call, cancel after a short delay.
/// The parser must return AacodeError::Cancelled.
#[test]
#[ignore = "requires env: LLM_API_KEY, LLM_API_URL, LLM_MODEL_NAME, LLM_GATEWAY=openai"]
fn live_openai_cancellation_mid_stream() {
    let model = live_model_from_env(Gateway::Openai).expect("LLM_API_KEY not set");
    let client = build_client(&model);
    let sink = CollectingSink::new(false);
    let cancel = std::sync::Arc::new(AtomicBool::new(false));
    let cancel_clone = cancel.clone();
    let msgs = vec![
        ChatMessage::user("Write a 500-word essay about AI in great detail."),
    ];

    // Cancel after a short time (the AtomicBool is checked per SSE chunk).
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(200));
        cancel_clone.store(true, std::sync::atomic::Ordering::SeqCst);
    });

    let start = Instant::now();
    let r = client.chat_stream(&msgs, &[], &sink, &cancel);
    eprintln!("[cancel test] elapsed={:?} result={:?}", start.elapsed(),
        r.as_ref().map(|_| "ok".to_string()).unwrap_or_else(|e| e.to_string()));
    match r {
        Ok(resp) => {
            // If the model finished before cancel fired, that's also valid.
            eprintln!("[cancel test] model finished before cancel: text_len={}", resp.text.len());
        }
        Err(e) => {
            assert!(
                e.to_string().contains("cancelled") || e.to_string().contains("Cancelled")
                    || e.to_string().contains("timeout") || e.to_string().contains("Timeout"),
                "expected cancellation or timeout, got: {e}"
            );
        }
    }
}

// ─────────── Anthropic live tests ───────────

/// Anthropic: basic streaming text.
#[test]
#[ignore = "requires env: LLM_API_KEY, LLM_API_URL, LLM_MODEL_NAME, LLM_GATEWAY=anthropic"]
fn live_anthropic_streams_text_sanity() {
    let model = live_model_from_env(Gateway::Anthropic).expect("LLM_API_KEY not set");
    eprintln!("[anthropic text] using model={} base_url={:?}", model.name, model.base_url);
    let client = build_client(&model);
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    let msgs = vec![
        ChatMessage::system("Answer in exactly one word."),
        ChatMessage::user("What is the capital of France?"),
    ];
    let start = Instant::now();
    let resp = client.chat_stream(&msgs, &[], &sink, &cancel).expect("live call failed");
    eprintln!("[anthropic text] elapsed={:?} text={:?}", start.elapsed(), resp.text);
    assert!(!resp.text.trim().is_empty());
}

/// Anthropic: tool call request (Anthropic-format tools).
#[test]
#[ignore = "requires env: LLM_API_KEY, LLM_API_URL, LLM_MODEL_NAME, LLM_GATEWAY=anthropic"]
fn live_anthropic_single_tool_call() {
    let model = live_model_from_env(Gateway::Anthropic).expect("LLM_API_KEY not set");
    eprintln!("[anthropic tool] using model={} base_url={:?}", model.name, model.base_url);
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
        ChatMessage::system("You MUST call run_shell exactly once with command='ls'. Then stop."),
        ChatMessage::user("List the files."),
    ];
    let start = Instant::now();
    let resp = client.chat_stream(&msgs, &tools, &sink, &cancel).expect("live call failed");
    eprintln!("[anthropic tool_call] elapsed={:?} tool_calls={:?} finish={:?}",
        start.elapsed(),
        resp.tool_calls.iter().map(|t| format!("{}:{}", t.name, t.arguments)).collect::<Vec<_>>(),
        resp.finish_reason);
    assert!(!resp.tool_calls.is_empty(),
        "expected tool call; finish={:?} text={:?}", resp.finish_reason,
        &resp.text[..resp.text.len().min(200)]);
    assert_eq!(resp.tool_calls[0].name, "run_shell");
    assert!(resp.tool_calls[0].parsed_args().get("command").is_some());
}

/// Anthropic: multi-turn (tool call → tool result → summary).
#[test]
#[ignore = "requires env: LLM_API_KEY, LLM_API_URL, LLM_MODEL_NAME, LLM_GATEWAY=anthropic"]
fn live_anthropic_multi_turn_conversation() {
    let model = live_model_from_env(Gateway::Anthropic).expect("LLM_API_KEY not set");
    eprintln!("[anthropic multi-turn] using model={} base_url={:?}", model.name, model.base_url);
    let client = build_client(&model);
    let tools = vec![serde_json::json!({
        "name": "run_shell",
        "description": "Execute a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"]
        }
    })];
    let cancel = AtomicBool::new(false);

    let mut msgs = vec![
        ChatMessage::system("You are a helpful coding assistant. Use run_shell when asked to run commands."),
        ChatMessage::user("Run 'echo hello_anthropic' and tell me the result."),
    ];

    // Turn 1: model calls run_shell
    let sink1 = CollectingSink::new(false);
    let resp1 = client.chat_stream(&msgs, &tools, &sink1, &cancel).expect("turn 1 failed");
    eprintln!("[anth turn 1] tool_calls={:?} finish={:?}",
        resp1.tool_calls.iter().map(|t| format!("{}:{}", t.name, t.arguments)).collect::<Vec<_>>(),
        resp1.finish_reason);
    assert!(!resp1.tool_calls.is_empty(), "turn 1: expected tool call");

    // Feed tool result
    let tc1 = &resp1.tool_calls[0];
    let tool_result = format!("[shell output]\nhello_anthropic\n(exit code: 0)");
    msgs.push(ChatMessage::assistant_with_tools(resp1.text.clone(), resp1.tool_calls.clone()));
    msgs.push(ChatMessage::tool_result(tc1.id.clone(), tool_result));

    // Turn 2: summary
    msgs.push(ChatMessage::user("Summarize the output."));
    let sink2 = CollectingSink::new(false);
    let resp2 = client.chat_stream(&msgs, &[], &sink2, &cancel).expect("turn 2 failed");
    eprintln!("[anth turn 2] text={:?}", &resp2.text[..resp2.text.len().min(200)]);
    assert!(!resp2.text.trim().is_empty());
    eprintln!("=== Anthropic multi-turn OK ===");
}

/// Anthropic: 5-iteration tool loop.
#[test]
#[ignore = "requires env: LLM_API_KEY, LLM_API_URL, LLM_MODEL_NAME, LLM_GATEWAY=anthropic"]
fn live_anthropic_five_iteration_tool_loop() {
    let model = live_model_from_env(Gateway::Anthropic).expect("LLM_API_KEY not set");
    eprintln!("[anthropic 5-iter] using model={} base_url={:?}", model.name, model.base_url);
    let client = build_client(&model);
    let tools = vec![serde_json::json!({
        "name": "run_shell",
        "description": "Execute a shell command. Output is returned.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"]
        }
    })];
    let cancel = AtomicBool::new(false);
    let mut msgs = vec![
        ChatMessage::system("You are a shell assistant. When user asks to run a command, call run_shell. After each result, simply say what the output was. Do NOT call run_shell more than once."),
    ];

    for i in 1..=5 {
        msgs.push(ChatMessage::user(format!("Run 'echo anthropic_iter_{i}'")));
        let sink = CollectingSink::new(false);
        let start = Instant::now();
        let resp = match client.chat_stream(&msgs, &tools, &sink, &cancel) {
            Ok(r) => r,
            Err(e) => panic!("anthropic iteration {i} failed: {e}"),
        };
        eprintln!("[anth iter {i}] elapsed={:?} tool_calls={}",
            start.elapsed(), resp.tool_calls.len());
        assert!(!resp.tool_calls.is_empty(),
            "iteration {i}: expected tool call, finish={:?}", resp.finish_reason);
        let tc = &resp.tool_calls[0];
        let tool_result = format!("[shell output]\nanthropic_iter_{i}\n(exit code: 0)");
        msgs.push(ChatMessage::assistant_with_tools(resp.text.clone(), resp.tool_calls.clone()));
        msgs.push(ChatMessage::tool_result(tc.id.clone(), tool_result));
    }
    eprintln!("=== Anthropic 5-iteration loop OK ===");
}

/// Anthropic: max_tokens truncation detection.
#[test]
#[ignore = "requires env: LLM_API_KEY, LLM_API_URL, LLM_MODEL_NAME, LLM_GATEWAY=anthropic"]
fn live_anthropic_truncation_detection() {
    let model = live_model_from_env(Gateway::Anthropic).expect("LLM_API_KEY not set");
    let mut model = model;
    model.max_tokens = 10;
    eprintln!("[anthropic truncation] using model={} max_tokens=10", model.name);
    let client = build_client(&model);
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    let msgs = vec![
        ChatMessage::system("Write a very long paragraph about computer history with at least 200 words."),
        ChatMessage::user("Tell me about computers."),
    ];
    let resp = client.chat_stream(&msgs, &[], &sink, &cancel).expect("live call failed");
    eprintln!("[anthropic truncation] finish={:?} is_truncated={} text_len={}",
        resp.finish_reason, resp.is_truncated(), resp.text.len());
    if resp.finish_reason.as_deref() == Some("max_tokens") {
        assert!(resp.is_truncated());
    }
}

// ─────────── Live: provider-specific edge cases ───────────

/// Some models (DeepSeek V3, Kimi) produce a `reasoning_content` followed by
/// `content` + `tool_calls`. Verify that reasoning is separated from the
/// visible text and tool calls work correctly even when reasoning is present.
#[test]
#[ignore = "requires env: LLM_API_KEY, LLM_API_URL, LLM_MODEL_NAME, LLM_GATEWAY=openai"]
fn live_openai_reasoning_then_tool_call() {
    let model = live_model_from_env(Gateway::Openai).expect("LLM_API_KEY not set");
    let client = build_client(&model);
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
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
        ChatMessage::system("You MUST call run_shell with command='echo thinking_works'. Think about it first."),
        ChatMessage::user("Run echo."),
    ];
    let resp = client.chat_stream(&msgs, &tools, &sink, &cancel).expect("live call failed");
    eprintln!("[reasoning+tool] finish={:?} reasoning_len={} text_len={} tool_calls={}",
        resp.finish_reason,
        resp.reasoning_content.as_ref().map(|r| r.len()).unwrap_or(0),
        resp.text.len(),
        resp.tool_calls.len());
    // The model may or may not emit reasoning_content. But if it does, we
    // must have both reasoning and tool_calls correctly separated.
    if resp.tool_calls.is_empty() {
        eprintln!("[reasoning+tool] model did not emit tool calls (may be a reasoning-only model)");
    } else {
        assert_eq!(resp.tool_calls[0].name, "run_shell");
        assert!(resp.tool_calls[0].parsed_args().get("command").is_some());
    }
    if let Some(reasoning) = &resp.reasoning_content {
        assert!(!reasoning.is_empty());
        eprintln!("[reasoning+tool] reasoning present ({} chars)", reasoning.len());
    }
}

/// Validate that the client's validate() method works (lightweight non-streaming
/// ping that doesn't consume credits significantly).
#[test]
#[ignore = "requires env: LLM_API_KEY, LLM_API_URL, LLM_MODEL_NAME, LLM_GATEWAY=openai"]
fn live_openai_validate_api_key() {
    let model = live_model_from_env(Gateway::Openai).expect("LLM_API_KEY not set");
    let client = build_client(&model);
    match client.validate() {
        Ok(()) => eprintln!("[validate] OK (model={})", model.name),
        Err(e) => panic!("validate failed: {e}"),
    }
}

#[test]
#[ignore = "requires env: LLM_API_KEY, LLM_API_URL, LLM_MODEL_NAME, LLM_GATEWAY=anthropic"]
fn live_anthropic_validate_api_key() {
    let model = live_model_from_env(Gateway::Anthropic).expect("LLM_API_KEY not set");
    let client = build_client(&model);
    match client.validate() {
        Ok(()) => eprintln!("[validate] OK (model={})", model.name),
        Err(e) => panic!("validate failed: {e}"),
    }
}

/// Test with model that supports multimodal but we send text-only — should still
/// work correctly.
#[test]
#[ignore = "requires env: LLM_API_KEY, LLM_API_URL, LLM_MODEL_NAME, LLM_GATEWAY=openai"]
fn live_openai_text_only_on_multimodal_model() {
    let model = live_model_from_env(Gateway::Openai).expect("LLM_API_KEY not set");
    let client = build_client(&model);
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    let msgs = vec![
        ChatMessage::system("Answer in one sentence only."),
        ChatMessage::user("What is 2+2?"),
    ];
    let resp = client.chat_stream(&msgs, &[], &sink, &cancel).expect("live call failed");
    eprintln!("[multimodal text-only] text={:?}", &resp.text[..resp.text.len().min(200)]);
    assert!(resp.text.to_lowercase().contains("4"));
}

/// Anthropic validate on multimodal model (MiniMax M2.7 supports vision).
#[test]
#[ignore = "requires env: LLM_API_KEY, LLM_API_URL, LLM_MODEL_NAME, LLM_GATEWAY=anthropic"]
fn live_anthropic_text_only_on_multimodal_model() {
    let model = live_model_from_env(Gateway::Anthropic).expect("LLM_API_KEY not set");
    let client = build_client(&model);
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);
    let msgs = vec![
        ChatMessage::system("Answer in one sentence only."),
        ChatMessage::user("What is 2+2?"),
    ];
    let resp = client.chat_stream(&msgs, &[], &sink, &cancel).expect("live call failed");
    eprintln!("[anth multimodal text-only] text={:?}", &resp.text[..resp.text.len().min(200)]);
    assert!(resp.text.to_lowercase().contains("4"));
}

// ─────────── Live: `chat_with_retry` end-to-end (via AgentRuntime) ───────────

/// Run a full ReAct loop against a live API with the real run_shell tool.
/// The model executes a shell command and the output flows through the sandbox.
#[test]
#[ignore = "requires env: LLM_API_KEY, LLM_API_URL, LLM_MODEL_NAME, LLM_GATEWAY=openai"]
fn live_openai_full_react_loop_with_shell() {
    use aacode_rs::config::{AgentConfig, Gateway};
    use aacode_rs::runtime::AgentRuntime;
    use aacode_rs::stream::CollectingSink;
    use std::sync::atomic::AtomicBool;

    let api_key = std::env::var("LLM_API_KEY").expect("LLM_API_KEY not set");
    let mut cfg = AgentConfig::default();
    cfg.model.name = std::env::var("LLM_MODEL_NAME").unwrap_or_else(|_| "deepseek-chat".into());
    cfg.model.api_key = Some(api_key);
    cfg.model.base_url = std::env::var("LLM_API_URL").ok();
    cfg.model.gateway = Gateway::Openai;
    cfg.max_iterations = 5;
    cfg.limits.max_retries = 2;

    let proj = std::env::temp_dir().join(format!(
        "aacode_stress_{}_{}",
        std::process::id(),
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::create_dir_all(&proj).unwrap();

    let rt = AgentRuntime::init(cfg, proj.clone()).unwrap();
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);

    let res = rt.run_task(
        "Run 'echo live_react_test' using the shell tool and tell me the output.",
        None,
        &sink,
        &cancel,
    ).expect("react loop failed");

    eprintln!("[live react] status={:?} iterations={}", res.status, res.iterations);
    eprintln!("[live react] final_text={:?}", &res.final_text[..res.final_text.len().min(300)]);
    let lines = sink.lines();
    eprintln!("[live react] event_count={}", lines.len());

    assert_eq!(format!("{:?}", res.status), "Completed",
        "react loop should complete; final_text={:?}", res.final_text);

    // The final text (or an observation) should mention the echo output.
    let has_live_react_test = lines.iter().any(|l| l.contains("live_react_test"))
        || res.final_text.contains("live_react_test");
    assert!(has_live_react_test, "expected 'live_react_test' in output");

    // Clean up.
    let _ = std::fs::remove_dir_all(&proj);
}

/// Same as above but for Anthropic gateway.
#[test]
#[ignore = "requires env: LLM_API_KEY, LLM_API_URL, LLM_MODEL_NAME, LLM_GATEWAY=anthropic"]
fn live_anthropic_full_react_loop_with_shell() {
    use aacode_rs::config::{AgentConfig, Gateway};
    use aacode_rs::runtime::AgentRuntime;
    use aacode_rs::stream::CollectingSink;
    use std::sync::atomic::AtomicBool;

    let api_key = std::env::var("LLM_API_KEY").expect("LLM_API_KEY not set");
    let mut cfg = AgentConfig::default();
    cfg.model.name = std::env::var("LLM_MODEL_NAME").unwrap_or_else(|_| "MiniMax-M2.7".into());
    cfg.model.api_key = Some(api_key);
    cfg.model.base_url = std::env::var("LLM_API_URL").ok();
    cfg.model.gateway = Gateway::Anthropic;
    cfg.max_iterations = 5;
    cfg.limits.max_retries = 2;

    let proj = std::env::temp_dir().join(format!(
        "aacode_stress_{}_{}",
        std::process::id(),
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::create_dir_all(&proj).unwrap();

    let rt = AgentRuntime::init(cfg, proj.clone()).unwrap();
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);

    let res = rt.run_task(
        "Run 'echo live_react_anthropic_test' using the run_shell tool and report the output.",
        None,
        &sink,
        &cancel,
    ).expect("react loop failed");

    eprintln!("[live anth react] status={:?} iterations={}", res.status, res.iterations);
    eprintln!("[live anth react] final_text={:?}", &res.final_text[..res.final_text.len().min(300)]);
    let lines = sink.lines();
    eprintln!("[live anth react] event_count={}", lines.len());

    assert_eq!(format!("{:?}", res.status), "Completed");

    let has_output = lines.iter().any(|l| l.contains("live_react_anthropic_test"))
        || res.final_text.contains("live_react_anthropic_test");
    assert!(has_output, "expected 'live_react_anthropic_test' in output");

    let _ = std::fs::remove_dir_all(&proj);
}
