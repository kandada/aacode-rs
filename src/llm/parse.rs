// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Shared SSE parsing logic used by both sync and async LLM clients.
//!
//! Extracts the pure state-transition logic from the I/O loop so that
//! `openai.rs`, `anthropic.rs`, and `async_llm.rs` share a single
//! implementation of the OpenAI/Anthropic wire-format parsing.

use super::types::{LlmResponse, ToolCall};
use crate::error::{AacodeError, Result};
use crate::stream::EventSink;
use serde_json::Value;
use std::collections::BTreeMap;

// ── OpenAI ────────────────────────────────────────────────────────────────

/// Accumulator for one streamed tool_call (fragments arrive by index).
#[derive(Default, Clone)]
pub struct OpenAiToolAcc {
    pub id: String,
    pub name: String,
    pub arguments: String,
    pub name_announced: bool,
    pub last_report: usize,
}

pub struct OpenAiParseState {
    pub text: String,
    pub reasoning: String,
    pub tool_accs: BTreeMap<i64, OpenAiToolAcc>,
    pub finish_reason: Option<String>,
    pub valid_chunks: usize,
    pub total_payloads: usize,
}

impl Default for OpenAiParseState {
    fn default() -> Self {
        OpenAiParseState {
            text: String::new(),
            reasoning: String::new(),
            tool_accs: BTreeMap::new(),
            finish_reason: None,
            valid_chunks: 0,
            total_payloads: 0,
        }
    }
}

/// Parse one OpenAI SSE `data:` payload into the accumulator state.
/// Returns `Ok(())` on success. The caller handles the I/O loop and
/// calls `finalize_openai` afterwards.
pub fn parse_openai_chunk(
    payload: &str,
    state: &mut OpenAiParseState,
    emitter: &dyn EventSink,
) -> Result<()> {
    state.total_payloads += 1;
    let chunk: Value = match serde_json::from_str(payload) {
        Ok(v) => {
            state.valid_chunks += 1;
            v
        }
        Err(_) => return Ok(()),
    };
    if let Some(err) = chunk.get("error") {
        let msg = err
            .get("message")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| err.to_string());
        return Err(AacodeError::Api(format!("stream error: {msg}")));
    }
    let choice = match chunk.get("choices").and_then(|c| c.get(0)) {
        Some(c) => c,
        None => return Ok(()),
    };
    if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
        state.finish_reason = Some(fr.to_string());
    }
    let delta = match choice.get("delta") {
        Some(d) => d,
        None => return Ok(()),
    };

    let rc_opt = delta
        .get("reasoning_content")
        .and_then(|v| v.as_str())
        .or_else(|| delta.get("reasoning").and_then(|v| v.as_str()));
    if let Some(rc) = rc_opt {
        if !rc.is_empty() {
            state.reasoning.push_str(rc);
            emitter.delta("thinking", rc);
        }
    }
    if let Some(c) = delta.get("content").and_then(|v| v.as_str()) {
        if !c.is_empty() {
            state.text.push_str(c);
            emitter.delta("thought", c);
        }
    }
    if let Some(tcs) = delta.get("tool_calls").and_then(|v| v.as_array()) {
        for tc in tcs {
            let idx = tc.get("index").and_then(|v| v.as_i64()).unwrap_or(0);
            let acc = state.tool_accs.entry(idx).or_default();
            if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                if !id.is_empty() {
                    acc.id = id.to_string();
                }
            }
            if let Some(func) = tc.get("function") {
                if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                    if !name.is_empty() {
                        acc.name.push_str(name);
                        if !acc.name_announced {
                            acc.name_announced = true;
                            emitter.tool_progress("building", &acc.name, 0);
                        }
                    }
                }
                if let Some(args) = func.get("arguments").and_then(|v| v.as_str()) {
                    acc.arguments.push_str(args);
                    let cur = acc.arguments.len();
                    if cur - acc.last_report >= 500 {
                        acc.last_report = cur - (cur % 500);
                        emitter.tool_progress("building", &acc.name, cur);
                    }
                }
            }
        }
    }
    Ok(())
}

/// Finalize: check for all-malformed, emit end segments, assemble tool_calls.
pub fn finalize_openai(
    state: OpenAiParseState,
    emitter: &dyn EventSink,
) -> Result<LlmResponse> {
    if state.total_payloads > 0 && state.valid_chunks == 0 {
        return Err(AacodeError::Api(
            "stream returned no parseable data (all chunks malformed)".into(),
        ));
    }

    let mut text = state.text;
    let reasoning = state.reasoning;

    if !reasoning.is_empty() {
        emitter.seg_large("thinking", &reasoning, 512);
    }
    emitter.seg_large("thought", &text, 512);

    let mut tool_calls = Vec::new();
    for (i, (_, acc)) in state.tool_accs.into_iter().enumerate() {
        if acc.name.is_empty() {
            continue;
        }
        let id = if acc.id.is_empty() {
            format!("call_{i}")
        } else {
            acc.id
        };
        emitter.tool_progress("done", &acc.name, acc.arguments.len());
        emitter.action(&acc.name, &acc.arguments);
        tool_calls.push(ToolCall {
            id,
            name: acc.name,
            arguments: acc.arguments,
        });
    }

    if matches!(state.finish_reason.as_deref(), Some("length")) {
        text.push_str(
            "\n\n[⚠️ WARNING: API response truncated (max_tokens). Reduce content or raise max_tokens.]",
        );
    }

    Ok(LlmResponse {
        text,
        tool_calls,
        reasoning_content: if reasoning.is_empty() {
            None
        } else {
            Some(reasoning)
        },
        finish_reason: state.finish_reason,
    })
}

// ── Anthropic ─────────────────────────────────────────────────────────────

#[derive(Default, Clone)]
pub struct AnthropicBlockAcc {
    pub kind: String,         // "text" | "thinking" | "tool_use"
    pub id: String,
    pub name: String,
    pub text: String,
    pub partial_json: String,
    pub name_announced: bool,
    pub last_report: usize,
}

pub struct AnthropicParseState {
    pub blocks: BTreeMap<i64, AnthropicBlockAcc>,
    pub text: String,
    pub reasoning: String,
    pub stop_reason: Option<String>,
    pub valid_chunks: usize,
    pub total_payloads: usize,
}

impl Default for AnthropicParseState {
    fn default() -> Self {
        AnthropicParseState {
            blocks: BTreeMap::new(),
            text: String::new(),
            reasoning: String::new(),
            stop_reason: None,
            valid_chunks: 0,
            total_payloads: 0,
        }
    }
}

pub fn parse_anthropic_chunk(
    payload: &str,
    state: &mut AnthropicParseState,
    emitter: &dyn EventSink,
) -> Result<()> {
    state.total_payloads += 1;
    let ev: Value = match serde_json::from_str(payload) {
        Ok(v) => {
            state.valid_chunks += 1;
            v
        }
        Err(_) => return Ok(()),
    };
    let etype = ev.get("type").and_then(|v| v.as_str()).unwrap_or("");
    match etype {
        "error" => {
            let msg = ev
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| ev.to_string());
            return Err(AacodeError::Api(format!("stream error: {msg}")));
        }
        "content_block_start" => {
            let idx = ev.get("index").and_then(|v| v.as_i64()).unwrap_or(0);
            let block = ev.get("content_block");
            let mut acc = AnthropicBlockAcc::default();
            if let Some(b) = block {
                acc.kind = b
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                acc.id = b
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                acc.name = b
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
            }
            if acc.kind == "tool_use" && !acc.name.is_empty() && !acc.name_announced {
                acc.name_announced = true;
                emitter.tool_progress("building", &acc.name, 0);
            }
            state.blocks.insert(idx, acc);
        }
        "content_block_delta" => {
            let idx = ev.get("index").and_then(|v| v.as_i64()).unwrap_or(0);
            let delta = ev.get("delta");
            if let Some(d) = delta {
                let dt = d.get("type").and_then(|v| v.as_str()).unwrap_or("");
                let acc = state.blocks.entry(idx).or_default();
                match dt {
                    "text_delta" => {
                        if let Some(t) = d.get("text").and_then(|v| v.as_str()) {
                            state.text.push_str(t);
                            acc.text.push_str(t);
                            emitter.delta("thought", t);
                        }
                    }
                    "thinking_delta" => {
                        if let Some(t) = d.get("thinking").and_then(|v| v.as_str()) {
                            state.reasoning.push_str(t);
                            acc.text.push_str(t);
                            emitter.delta("thinking", t);
                        }
                    }
                    "input_json_delta" => {
                        if let Some(pj) = d.get("partial_json").and_then(|v| v.as_str()) {
                            acc.partial_json.push_str(pj);
                            let cur = acc.partial_json.len();
                            if cur - acc.last_report >= 500 {
                                acc.last_report = cur - (cur % 500);
                                emitter.tool_progress("building", &acc.name, cur);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        "message_delta" => {
            if let Some(sr) = ev
                .get("delta")
                .and_then(|d| d.get("stop_reason"))
                .and_then(|v| v.as_str())
            {
                state.stop_reason = Some(sr.to_string());
            }
        }
        _ => {}
    }
    Ok(())
}

pub fn finalize_anthropic(
    mut state: AnthropicParseState,
    emitter: &dyn EventSink,
) -> Result<LlmResponse> {
    if state.total_payloads > 0 && state.valid_chunks == 0 {
        return Err(AacodeError::Api(
            "stream returned no parseable data (all chunks malformed)".into(),
        ));
    }

    if state.stop_reason.is_none()
        && (!state.text.is_empty() || !state.reasoning.is_empty() || !state.blocks.is_empty())
    {
        state.stop_reason = Some("connection_closed".to_string());
    }

    if !state.reasoning.is_empty() {
        emitter.seg_large("thinking", &state.reasoning, 512);
    }
    emitter.seg_large("thought", &state.text, 512);

    let mut tool_calls = Vec::new();
    for (i, (_, acc)) in state.blocks.into_iter().enumerate() {
        if acc.kind != "tool_use" || acc.name.is_empty() {
            continue;
        }
        let args = if acc.partial_json.trim().is_empty() {
            "{}".to_string()
        } else {
            acc.partial_json
        };
        let id = if acc.id.is_empty() {
            format!("call_{i}")
        } else {
            acc.id
        };
        emitter.tool_progress("done", &acc.name, args.len());
        emitter.action(&acc.name, &args);
        tool_calls.push(ToolCall {
            id,
            name: acc.name,
            arguments: args,
        });
    }

    if matches!(state.stop_reason.as_deref(), Some("max_tokens")) {
        state.text.push_str(
            "\n\n[⚠️ WARNING: API response truncated (max_tokens). Reduce content or raise max_tokens.]",
        );
    }

    Ok(LlmResponse {
        text: state.text,
        tool_calls,
        reasoning_content: if state.reasoning.is_empty() {
            None
        } else {
            Some(state.reasoning)
        },
        finish_reason: state.stop_reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::CollectingSink;

    // ── OpenAI parsing ──────────────────────────────────────────────────

    #[test]
    fn openai_parses_content() {
        let sink = CollectingSink::new(false);
        let mut state = OpenAiParseState::default();
        parse_openai_chunk(r#"{"choices":[{"delta":{"content":"Hello"}}]}"#, &mut state, &sink).unwrap();
        parse_openai_chunk(r#"{"choices":[{"delta":{"content":" world"}}]}"#, &mut state, &sink).unwrap();
        assert_eq!(state.text, "Hello world");
        assert_eq!(state.valid_chunks, 2);
    }

    #[test]
    fn openai_detects_in_stream_error() {
        let sink = CollectingSink::new(false);
        let mut state = OpenAiParseState::default();
        let err = parse_openai_chunk(r#"{"error":{"message":"overloaded"}}"#, &mut state, &sink);
        assert!(err.is_err());
        assert!(format!("{}", err.unwrap_err()).contains("overloaded"));
    }

    #[test]
    fn openai_all_malformed_errors() {
        let sink = CollectingSink::new(false);
        let state = OpenAiParseState {
            total_payloads: 3,
            valid_chunks: 0,
            ..Default::default()
        };
        let r = finalize_openai(state, &sink);
        assert!(r.is_err());
        assert!(format!("{}", r.unwrap_err()).contains("malformed"));
    }

    #[test]
    fn openai_reasoning_accumulates() {
        let sink = CollectingSink::new(false);
        let mut state = OpenAiParseState::default();
        parse_openai_chunk(r#"{"choices":[{"delta":{"reasoning_content":"step 1"}}]}"#, &mut state, &sink).unwrap();
        parse_openai_chunk(r#"{"choices":[{"delta":{"reasoning_content":" step 2"}}]}"#, &mut state, &sink).unwrap();
        assert_eq!(state.reasoning, "step 1 step 2");
        assert!(sink.lines().iter().any(|l| l.contains(r#""seg":"thinking""#)));
    }

    #[test]
    fn openai_reasoning_alt_field() {
        let sink = CollectingSink::new(false);
        let mut state = OpenAiParseState::default();
        parse_openai_chunk(r#"{"choices":[{"delta":{"reasoning":"thinking"}}]}"#, &mut state, &sink).unwrap();
        assert_eq!(state.reasoning, "thinking");
    }

    #[test]
    fn openai_tool_call_fragments() {
        let sink = CollectingSink::new(false);
        let mut state = OpenAiParseState::default();
        parse_openai_chunk(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"run_shell","arguments":"{\"cmd\":"}}]}}]}"#, &mut state, &sink).unwrap();
        parse_openai_chunk(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"ls\"}"}}]}}]}"#, &mut state, &sink).unwrap();
        let resp = finalize_openai(state, &sink).unwrap();
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "run_shell");
        assert_eq!(resp.tool_calls[0].id, "c1");
        assert_eq!(resp.tool_calls[0].arguments, r#"{"cmd":"ls"}"#);
    }

    #[test]
    fn openai_tool_call_no_index_defaults_zero() {
        let sink = CollectingSink::new(false);
        let mut state = OpenAiParseState::default();
        parse_openai_chunk(r#"{"choices":[{"delta":{"tool_calls":[{"function":{"name":"ls"}}]}}]}"#, &mut state, &sink).unwrap();
        assert_eq!(state.tool_accs.get(&0).unwrap().name, "ls");
    }

    #[test]
    fn openai_length_truncation_warning() {
        let sink = CollectingSink::new(false);
        let state = OpenAiParseState {
            text: "partial".into(),
            finish_reason: Some("length".into()),
            ..Default::default()
        };
        let resp = finalize_openai(state, &sink).unwrap();
        assert!(resp.is_truncated());
        assert!(resp.text.contains("truncated"));
    }

    #[test]
    fn openai_content_with_finish_reason() {
        let sink = CollectingSink::new(false);
        let mut state = OpenAiParseState::default();
        parse_openai_chunk(r#"{"choices":[{"delta":{"content":"last"},"finish_reason":"stop"}]}"#, &mut state, &sink).unwrap();
        assert_eq!(state.text, "last");
        assert_eq!(state.finish_reason.as_deref(), Some("stop"));
    }

    // ── Anthropic parsing ───────────────────────────────────────────────

    #[test]
    fn anthropic_parses_text() {
        let sink = CollectingSink::new(false);
        let mut state = AnthropicParseState::default();
        parse_anthropic_chunk(r#"{"type":"content_block_start","index":0,"content_block":{"type":"text"}}"#, &mut state, &sink).unwrap();
        parse_anthropic_chunk(r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#, &mut state, &sink).unwrap();
        assert_eq!(state.text, "Hello");
    }

    #[test]
    fn anthropic_parses_tool_use() {
        let sink = CollectingSink::new(false);
        let mut state = AnthropicParseState::default();
        parse_anthropic_chunk(r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t1","name":"run_shell"}}"#, &mut state, &sink).unwrap();
        parse_anthropic_chunk(r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"cmd\":\"ls\"}"}}"#, &mut state, &sink).unwrap();
        let resp = finalize_anthropic(state, &sink).unwrap();
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "run_shell");
        assert_eq!(resp.tool_calls[0].id, "t1");
        assert_eq!(resp.tool_calls[0].parsed_args()["cmd"], "ls");
    }

    #[test]
    fn anthropic_detects_error_event() {
        let sink = CollectingSink::new(false);
        let mut state = AnthropicParseState::default();
        let err = parse_anthropic_chunk(r#"{"type":"error","error":{"type":"overloaded","message":"Overloaded"}}"#, &mut state, &sink);
        assert!(err.is_err());
        assert!(format!("{}", err.unwrap_err()).contains("Overloaded"));
    }

    #[test]
    fn anthropic_max_tokens_truncation() {
        let sink = CollectingSink::new(false);
        let state = AnthropicParseState {
            text: "partial".into(),
            stop_reason: Some("max_tokens".into()),
            ..Default::default()
        };
        let resp = finalize_anthropic(state, &sink).unwrap();
        assert!(resp.is_truncated());
        assert!(resp.text.contains("truncated"));
    }

    #[test]
    fn anthropic_connection_closed_detected() {
        let sink = CollectingSink::new(false);
        let state = AnthropicParseState {
            text: "some content".into(),
            stop_reason: None,
            ..Default::default()
        };
        let resp = finalize_anthropic(state, &sink).unwrap();
        assert_eq!(resp.finish_reason.as_deref(), Some("connection_closed"));
        assert!(resp.is_truncated());
    }

    #[test]
    fn anthropic_empty_tool_input_defaults() {
        let sink = CollectingSink::new(false);
        let mut state = AnthropicParseState::default();
        parse_anthropic_chunk(r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t1","name":"get_status"}}"#, &mut state, &sink).unwrap();
        let resp = finalize_anthropic(state, &sink).unwrap();
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].arguments, "{}");
    }

    #[test]
    fn anthropic_skips_malformed_json() {
        let sink = CollectingSink::new(false);
        let mut state = AnthropicParseState::default();
        parse_anthropic_chunk("not valid json", &mut state, &sink).unwrap();
        assert_eq!(state.total_payloads, 1);
        assert_eq!(state.valid_chunks, 0);
    }
}
