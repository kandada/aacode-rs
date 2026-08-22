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

/// Field names tried in priority order to extract thinking from a delta.
///
/// 1. `reasoning_content` — DeepSeek / Kimi-K2-thinking convention
/// 2. `reasoning` — some private-provider convention
/// 3. `thinking` — another private-provider convention
///
/// Field strategies take priority over the inline `<think>...</think>` tag
/// strategy to avoid double-counting when both formats appear in the same
/// response. **Not model-specific**: any provider whose delta uses one of
/// these field names (or inline tags) is supported without configuration.
pub const OPENAI_THINKING_FIELDS: &[&str] = &["reasoning_content", "reasoning", "thinking"];

/// `<think>` opening tag (Qwen-style inline reasoning).
const TAG_OPEN: &str = "<think>";
/// `</think>` closing tag.
const TAG_CLOSE: &str = "</think>";

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
    /// Cross-chunk hold-back buffer for the inline `<think>` tag parser.
    /// Always flushed by `finalize_openai`.
    pub tag_buffer: String,
    /// Whether the tag parser is currently inside a `<think>` block.
    pub in_think: bool,
    /// Whether the very first fragment of the current `thinking`
    /// segment has been emitted. Cleared by `feed_tag_parser` each
    /// time the parser toggles into `in_think=true`. Used to drop the
    /// formatting whitespace right after `<think>`.
    pub first_think_pending: bool,
    /// Same as `first_think_pending`, for the `thought` channel.
    pub first_content_pending: bool,
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
            tag_buffer: String::new(),
            in_think: false,
            first_think_pending: true,
            first_content_pending: true,
        }
    }
}

/// Parse one OpenAI SSE `data:` payload into the accumulator state.
/// Returns `Ok(())` on success. The caller handles the I/O loop and
/// calls `finalize_openai` afterwards.
///
/// Thinking extraction follows a two-tier strategy:
///
/// 1. **Field strategies** — try each name in `OPENAI_THINKING_FIELDS`
///    in priority order. If one is present, the field value becomes
///    `thinking` and `content` is passed through unchanged (no tag
///    re-parsing, to avoid double-counting when both formats coexist).
/// 2. **Tag strategy** — if no field hit, run `content` through the
///    stateful `<think>...</think>` parser.
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

    // ── Layer A: field-based strategies ──
    let mut field_hit = false;
    for &field in OPENAI_THINKING_FIELDS {
        if let Some(rc) = delta.get(field).and_then(|v| v.as_str()) {
            if !rc.is_empty() {
                state.reasoning.push_str(rc);
                emitter.delta("thinking", rc);
            }
            field_hit = true;
            break;
        }
    }

    // Field hit → pass content through unchanged (no tag re-parsing)
    if field_hit {
        if let Some(c) = delta.get("content").and_then(|v| v.as_str()) {
            if !c.is_empty() {
                state.text.push_str(c);
                emitter.delta("thought", c);
            }
        }
    } else {
        // ── Layer B: tag strategy ──
        let content = delta
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let (think_chunk, content_chunk) = feed_tag_parser(
            &mut state.tag_buffer,
            &mut state.in_think,
            &mut state.first_think_pending,
            &mut state.first_content_pending,
            content,
        );
        if !think_chunk.is_empty() {
            state.reasoning.push_str(&think_chunk);
            emitter.delta("thinking", &think_chunk);
        }
        if !content_chunk.is_empty() {
            state.text.push_str(&content_chunk);
            emitter.delta("thought", &content_chunk);
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

/// Feed one content chunk into the `<think>...</think>` parser.
///
/// Returns `(think_chunk, content_chunk)` to emit immediately. Both
/// may be empty.
///
/// Hold-back model: only the suffix starting at the **first `<`**
/// in the residual buffer can possibly form the next tag (`<think>` /
/// `</think>`); everything before that `<` is guaranteed not to start
/// a tag and is emitted right away. The suffix itself stays in
/// `tag_buffer` for the next call. This replaces the previous
/// "fixed 8-byte look-behind" model, which emitted trailing/leading
/// whitespace (e.g. the lone `\n` right after `<think>`) as spurious
/// 1-byte `thinking` deltas that leaked into `state.reasoning` and
/// showed up as blank lines at the top of thinking segments in the
/// mobile UI.
///
/// Whitespace trimming around tag boundaries:
/// - A segment immediately preceding a tag close (`</think>`) gets
///   its trailing whitespace stripped (the model's formatting newline
///   right before `</think>` is not real content).
/// - The very first emit of a `thinking` / `thought` segment gets its
///   leading whitespace stripped (the formatting newline right after
///   `<think>` or `</think>` is not real content either).
///
/// Together this kills the 1-byte phantom bubble and prevents leading
/// or trailing newlines from polluting segment text. `first_think`
/// / `first_content` are stored on the caller's state so the flag
/// survives across chunks (a thinking segment often spans many
/// chunks).
///
/// Tag-content and tag-finding logic are unchanged.
fn feed_tag_parser(
    tag_buffer: &mut String,
    in_think: &mut bool,
    first_think_pending: &mut bool,
    first_content_pending: &mut bool,
    content_chunk: &str,
) -> (String, String) {
    if content_chunk.is_empty() {
        return (String::new(), String::new());
    }

    let mut full = std::mem::take(tag_buffer);
    full.push_str(content_chunk);

    let mut out_think = String::new();
    let mut out_content = String::new();
    let mut last_safe_idx = 0;

    loop {
        let search_from = &full[last_safe_idx..];
        let (needle, _needle_len) = if *in_think {
            (TAG_CLOSE, TAG_CLOSE.len())
        } else {
            (TAG_OPEN, TAG_OPEN.len())
        };
        if let Some(rel_idx) = search_from.find(needle) {
            let abs_idx = last_safe_idx + rel_idx;
            let segment = &full[last_safe_idx..abs_idx];
            // The segment just before this tag is always "before tag
            // close" from that segment's POV — it's flush against the
            // tag boundary, so formatting whitespace there is noise.
            if *in_think {
                push_segment(&mut out_think, segment, true, *first_think_pending);
                *first_think_pending = false;
            } else {
                push_segment(&mut out_content, segment, false, *first_content_pending);
                *first_content_pending = false;
            }
            *in_think = !*in_think;
            last_safe_idx = abs_idx + needle.len();
            // After toggling, the NEXT segment of the channel we just
            // entered will see a "first emit" (so a fresh tag open /
            // close resets the first-emit flag for the channel we're
            // about to emit into).
            if *in_think {
                *first_think_pending = true;
            } else {
                *first_content_pending = true;
            }
        } else {
            break;
        }
    }

    let remaining = &full[last_safe_idx..];
    if remaining.is_empty() {
        return (out_think, out_content);
    }

    // Emit everything up to the first `<` (guaranteed not to start a
    // tag), and hold back the rest. `find` always lands on a byte
    // boundary because `<` is 1-byte ASCII — so UTF-8 content is
    // never cut mid-codepoint.
    let lt_idx = remaining.find('<');
    let (emit_part, hold_part) = match lt_idx {
        Some(i) => (&remaining[..i], &remaining[i..]),
        None => (remaining, ""),
    };

    if !emit_part.is_empty() {
        // This residue is NOT followed by a tag in this call (else
        // the loop above would've matched it). No trailing-trim here;
        // only the first-emit leading-trim applies.
        if *in_think {
            push_segment(&mut out_think, emit_part, false, *first_think_pending);
            *first_think_pending = false;
        } else {
            push_segment(&mut out_content, emit_part, false, *first_content_pending);
            *first_content_pending = false;
        }
    }
    if !hold_part.is_empty() {
        tag_buffer.push_str(hold_part);
    }

    (out_think, out_content)
}

/// Push a non-empty `seg` into `out`, trimming formatting whitespace
/// around tag boundaries:
///   • `is_before_tag_close` — the segment is immediately followed
///     by `</think>`; trim trailing whitespace.
///   • `is_first` — this is the very first emit of the current segment
///     type in this stream; trim leading whitespace.
/// Together they kill the 1-byte phantom bubble. Internal whitespace
/// inside a segment (e.g. `"hello\nworld"`) is preserved.
fn push_segment(out: &mut String, seg: &str, is_before_tag_close: bool, is_first: bool) {
    if seg.is_empty() {
        return;
    }
    let to_emit: std::borrow::Cow<'_, str> = if is_first && is_before_tag_close {
        std::borrow::Cow::Owned(seg.trim().to_string())
    } else if is_first {
        std::borrow::Cow::Owned(seg.trim_start().to_string())
    } else if is_before_tag_close {
        std::borrow::Cow::Owned(seg.trim_end().to_string())
    } else {
        std::borrow::Cow::Borrowed(seg)
    };
    if to_emit.is_empty() {
        return;
    }
    out.push_str(&to_emit);
}

/// Flush the tag parser's residual buffer at end-of-stream.
///
/// Returns whatever was still held back, attributed based on `in_think`:
/// - if still inside `<think>`, treat the residue as thinking content
///   (best-effort recovery from an unclosed tag).
/// - otherwise treat it as plain content.
fn flush_tag_parser(tag_buffer: &mut String, in_think: bool) -> (String, String) {
    if tag_buffer.is_empty() {
        return (String::new(), String::new());
    }
    let out = std::mem::take(tag_buffer);
    if in_think {
        (out, String::new())
    } else {
        (String::new(), out)
    }
}

/// Finalize: check for all-malformed, emit end segments, assemble tool_calls.
pub fn finalize_openai(
    mut state: OpenAiParseState,
    emitter: &dyn EventSink,
) -> Result<LlmResponse> {
    if state.total_payloads > 0 && state.valid_chunks == 0 {
        return Err(AacodeError::Api(
            "stream returned no parseable data (all chunks malformed)".into(),
        ));
    }

    // Flush any cross-chunk residue from the <think> tag parser
    // before assembling the final response.
    let (tail_think, tail_content) = flush_tag_parser(&mut state.tag_buffer, state.in_think);
    if !tail_think.is_empty() {
        state.reasoning.push_str(&tail_think);
    }
    if !tail_content.is_empty() {
        state.text.push_str(&tail_content);
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

    // ── thinking extraction: field strategies ───────────────────────────

    /// `thinking` field (third in priority order) is recognized.
    #[test]
    fn openai_thinking_field_recognized() {
        let sink = CollectingSink::new(false);
        let mut state = OpenAiParseState::default();
        parse_openai_chunk(
            r#"{"choices":[{"delta":{"thinking":"plan A"}}]}"#,
            &mut state,
            &sink,
        )
        .unwrap();
        parse_openai_chunk(
            r#"{"choices":[{"delta":{"content":"go"}}]}"#,
            &mut state,
            &sink,
        )
        .unwrap();
        assert_eq!(state.reasoning, "plan A");
        assert_eq!(state.text, "go");
    }

    /// Among the three field strategies, `reasoning_content` wins
    /// (highest priority) even when other fields are present.
    #[test]
    fn openai_field_priority_reasoning_content_first() {
        let sink = CollectingSink::new(false);
        let mut state = OpenAiParseState::default();
        parse_openai_chunk(
            r#"{"choices":[{"delta":{"reasoning_content":"R","reasoning":"Re","thinking":"T"}}]}"#,
            &mut state,
            &sink,
        )
        .unwrap();
        assert_eq!(state.reasoning, "R");
    }

    /// `reasoning` beats `thinking` (second priority).
    #[test]
    fn openai_field_priority_reasoning_beats_thinking() {
        let sink = CollectingSink::new(false);
        let mut state = OpenAiParseState::default();
        parse_openai_chunk(
            r#"{"choices":[{"delta":{"reasoning":"Re","thinking":"T"}}]}"#,
            &mut state,
            &sink,
        )
        .unwrap();
        assert_eq!(state.reasoning, "Re");
    }

    /// When a field strategy hits, `content` is passed through unchanged
    /// — no tag re-parsing. This prevents double-counting when a model
    /// uses fields AND happens to include `<think>` text in content.
    #[test]
    fn openai_field_hit_does_not_reparse_content_tags() {
        let sink = CollectingSink::new(false);
        let mut state = OpenAiParseState::default();
        parse_openai_chunk(
            r#"{"choices":[{"delta":{"reasoning_content":"R","content":"<think>T</think>C"}}]}"#,
            &mut state,
            &sink,
        )
        .unwrap();
        assert_eq!(state.reasoning, "R");
        // content with embedded tags passes through verbatim
        assert_eq!(state.text, "<think>T</think>C");
    }

    // ── thinking extraction: inline <think> tag strategy ────────────────

    /// Complete inline tag in one chunk: cleanly splits thinking from content.
    #[test]
    fn openai_think_tag_complete_in_one_chunk() {
        let sink = CollectingSink::new(false);
        let mut state = OpenAiParseState::default();
        parse_openai_chunk(
            r#"{"choices":[{"delta":{"content":"<think>plan</think>result"}}]}"#,
            &mut state,
            &sink,
        )
        .unwrap();
        assert_eq!(state.reasoning, "plan");
        assert_eq!(state.text, "result");
    }

    /// Tag opens across two chunks: `<thi` + `nk>hello</think>world`.
    #[test]
    fn openai_think_tag_open_split_across_chunks() {
        let sink = CollectingSink::new(false);
        let mut state = OpenAiParseState::default();
        parse_openai_chunk(
            r#"{"choices":[{"delta":{"content":"<thi"}}]}"#,
            &mut state,
            &sink,
        )
        .unwrap();
        parse_openai_chunk(
            r#"{"choices":[{"delta":{"content":"nk>hello</think>world"}}]}"#,
            &mut state,
            &sink,
        )
        .unwrap();
        assert_eq!(state.reasoning, "hello");
        assert_eq!(state.text, "world");
    }

    /// Closing tag split across chunks: `<think>hello</th` + `ink>world`.
    #[test]
    fn openai_think_tag_close_split_across_chunks() {
        let sink = CollectingSink::new(false);
        let mut state = OpenAiParseState::default();
        parse_openai_chunk(
            r#"{"choices":[{"delta":{"content":"<think>hello</th"}}]}"#,
            &mut state,
            &sink,
        )
        .unwrap();
        parse_openai_chunk(
            r#"{"choices":[{"delta":{"content":"ink>world"}}]}"#,
            &mut state,
            &sink,
        )
        .unwrap();
        assert_eq!(state.reasoning, "hello");
        assert_eq!(state.text, "world");
    }

    /// Pathological: one char per chunk.
    #[test]
    fn openai_think_tag_one_char_per_chunk() {
        let sink = CollectingSink::new(false);
        let mut state = OpenAiParseState::default();
        let s = "<think>thinking</think>answer";
        for c in s.chars() {
            let payload = format!(r#"{{"choices":[{{"delta":{{"content":"{c}"}}}}]}}"#);
            parse_openai_chunk(&payload, &mut state, &sink).unwrap();
        }
        assert_eq!(state.reasoning, "thinking");
        assert_eq!(state.text, "answer");
    }

    /// Unclosed `<think>` at end of stream: residue flushed as thinking
    /// (best-effort recovery).
    #[test]
    fn openai_think_tag_unclosed_flushed_as_thinking() {
        let sink = CollectingSink::new(false);
        let mut state = OpenAiParseState::default();
        parse_openai_chunk(
            r#"{"choices":[{"delta":{"content":"<think>still thinking"}}]}"#,
            &mut state,
            &sink,
        )
        .unwrap();
        let resp = finalize_openai(state, &sink).unwrap();
        assert_eq!(resp.reasoning_content.as_deref(), Some("still thinking"));
        assert_eq!(resp.text, "");
    }

    /// Multiple consecutive think blocks within a single stream.
    #[test]
    fn openai_think_tag_multiple_blocks() {
        let sink = CollectingSink::new(false);
        let mut state = OpenAiParseState::default();
        parse_openai_chunk(
            r#"{"choices":[{"delta":{"content":"<think>a</think>X<think>b</think>Y"}}]}"#,
            &mut state,
            &sink,
        )
        .unwrap();
        assert_eq!(state.reasoning, "ab");
        assert_eq!(state.text, "XY");
    }

    /// Plain text without any tags is treated as plain content.
    #[test]
    fn openai_no_tags_plain_content() {
        let sink = CollectingSink::new(false);
        let mut state = OpenAiParseState::default();
        parse_openai_chunk(
            r#"{"choices":[{"delta":{"content":"plain answer"}}]}"#,
            &mut state,
            &sink,
        )
        .unwrap();
        assert_eq!(state.reasoning, "");
        assert_eq!(state.text, "plain answer");
    }

    /// Optimization: in normal mode, content not starting with `<` is
    /// emitted immediately with no latency from the look-behind buffer.
    #[test]
    fn openai_normal_mode_emits_non_tag_buffer_immediately() {
        let sink = CollectingSink::new(false);
        let mut state = OpenAiParseState::default();
        parse_openai_chunk(
            r#"{"choices":[{"delta":{"content":"hello world, no tags here"}}]}"#,
            &mut state,
            &sink,
        )
        .unwrap();
        // All content flushed in the same chunk (no buffering delay).
        assert_eq!(state.text, "hello world, no tags here");
        assert!(state.tag_buffer.is_empty());
    }

    // ── thinking extraction: backward compatibility ─────────────────────

    /// Existing DeepSeek/Kimi flow (`reasoning_content`) is unchanged:
    /// still recognized, still accumulates, still emits thinking seg.
    #[test]
    fn openai_existing_reasoning_content_flow_unchanged() {
        let sink = CollectingSink::new(false);
        let mut state = OpenAiParseState::default();
        parse_openai_chunk(
            r#"{"choices":[{"delta":{"reasoning_content":"Let me "}}]}"#,
            &mut state,
            &sink,
        )
        .unwrap();
        parse_openai_chunk(
            r#"{"choices":[{"delta":{"reasoning_content":"think..."}}]}"#,
            &mut state,
            &sink,
        )
        .unwrap();
        parse_openai_chunk(
            r#"{"choices":[{"delta":{"content":"answer"}}]}"#,
            &mut state,
            &sink,
        )
        .unwrap();
        let resp = finalize_openai(state, &sink).unwrap();
        assert_eq!(resp.reasoning_content.as_deref(), Some("Let me think..."));
        assert_eq!(resp.text, "answer");
        // Both thinking and thought segs are emitted at finalize.
        let lines = sink.lines();
        assert!(lines.iter().any(|l| l.contains(r#""seg":"thinking""#)));
        assert!(lines.iter().any(|l| l.contains(r#""seg":"thought""#)));
    }

    // ── thinking extraction: no phantom bubble / no leading-whitespace leak ─

    /// Regression: a chunk arriving right after `<think>` that carries only
    /// the model's standard `\n` + a few characters (e.g. MiniMax-M3 stream
    /// chunk 1 = `"<think>\nThe user"`) must NOT produce a 1-byte thinking
    /// delta carrying just `"\n"`. The leading whitespace after the open
    /// tag is dropped on first emit; the first real thinking delta should
    /// carry real text from the first chunk that has any.
    #[test]
    fn openai_no_phantom_newline_thinking_bubble() {
        let sink = CollectingSink::new(false);
        let mut state = OpenAiParseState::default();
        // Chunk 1: just "<think>\n" + 1 trailing char — small enough that
        // the old fixed-8-byte hold-back would emit "\n" alone here.
        parse_openai_chunk(
            r#"{"choices":[{"delta":{"content":"<think>\nT"}}]}"#,
            &mut state,
            &sink,
        )
        .unwrap();
        // No 1-byte "\n" delta fires; the only one we'd see is when real
        // text actually arrives. In a single-chunk world state stays
        // empty (hold-back wins), in a multi-chunk world the first emit
        // is the trimmed real text.
        assert!(
            !sink.lines().iter().any(|l| {
                l.contains(r#""type":"delta""#)
                    && l.contains(r#""seg":"thinking""#)
                    && l.contains(r#""content":"\n""#)
            }),
            "no standalone `\\n` thinking delta should be emitted; saw: {:?}",
            sink.lines()
        );

        // Chunk 2: rest arrives, including the close tag + leading
        // newline + actual response.
        parse_openai_chunk(
            r#"{"choices":[{"delta":{"content":"he user</think>\nActual"}}]}"#,
            &mut state,
            &sink,
        )
        .unwrap();
        let resp = finalize_openai(state, &sink).unwrap();
        // Final reasoning / text must not have a leading newline.
        assert_eq!(resp.reasoning_content.as_deref(), Some("The user"));
        assert_eq!(resp.text, "Actual");
    }

    /// A model that puts a leading `\n` right before `</think>\n` (e.g.
    /// `"...thinking\n</think>\nresponse"`) must not leak the newline
    /// into either `state.reasoning` or `state.text`.
    #[test]
    fn openai_no_leading_newline_in_segments_after_tag() {
        let sink = CollectingSink::new(false);
        let mut state = OpenAiParseState::default();
        parse_openai_chunk(
            r#"{"choices":[{"delta":{"content":"<think>plan\n</think>\nresult"}}]}"#,
            &mut state,
            &sink,
        )
        .unwrap();
        let resp = finalize_openai(state, &sink).unwrap();
        assert_eq!(resp.reasoning_content.as_deref(), Some("plan"));
        assert_eq!(resp.text, "result");
    }

    /// Even when the post-`<think>` whitespace arrives in its own chunk
    /// (no other content), it must NOT produce a phantom thinking delta
    /// — it must stay buffered and only contribute (as trimmed prefix)
    /// once real text arrives.
    #[test]
    fn openai_lone_newline_chunk_emits_no_delta() {
        let sink = CollectingSink::new(false);
        let mut state = OpenAiParseState::default();
        // Open tag only.
        parse_openai_chunk(
            r#"{"choices":[{"delta":{"content":"<think>"}}]}"#,
            &mut state,
            &sink,
        )
        .unwrap();
        // Just whitespace.
        parse_openai_chunk(
            r#"{"choices":[{"delta":{"content":"\n   \t"}}]}"#,
            &mut state,
            &sink,
        )
        .unwrap();
        // Real text + close + leading whitespace + actual response.
        parse_openai_chunk(
            r#"{"choices":[{"delta":{"content":"real</think>\nresp"}}]}"#,
            &mut state,
            &sink,
        )
        .unwrap();
        let resp = finalize_openai(state, &sink).unwrap();
        assert_eq!(resp.reasoning_content.as_deref(), Some("real"));
        assert_eq!(resp.text, "resp");

        // None of the deltas should carry pure whitespace.
        for l in sink.lines() {
            if l.contains(r#""type":"delta""#) {
                let v: serde_json::Value = serde_json::from_str(&l).unwrap();
                let content = v["content"].as_str().unwrap_or("");
                let trimmed = content.trim();
                assert!(
                    !trimmed.is_empty(),
                    "no pure-whitespace delta should fire: {l}"
                );
            }
        }
    }

    /// Mid-chunk tag split still works exactly like before: chunk ends
    /// in the middle of `</think` and the next chunk provides the rest.
    /// The new `<`-based hold-back must still emit the leading part
    /// cleanly once the tag is matched.
    #[test]
    fn openai_tag_split_chunk_boundary_still_correct() {
        let sink = CollectingSink::new(false);
        let mut state = OpenAiParseState::default();
        // First chunk ends with "<" — just the leading char of "</think".
        parse_openai_chunk(
            r#"{"choices":[{"delta":{"content":"<think>plan X<"}}]}"#,
            &mut state,
            &sink,
        )
        .unwrap();
        // The "<" must stay buffered; no phantom thinking delta for it.
        // (state.reasoning may include the "plan X" already, that's fine.)
        parse_openai_chunk(
            r#"{"choices":[{"delta":{"content":"/think>actual"}}]}"#,
            &mut state,
            &sink,
        )
        .unwrap();
        let resp = finalize_openai(state, &sink).unwrap();
        assert_eq!(resp.reasoning_content.as_deref(), Some("plan X"));
        assert_eq!(resp.text, "actual");
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
