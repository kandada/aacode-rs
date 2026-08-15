// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Anthropic-compatible SSE stream parser.
//!
//! Anthropic SSE uses named events: `content_block_start`, `content_block_delta`
//! (`thinking_delta` / `text_delta` / `input_json_delta`), `message_delta`
//! (carries stop_reason). tool_use blocks accumulate their `input` JSON via
//! `input_json_delta`.

use super::sse::SseReader;
use super::types::{LlmResponse, ToolCall};
use crate::error::{AacodeError, Result};
use crate::stream::EventSink;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Default, Clone)]
struct BlockAcc {
    kind: String,         // "text" | "thinking" | "tool_use"
    id: String,
    name: String,
    text: String,         // for text/thinking
    partial_json: String, // for tool_use input
    name_announced: bool,
    last_report: usize,
}

/// Parse an Anthropic SSE stream from any reader.
pub fn parse_anthropic_stream<R: std::io::Read>(
    reader: R,
    emitter: &dyn EventSink,
    cancel: &AtomicBool,
) -> Result<LlmResponse> {
    let mut sse = SseReader::new(reader);
    let mut blocks: BTreeMap<i64, BlockAcc> = BTreeMap::new();
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut stop_reason: Option<String> = None;
    let mut valid_chunks: usize = 0;
    let mut total_payloads: usize = 0;

    while let Some(payload) = match sse.next_data() {
        Ok(Some(p)) => Some(p),
        Ok(None) => None,
        Err(e) => return Err(AacodeError::Network(format!("SSE read error: {e}"))),
    } {
        if cancel.load(Ordering::SeqCst) {
            return Err(AacodeError::Cancelled);
        }
        total_payloads += 1;
        let ev: Value = match serde_json::from_str(&payload) {
            Ok(v) => {
                valid_chunks += 1;
                v
            }
            Err(_) => continue,
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
                let mut acc = BlockAcc::default();
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
                blocks.insert(idx, acc);
            }
            "content_block_delta" => {
                let idx = ev.get("index").and_then(|v| v.as_i64()).unwrap_or(0);
                let delta = ev.get("delta");
                if let Some(d) = delta {
                    let dt = d.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    let acc = blocks.entry(idx).or_default();
                    match dt {
                        "text_delta" => {
                            if let Some(t) = d.get("text").and_then(|v| v.as_str()) {
                                text.push_str(t);
                                acc.text.push_str(t);
                                emitter.delta("thought", t);
                            }
                        }
                        "thinking_delta" => {
                            if let Some(t) = d.get("thinking").and_then(|v| v.as_str()) {
                                reasoning.push_str(t);
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
                    stop_reason = Some(sr.to_string());
                }
            }
            _ => {}
        }
    }

    if total_payloads > 0 && valid_chunks == 0 {
        return Err(AacodeError::Api(
            "stream returned no parseable data (all chunks malformed)".into(),
        ));
    }

    if stop_reason.is_none()
        && (!text.is_empty() || !reasoning.is_empty() || !blocks.is_empty())
    {
        stop_reason = Some("connection_closed".to_string());
    }

    if !reasoning.is_empty() {
        emitter.seg_large("thinking", &reasoning, 512);
    }
    emitter.seg_large("thought", &text, 512);

    let mut tool_calls = Vec::new();
    for (i, (_, acc)) in blocks.into_iter().enumerate() {
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

    if matches!(stop_reason.as_deref(), Some("max_tokens")) {
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
        finish_reason: stop_reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::CollectingSink;
    use std::io::Cursor;

    #[test]
    fn stream_parses_text_and_thinking() {
        let raw = concat!(
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"reason\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
            "data: [DONE]\n\n"
        );
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let resp =
            parse_anthropic_stream(Cursor::new(raw.as_bytes().to_vec()), &sink, &cancel).unwrap();
        assert_eq!(resp.text, "hi");
        assert_eq!(resp.reasoning_content.as_deref(), Some("reason"));
    }

    #[test]
    fn stream_parses_tool_use() {
        let raw = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"run_shell\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"command\\\":\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"ls\\\"}\"}}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n",
            "data: [DONE]\n\n"
        );
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let resp =
            parse_anthropic_stream(Cursor::new(raw.as_bytes().to_vec()), &sink, &cancel).unwrap();
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "run_shell");
        assert_eq!(resp.tool_calls[0].id, "t1");
        assert_eq!(resp.tool_calls[0].parsed_args()["command"], "ls");
    }

    #[test]
    fn stream_max_tokens_warning() {
        let raw = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"p\"}}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\"}}\n\n",
            "data: [DONE]\n\n"
        );
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let resp =
            parse_anthropic_stream(Cursor::new(raw.as_bytes().to_vec()), &sink, &cancel).unwrap();
        assert!(resp.is_truncated());
        assert!(resp.text.contains("truncated"));
    }

    #[test]
    fn sse_read_error_propagates_in_anthropic_parser() {
        struct BrokenReader;
        impl std::io::Read for BrokenReader {
            fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "simulated timeout",
                ))
            }
        }
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let r = parse_anthropic_stream(BrokenReader, &sink, &cancel);
        assert!(
            r.is_err(),
            "SSE read error must propagate to Anthropic parser"
        );
    }

    #[test]
    fn stream_no_stop_reason_gets_connection_closed() {
        let raw = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"content without stop\"}}\n\n"
        );
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let resp =
            parse_anthropic_stream(Cursor::new(raw.as_bytes().to_vec()), &sink, &cancel).unwrap();
        assert_eq!(resp.finish_reason, Some("connection_closed".to_string()));
        assert!(resp.is_truncated());
        assert_eq!(resp.text, "content without stop");
    }

    #[test]
    fn all_malformed_chunks_error() {
        let raw = concat!(
            "data: {not valid json\n\n",
            "data: }also not json\n\n",
            "data: [DONE]\n\n"
        );
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let r = parse_anthropic_stream(Cursor::new(raw.as_bytes().to_vec()), &sink, &cancel);
        assert!(r.is_err());
        assert!(format!("{}", r.err().unwrap()).contains("malformed"));
    }
}
