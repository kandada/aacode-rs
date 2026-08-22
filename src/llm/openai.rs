// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! OpenAI-compatible SSE stream parser (sync I/O wrapper).
//!
//! Owns the I/O loop around `SseReader` and delegates wire-format
//! parsing to `super::parse::parse_openai_chunk` + `finalize_openai`,
//! so this binary client and the async client in `async_llm.rs`
//! share one implementation.

use super::parse::{self, OpenAiParseState};
use super::sse::SseReader;
use super::types::LlmResponse;
use crate::error::Result;
use crate::stream::EventSink;
use std::sync::atomic::{AtomicBool, Ordering};

/// Parse an OpenAI-style SSE chat stream from any reader.
pub fn parse_openai_stream<R: std::io::Read>(
    reader: R,
    emitter: &dyn EventSink,
    cancel: &AtomicBool,
) -> Result<LlmResponse> {
    let mut sse = SseReader::new(reader);
    let mut state = OpenAiParseState::default();

    while let Some(payload) = match sse.next_data() {
        Ok(Some(p)) => Some(p),
        Ok(None) => None,
        Err(e) => {
            return Err(crate::error::AacodeError::Network(format!(
                "SSE read error: {e}"
            )))
        }
    } {
        if cancel.load(Ordering::SeqCst) {
            return Err(crate::error::AacodeError::Cancelled);
        }
        parse::parse_openai_chunk(&payload, &mut state, emitter)?;
    }

    parse::finalize_openai(state, emitter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::CollectingSink;
    use std::io::Cursor;

    #[test]
    fn stream_parses_content() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let resp =
            parse_openai_stream(Cursor::new(raw.as_bytes().to_vec()), &sink, &cancel).unwrap();
        assert_eq!(resp.text, "Hello world");
        assert!(resp.tool_calls.is_empty());
        assert_eq!(resp.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn stream_accumulates_fragmented_tool_calls() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"run_shell\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"command\\\":\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"ls\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let resp =
            parse_openai_stream(Cursor::new(raw.as_bytes().to_vec()), &sink, &cancel).unwrap();
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "run_shell");
        assert_eq!(resp.tool_calls[0].id, "c1");
        assert_eq!(resp.tool_calls[0].arguments, "{\"command\":\"ls\"}");
        assert_eq!(resp.tool_calls[0].parsed_args()["command"], "ls");
    }

    #[test]
    fn stream_separates_reasoning_and_content() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"let me think\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n",
            "data: [DONE]\n\n"
        );
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let resp =
            parse_openai_stream(Cursor::new(raw.as_bytes().to_vec()), &sink, &cancel).unwrap();
        assert_eq!(resp.text, "answer");
        assert_eq!(resp.reasoning_content.as_deref(), Some("let me think"));
        let lines = sink.lines();
        assert!(lines.iter().any(|l| l.contains(r#""seg":"thinking""#)));
        assert!(lines.iter().any(|l| l.contains(r#""seg":"thought""#)));
    }

    #[test]
    fn stream_truncation_warning() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let resp =
            parse_openai_stream(Cursor::new(raw.as_bytes().to_vec()), &sink, &cancel).unwrap();
        assert!(resp.is_truncated());
        assert!(resp.text.contains("truncated"));
    }

    #[test]
    fn stream_respects_cancel() {
        let raw = "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n";
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(true);
        let r = parse_openai_stream(Cursor::new(raw.as_bytes().to_vec()), &sink, &cancel);
        assert!(matches!(r, Err(crate::error::AacodeError::Cancelled)));
    }

    #[test]
    fn sse_read_error_propagates_as_stream_error() {
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
        let r = parse_openai_stream(BrokenReader, &sink, &cancel);
        assert!(
            r.is_err(),
            "SSE read error must propagate, not silently return Ok"
        );
    }

    #[test]
    fn mid_stream_disconnect_not_empty_tool_calls() {
        struct DropAfterFirst {
            sent: bool,
        }
        impl std::io::Read for DropAfterFirst {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                if self.sent {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::ConnectionReset,
                        "reset",
                    ))
                } else {
                    self.sent = true;
                    let data =
                        b"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n";
                    let n = data.len().min(buf.len());
                    buf[..n].copy_from_slice(&data[..n]);
                    Ok(n)
                }
            }
        }
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let r = parse_openai_stream(DropAfterFirst { sent: false }, &sink, &cancel);
        assert!(
            r.is_err(),
            "mid-stream disconnect must error, not return empty tool_calls"
        );
    }

    #[test]
    fn all_malformed_chunks_error_instead_of_empty_response() {
        let raw = concat!(
            "data: {this is not valid json\n\n",
            "data: }also not json\n\n",
            "data: [DONE]\n\n"
        );
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let r = parse_openai_stream(Cursor::new(raw.as_bytes().to_vec()), &sink, &cancel);
        assert!(
            r.is_err(),
            "all-malformed chunks must produce an error"
        );
        let msg = format!("{}", r.err().unwrap());
        assert!(msg.contains("malformed") || msg.contains("parseable"));
    }

    #[test]
    fn thinking_seg_emitted_at_stream_end() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think 1\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\" think 2\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"answer 1\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\" answer 2\"}}]}\n\n",
            "data: [DONE]\n\n"
        );
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let resp =
            parse_openai_stream(Cursor::new(raw.as_bytes().to_vec()), &sink, &cancel).unwrap();
        assert_eq!(resp.text, "answer 1 answer 2");
        assert_eq!(resp.reasoning_content.as_deref(), Some("think 1 think 2"));

        let lines = sink.lines();
        let first_delta = lines
            .iter()
            .position(|l| l.contains(r#""type":"delta""#))
            .expect("delta missing");
        let thinking_seg_pos = lines
            .iter()
            .position(|l| {
                l.contains(r#""type":"seg_content""#) && l.contains(r#""seg":"thinking""#)
            })
            .expect("thinking seg missing");
        let thought_seg_pos = lines
            .iter()
            .position(|l| {
                l.contains(r#""type":"seg_content""#) && l.contains(r#""seg":"thought""#)
            })
            .expect("thought seg missing");

        assert!(
            thinking_seg_pos > first_delta,
            "thinking seg must come after deltas"
        );
        assert!(
            thinking_seg_pos < thought_seg_pos,
            "thinking seg must come before thought seg"
        );
    }

    #[test]
    fn reasoning_only_stream_emits_thinking_seg() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking only\"}}]}\n\n",
            "data: [DONE]\n\n"
        );
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let resp =
            parse_openai_stream(Cursor::new(raw.as_bytes().to_vec()), &sink, &cancel).unwrap();
        assert_eq!(resp.text, "");
        assert_eq!(resp.reasoning_content.as_deref(), Some("thinking only"));

        let lines = sink.lines();
        assert!(lines.iter().any(|l| l.contains(r#""seg":"thinking""#)));
        assert!(lines.iter().any(|l| l.contains(r#""seg":"thought""#)));
    }

    #[test]
    fn content_only_stream_no_thinking_seg() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
            "data: [DONE]\n\n"
        );
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let resp =
            parse_openai_stream(Cursor::new(raw.as_bytes().to_vec()), &sink, &cancel).unwrap();
        assert_eq!(resp.text, "hello");
        assert!(resp.reasoning_content.is_none());

        let lines = sink.lines();
        assert!(!lines.iter().any(|l| l.contains(r#""seg":"thinking""#)));
        assert!(lines.iter().any(|l| l.contains(r#""seg":"thought""#)));
    }

    #[test]
    fn mid_stream_failure_does_not_emit_thinking_seg() {
        struct FailAfterTwoReads {
            reads: u32,
        }
        impl std::io::Read for FailAfterTwoReads {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                self.reads += 1;
                if self.reads > 2 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::ConnectionReset,
                        "mid-stream disconnect",
                    ));
                }
                let data = b"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"partial\"}}]}\n\n";
                let n = data.len().min(buf.len());
                buf[..n].copy_from_slice(&data[..n]);
                Ok(n)
            }
        }
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let r = parse_openai_stream(FailAfterTwoReads { reads: 0 }, &sink, &cancel);
        assert!(r.is_err());

        let lines = sink.lines();
        let has_thinking_delta = lines
            .iter()
            .any(|l| l.contains(r#""type":"delta""#) && l.contains(r#""seg":"thinking""#));
        assert!(has_thinking_delta);
        assert!(!lines.iter().any(|l| {
            l.contains(r#""type":"seg_content""#) && l.contains(r#""seg":"thinking""#)
        }));
    }

    #[test]
    fn thinking_seg_content_matches_accumulated_deltas() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"step 1\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\" step 2\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"final\"}}]}\n\n",
            "data: [DONE]\n\n"
        );
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let resp =
            parse_openai_stream(Cursor::new(raw.as_bytes().to_vec()), &sink, &cancel).unwrap();
        assert_eq!(resp.reasoning_content.as_deref(), Some("step 1 step 2"));

        let lines = sink.lines();
        let thinking_seg_line = lines
            .iter()
            .find(|l| {
                l.contains(r#""type":"seg_content""#) && l.contains(r#""seg":"thinking""#)
            })
            .expect("thinking seg missing");
        let seg_json: serde_json::Value =
            serde_json::from_str(thinking_seg_line).expect("invalid JSON");
        let seg_content = seg_json["content"].as_str().unwrap();

        let accumulated_deltas: String = lines
            .iter()
            .filter(|l| {
                l.contains(r#""type":"delta""#) && l.contains(r#""seg":"thinking""#)
            })
            .map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).unwrap();
                v["content"].as_str().unwrap().to_string()
            })
            .collect();

        assert_eq!(seg_content, accumulated_deltas);
    }

    #[test]
    fn interleaved_reasoning_content_thinking_seg_still_at_end() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"I need\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"Let's\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\" to check\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\" do it\"}}]}\n\n",
            "data: [DONE]\n\n"
        );
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let resp =
            parse_openai_stream(Cursor::new(raw.as_bytes().to_vec()), &sink, &cancel).unwrap();
        assert_eq!(resp.text, "Let's do it");
        assert_eq!(resp.reasoning_content.as_deref(), Some("I need to check"));

        let lines = sink.lines();
        let thinking_seg_count = lines
            .iter()
            .filter(|l| {
                l.contains(r#""type":"seg_content""#) && l.contains(r#""seg":"thinking""#)
            })
            .count();
        assert_eq!(thinking_seg_count, 1);
        let thought_seg_count = lines
            .iter()
            .filter(|l| {
                l.contains(r#""type":"seg_content""#) && l.contains(r#""seg":"thought""#)
            })
            .count();
        assert_eq!(thought_seg_count, 1);
    }

    // ── thinking extraction: end-to-end via SSE I/O loop ────────────────

    /// `thinking` field works end-to-end through the sync SSE reader.
    #[test]
    fn sync_stream_thinking_field() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"thinking\":\"plan X\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"do it\"}}]}\n\n",
            "data: [DONE]\n\n"
        );
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let resp =
            parse_openai_stream(Cursor::new(raw.as_bytes().to_vec()), &sink, &cancel).unwrap();
        assert_eq!(resp.reasoning_content.as_deref(), Some("plan X"));
        assert_eq!(resp.text, "do it");
    }

    /// `<think>` tag works end-to-end and survives chunk boundaries.
    #[test]
    fn sync_stream_think_tag_split_across_chunks() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"<think>I am thi\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"nking</think>done\"}}]}\n\n",
            "data: [DONE]\n\n"
        );
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let resp =
            parse_openai_stream(Cursor::new(raw.as_bytes().to_vec()), &sink, &cancel).unwrap();
        assert_eq!(resp.reasoning_content.as_deref(), Some("I am thinking"));
        assert_eq!(resp.text, "done");
        let lines = sink.lines();
        assert!(lines.iter().any(|l| l.contains(r#""seg":"thinking""#)));
        assert!(lines.iter().any(|l| l.contains(r#""seg":"thought""#)));
    }
}
