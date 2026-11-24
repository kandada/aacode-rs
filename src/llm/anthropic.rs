// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Anthropic-compatible SSE stream parser.
//!
//! Anthropic SSE uses named events: `content_block_start`, `content_block_delta`
//! (`thinking_delta` / `text_delta` / `input_json_delta`), `message_delta`
//! (carries stop_reason). tool_use blocks accumulate their `input` JSON via
//! `input_json_delta`.

use super::sse::SseReader;
use super::types::LlmResponse;
use crate::error::{AacodeError, Result};
use crate::stream::EventSink;
use std::sync::atomic::{AtomicBool, Ordering};

/// Parse an Anthropic SSE stream from any reader.
pub fn parse_anthropic_stream<R: std::io::Read>(
    reader: R,
    emitter: &dyn EventSink,
    cancel: &AtomicBool,
) -> Result<LlmResponse> {
    // Keep the sync wrapper as a thin I/O loop over the shared parser. The
    // previous duplicate implementation dropped provider `usage` entirely
    // (always `usage: None`), so sync callers could never account for tokens.
    let mut sse = SseReader::new(reader);
    let mut state = super::parse::AnthropicParseState::default();
    while let Some(payload) = match sse.next_data() {
        Ok(Some(p)) => Some(p),
        Ok(None) => None,
        Err(e) => {
            return Err(AacodeError::Network(format!("SSE read error: {e}")));
        }
    } {
        if cancel.load(Ordering::SeqCst) {
            return Err(AacodeError::Cancelled);
        }
        super::parse::parse_anthropic_chunk(&payload, &mut state, emitter)?;
    }
    super::parse::finalize_anthropic(state, emitter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::CollectingSink;
    use std::io::Cursor;

    #[test]
    fn sync_stream_preserves_provider_usage() {
        let raw = concat!(
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":20,\"cache_read_input_tokens\":700,\"cache_creation_input_tokens\":30,\"output_tokens\":1}}}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":12}}\n\n",
            "data: [DONE]\n\n"
        );
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let resp =
            parse_anthropic_stream(Cursor::new(raw.as_bytes().to_vec()), &sink, &cancel).unwrap();
        let usage = resp.usage.expect("provider usage");
        assert_eq!(usage.input_tokens, 750);
        assert_eq!(usage.cache_read_tokens, 700);
        assert_eq!(usage.cache_write_tokens, 30);
        assert_eq!(usage.output_tokens, 12);
    }

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
