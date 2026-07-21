// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Minimal Server-Sent-Events (SSE) line framing.
//!
//! Reads an underlying byte stream and yields the payload of each `data:`
//! field. Terminates the iteration when it sees `data: [DONE]` or EOF.
//! Comment lines (`:`) and other SSE fields (event:, id:, retry:) are ignored,
//! which is sufficient for OpenAI/Anthropic chat streaming.

use std::io::{BufRead, BufReader, Read};

/// Iterator-like reader over SSE `data:` payloads.
pub struct SseReader<R: Read> {
    inner: BufReader<R>,
    done: bool,
}

impl<R: Read> SseReader<R> {
    pub fn new(reader: R) -> Self {
        SseReader {
            inner: BufReader::new(reader),
            done: false,
        }
    }

    /// Return the next `data:` payload as an owned String, or None at end.
    /// `[DONE]` sentinel yields None (stream complete).
    pub fn next_data(&mut self) -> Option<String> {
        if self.done {
            return None;
        }
        let mut line = String::new();
        loop {
            line.clear();
            let n = match self.inner.read_line(&mut line) {
                Ok(0) => {
                    // EOF
                    self.done = true;
                    return None;
                }
                Ok(n) => n,
                Err(_) => {
                    self.done = true;
                    return None;
                }
            };
            let _ = n;
            let trimmed = line.trim_end_matches(['\r', '\n']);
            if trimmed.is_empty() {
                // event boundary; keep scanning
                continue;
            }
            // Only care about `data:` fields.
            let payload = if let Some(rest) = trimmed.strip_prefix("data:") {
                rest.strip_prefix(' ').unwrap_or(rest)
            } else {
                // ignore event:/id:/retry:/comment lines
                continue;
            };
            if payload == "[DONE]" {
                self.done = true;
                return None;
            }
            return Some(payload.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parses_data_lines() {
        let raw = "data: {\"a\":1}\n\ndata: {\"b\":2}\n\ndata: [DONE]\n\n";
        let mut r = SseReader::new(Cursor::new(raw.as_bytes().to_vec()));
        assert_eq!(r.next_data().unwrap(), "{\"a\":1}");
        assert_eq!(r.next_data().unwrap(), "{\"b\":2}");
        assert!(r.next_data().is_none());
    }

    #[test]
    fn ignores_comments_and_other_fields() {
        let raw = ": ping\nevent: message\ndata: hello\nid: 5\n\n";
        let mut r = SseReader::new(Cursor::new(raw.as_bytes().to_vec()));
        assert_eq!(r.next_data().unwrap(), "hello");
        assert!(r.next_data().is_none());
    }

    #[test]
    fn handles_crlf_and_no_space_after_colon() {
        let raw = "data:{\"x\":1}\r\n\r\n";
        let mut r = SseReader::new(Cursor::new(raw.as_bytes().to_vec()));
        assert_eq!(r.next_data().unwrap(), "{\"x\":1}");
    }

    #[test]
    fn eof_without_done() {
        let raw = "data: a\n\ndata: b\n";
        let mut r = SseReader::new(Cursor::new(raw.as_bytes().to_vec()));
        assert_eq!(r.next_data().unwrap(), "a");
        assert_eq!(r.next_data().unwrap(), "b");
        assert!(r.next_data().is_none());
    }
}
