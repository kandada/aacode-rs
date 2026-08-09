// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! OpenAI-compatible streaming client (DeepSeek / Kimi / GPT / etc).
//!
//! Uses synchronous `ureq` and the hand-written SSE reader. Streams
//! `delta.content` / `delta.reasoning_content` / `delta.tool_calls`, emitting
//! segments to the sink and accumulating fragmented tool_calls by index.

use super::sse::SseReader;
use super::types::{ChatMessage, LlmResponse, ToolCall};
use super::LlmClient;
use super::truncate;
use crate::config::ModelConfig;
use crate::error::{AacodeError, Result};
use crate::stream::EventSink;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Connect timeout: fail fast on dead links so the retry loop can kick in
/// (mobile networks flap; an unbounded connect looks like a frozen UI).
const CONNECT_TIMEOUT_SECS: u64 = 10;
/// Per-read socket timeout for the SSE stream. Overridable via AACODE_LLM_READ_TIMEOUT.
const DEFAULT_READ_TIMEOUT_SECS: u64 = 30;
const WRITE_TIMEOUT_SECS: u64 = 30;

/// Builds the shared HTTP agent with bounded connect/read/write/total timeouts.
pub(crate) fn llm_agent(request_timeout_secs: Option<u64>) -> ureq::Agent {
    let read_secs = std::env::var("AACODE_LLM_READ_TIMEOUT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_READ_TIMEOUT_SECS);
    let total_secs = request_timeout_secs
        .or_else(|| std::env::var("AACODE_LLM_REQUEST_TIMEOUT").ok().and_then(|v| v.parse().ok()))
        .unwrap_or(300);
    ureq::builder()
        .timeout_connect(Duration::from_secs(CONNECT_TIMEOUT_SECS))
        .timeout_read(Duration::from_secs(read_secs))
        .timeout_write(Duration::from_secs(WRITE_TIMEOUT_SECS))
        .timeout(Duration::from_secs(total_secs))
        .build()
}

pub struct OpenAiClient {
    model: ModelConfig,
    agent: ureq::Agent,
}

impl OpenAiClient {
    pub fn new(model: ModelConfig) -> Self {
        OpenAiClient {
            agent: llm_agent(model.request_timeout_secs),
            model,
        }
    }

    fn endpoint(&self) -> String {
        let base = self.model.resolved_base_url();
        format!("{}/chat/completions", base.trim_end_matches('/'))
    }

    /// Convert internal ChatMessages to OpenAI JSON message objects.
    fn build_messages(messages: &[ChatMessage]) -> Vec<Value> {
        let mut out = Vec::with_capacity(messages.len());
        for m in messages {
            let mut obj = json!({ "role": m.role });
            let map = obj.as_object_mut().unwrap();
            // tool messages need tool_call_id; assistant tool_calls carried through.
            if let Some(tcs) = &m.tool_calls {
                let arr: Vec<Value> = tcs
                    .iter()
                    .map(|tc| {
                        json!({
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments}
                        })
                    })
                    .collect();
                map.insert("tool_calls".to_string(), Value::Array(arr));
                // assistant content may be empty string alongside tool_calls
                map.insert("content".to_string(), Value::String(m.content.clone()));
            } else {
                map.insert("content".to_string(), Value::String(m.content.clone()));
            }
            if let Some(id) = &m.tool_call_id {
                map.insert("tool_call_id".to_string(), Value::String(id.clone()));
            }
            if let Some(rc) = &m.reasoning_content {
                map.insert(
                    "reasoning_content".to_string(),
                    Value::String(rc.clone()),
                );
            }
            out.push(obj);
        }
        out
    }

    fn build_body(&self, messages: &[ChatMessage], tools: &[Value], stream: bool) -> Value {
        // Kimi models only accept temperature = 1.
        let m = self.model.name.to_lowercase();
        let temperature = if m.contains("kimi") || m.contains("moonshot") {
            1.0
        } else {
            self.model.temperature
        };
        let mut body = json!({
            "model": self.model.name,
            "messages": Self::build_messages(messages),
            "temperature": temperature,
            "max_tokens": self.model.max_tokens,
            "stream": stream,
        });
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools.to_vec());
            body["tool_choice"] = Value::String("auto".to_string());
        }
        body
    }

    fn api_key(&self) -> Result<String> {
        self.model
            .api_key
            .clone()
            .filter(|k| !k.trim().is_empty())
            .ok_or_else(|| AacodeError::Config("API key not configured".into()))
    }
}

/// Accumulator for one streamed tool_call (fragments arrive by index).
#[derive(Default, Clone)]
struct ToolAcc {
    id: String,
    name: String,
    arguments: String,
    name_announced: bool,
    last_report: usize,
}

impl LlmClient for OpenAiClient {
    fn chat_stream(
        &self,
        messages: &[ChatMessage],
        tools: &[Value],
        emitter: &dyn EventSink,
        cancel: &AtomicBool,
    ) -> Result<LlmResponse> {
        let api_key = self.api_key()?;
        let body = self.build_body(messages, tools, true);

        let resp = self
            .agent
            .post(&self.endpoint())
            .set("Authorization", &format!("Bearer {api_key}"))
            .set("Content-Type", "application/json")
            .set("Accept", "text/event-stream")
            .send_json(body);

        let reader = match resp {
            Ok(r) => r.into_reader(),
            Err(ureq::Error::Status(code, r)) => {
                let msg = r.into_string().unwrap_or_default();
                return Err(AacodeError::Api(format!("HTTP {code}: {}", truncate(&msg, 500))));
            }
            Err(e) => return Err(AacodeError::Network(e.to_string())),
        };

        parse_openai_stream(reader, emitter, cancel)
    }

    fn validate(&self) -> Result<()> {
        let api_key = self.api_key()?;
        let body = json!({
            "model": self.model.name,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 4,
            "stream": false,
        });
        match self
            .agent
            .post(&self.endpoint())
            .set("Authorization", &format!("Bearer {api_key}"))
            .set("Content-Type", "application/json")
            .send_json(body)
        {
            Ok(_) => Ok(()),
            Err(ureq::Error::Status(code, r)) => {
                let msg = r.into_string().unwrap_or_default();
                Err(AacodeError::Api(format!("HTTP {code}: {}", truncate(&msg, 300))))
            }
            Err(e) => Err(AacodeError::Network(e.to_string())),
        }
    }
}

/// Parse an OpenAI-style SSE chat stream from any reader. Extracted so it can
/// be unit-tested against recorded byte streams.
pub fn parse_openai_stream<R: std::io::Read>(
    reader: R,
    emitter: &dyn EventSink,
    cancel: &AtomicBool,
) -> Result<LlmResponse> {
        let mut sse = SseReader::new(reader);
        let mut text = String::new();
        let mut reasoning = String::new();
        let mut tool_accs: BTreeMap<i64, ToolAcc> = BTreeMap::new();
        let mut finish_reason: Option<String> = None;
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
            let chunk: Value = match serde_json::from_str(&payload) {
                Ok(v) => { valid_chunks += 1; v }
                Err(_) => continue, // skip malformed keep-alive fragments
            };
            // In-stream error: some providers send {"error": {...}} mid-stream
            // (HTTP 200 but a streamed failure). Surface it instead of silently
            // dropping the chunk. (Learned from rust-genai.)
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
                None => continue,
            };
            if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                finish_reason = Some(fr.to_string());
            }
            let delta = match choice.get("delta") {
                Some(d) => d,
                None => continue,
            };

            // reasoning_content (thinking). Some providers (Ollama, some proxies)
            // use `reasoning` instead of `reasoning_content`; accept both.
            let rc_opt = delta
                .get("reasoning_content")
                .and_then(|v| v.as_str())
                .or_else(|| delta.get("reasoning").and_then(|v| v.as_str()));
            if let Some(rc) = rc_opt {
                if !rc.is_empty() {
                    reasoning.push_str(rc);
                    emitter.delta("thinking", rc);
                }
            }
            // content (visible thought)
            if let Some(c) = delta.get("content").and_then(|v| v.as_str()) {
                if !c.is_empty() {
                    text.push_str(c);
                    emitter.delta("thought", c);
                }
            }
            // tool_calls fragments
            if let Some(tcs) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                for tc in tcs {
                    let idx = tc.get("index").and_then(|v| v.as_i64()).unwrap_or(0);
                    let acc = tool_accs.entry(idx).or_default();
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
        }

        // If the stream produced SSE payloads but NONE were parseable JSON,
        // the connection is producing garbage — don't silently return an empty
        // response (which the ReAct loop misinterprets as "completed").
        if total_payloads > 0 && valid_chunks == 0 {
            return Err(AacodeError::Api(
                "stream returned no parseable data (all chunks malformed)".into(),
            ));
        }

        // Emit final segments at stream completion (same as Anthropic).
        // Deferring seg to the end avoids mid-stream content "rollback" when
        // a first attempt leaks partial output before a retry.
        if !reasoning.is_empty() {
            emitter.seg("thinking", &reasoning);
        }
        emitter.seg("thought", &text);

        // Assemble tool_calls in index order.
        let mut tool_calls = Vec::new();
        for (i, (_, acc)) in tool_accs.into_iter().enumerate() {
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

        // Truncation warning appended to text (mirrors Python).
        if matches!(finish_reason.as_deref(), Some("length")) {
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
            finish_reason,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_is_char_boundary_safe() {
        // Chinese error messages must not panic (regression: byte slicing).
        let s = "错误：认证失败，请检查你的密钥配置是否正确无误";
        let t = truncate(s, 5);
        assert!(t.ends_with("..."));
        assert_eq!(t.chars().count(), 8); // 5 + "..."
        // short strings returned as-is
        assert_eq!(truncate("hi", 10), "hi");
        // emoji safe
        assert!(!truncate("🚀🚀🚀🚀🚀", 2).is_empty());
    }

    fn client() -> OpenAiClient {
        OpenAiClient::new(ModelConfig {
            name: "deepseek-chat".into(),
            api_key: Some("sk-x".into()),
            ..Default::default()
        })
    }

    #[test]
    fn endpoint_built() {
        assert_eq!(
            client().endpoint(),
            "https://api.deepseek.com/v1/chat/completions"
        );
    }

    #[test]
    fn build_messages_includes_tool_calls() {
        let msgs = vec![
            ChatMessage::user("hi"),
            ChatMessage::assistant_with_tools(
                "",
                vec![ToolCall {
                    id: "c1".into(),
                    name: "run_shell".into(),
                    arguments: "{\"command\":\"ls\"}".into(),
                }],
            ),
            ChatMessage::tool_result("c1", "ok"),
        ];
        let built = OpenAiClient::build_messages(&msgs);
        assert_eq!(built[0]["role"], "user");
        assert_eq!(built[1]["tool_calls"][0]["function"]["name"], "run_shell");
        assert_eq!(built[2]["tool_call_id"], "c1");
    }

    /// A full multi-turn conversation with tool use must serialize into the
    /// exact wire shape the OpenAI API requires: assistant message with
    /// tool_calls[], then a `role:"tool"` message with matching tool_call_id,
    /// then the next assistant turn. This is where multi-turn format bugs hide.
    #[test]
    fn build_messages_full_tool_roundtrip_wire_shape() {
        let msgs = vec![
            ChatMessage::system("you are helpful"),
            ChatMessage::user("list files"),
            ChatMessage::assistant_with_tools(
                "I'll list them.",
                vec![ToolCall {
                    id: "call_1".into(),
                    name: "run_shell".into(),
                    arguments: "{\"command\":\"ls\"}".into(),
                }],
            ),
            ChatMessage::tool_result("call_1", "a.txt\nb.txt"),
            ChatMessage::user("now count them"),
        ];
        let built = OpenAiClient::build_messages(&msgs);

        // system
        assert_eq!(built[0]["role"], "system");
        assert_eq!(built[0]["content"], "you are helpful");
        // user
        assert_eq!(built[1]["role"], "user");
        // assistant with tool_calls: must have content (may be text) + tool_calls array
        assert_eq!(built[2]["role"], "assistant");
        assert_eq!(built[2]["content"], "I'll list them.");
        assert_eq!(built[2]["tool_calls"][0]["id"], "call_1");
        assert_eq!(built[2]["tool_calls"][0]["type"], "function");
        assert_eq!(built[2]["tool_calls"][0]["function"]["name"], "run_shell");
        assert_eq!(
            built[2]["tool_calls"][0]["function"]["arguments"],
            "{\"command\":\"ls\"}"
        );
        // tool result: role=tool + tool_call_id linking back
        assert_eq!(built[3]["role"], "tool");
        assert_eq!(built[3]["tool_call_id"], "call_1");
        assert_eq!(built[3]["content"], "a.txt\nb.txt");
        // next user turn
        assert_eq!(built[4]["role"], "user");
        // whole thing must be valid serializable JSON
        assert!(serde_json::to_string(&built).is_ok());
    }

    /// reasoning_content on an assistant history message must be carried through
    /// (Kimi/DeepSeek require consistency).
    #[test]
    fn build_messages_carries_reasoning_content() {
        let mut m = ChatMessage::assistant("answer");
        m.reasoning_content = Some("my thoughts".into());
        let built = OpenAiClient::build_messages(&[m]);
        assert_eq!(built[0]["reasoning_content"], "my thoughts");
    }

    #[test]
    fn body_omits_tools_when_empty() {
        let b = client().build_body(&[ChatMessage::user("x")], &[], true);
        assert!(b.get("tools").is_none());
        assert_eq!(b["stream"], true);
    }

    #[test]
    fn kimi_forces_temp_1() {
        let c = OpenAiClient::new(ModelConfig {
            name: "kimi-k2".into(),
            api_key: Some("x".into()),
            temperature: 0.1,
            ..Default::default()
        });
        let b = c.build_body(&[ChatMessage::user("x")], &[], false);
        assert_eq!(b["temperature"], 1.0);
    }

    // ---- stream parsing (recorded SSE) ----

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
        // parsed args
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

    /// Regression: when the SSE stream drops mid-response (socket timeout /
    /// connection reset), the parser must return an Err — not silently
    /// return Ok with empty tool_calls.
    #[test]
    fn sse_read_error_propagates_as_stream_error() {
        struct BrokenReader;
        impl std::io::Read for BrokenReader {
            fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "simulated timeout"))
            }
        }
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let r = parse_openai_stream(BrokenReader, &sink, &cancel);
        assert!(r.is_err(), "SSE read error must propagate, not silently return Ok");
    }

    /// A partial stream that has begun but is cut off before any tool_calls:
    /// the parser must return an error so the caller can distinguish
    /// "connection lost" from "model finished".
    #[test]
    fn mid_stream_disconnect_not_empty_tool_calls() {
        // Simulate: first chunk arrives, second read times out.
        struct DropAfterFirst { sent: bool }
        impl std::io::Read for DropAfterFirst {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                if self.sent {
                    Err(std::io::Error::new(std::io::ErrorKind::ConnectionReset, "reset"))
                } else {
                    self.sent = true;
                    let data = b"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n";
                    let n = data.len().min(buf.len());
                    buf[..n].copy_from_slice(&data[..n]);
                    Ok(n)
                }
            }
        }
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let r = parse_openai_stream(DropAfterFirst { sent: false }, &sink, &cancel);
        assert!(r.is_err(), "mid-stream disconnect must error, not return empty tool_calls");
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
        assert!(r.is_err(), "all-malformed chunks must produce an error");
        let msg = format!("{}", r.err().unwrap());
        assert!(msg.contains("malformed") || msg.contains("parseable"));
    }

    // ── deferred thinking seg tests ─────────────────────────────────

    #[test]
    fn thinking_seg_emitted_at_stream_end() {
        // Reasoning deltas come first, then content. The seg("thinking")
        // must appear AFTER all deltas (at stream completion), not mid-stream.
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
        // Find the index of each event type to verify ordering.
        let first_delta = lines
            .iter()
            .position(|l| l.contains(r#""type":"delta""#))
            .expect("delta missing");
        let thinking_seg_pos = lines
            .iter()
            .position(|l| l.contains(r#""type":"seg_content""#) && l.contains(r#""seg":"thinking""#))
            .expect("thinking seg missing");
        let thought_seg_pos = lines
            .iter()
            .position(|l| l.contains(r#""type":"seg_content""#) && l.contains(r#""seg":"thought""#))
            .expect("thought seg missing");

        // seg("thinking") must come after all deltas and before seg("thought").
        assert!(
            thinking_seg_pos > first_delta,
            "thinking seg must come after deltas, not mid-stream"
        );
        assert!(
            thinking_seg_pos < thought_seg_pos,
            "thinking seg must come before thought seg"
        );
    }

    #[test]
    fn reasoning_only_stream_emits_thinking_seg() {
        // Pure reasoning stream (no content deltas). seg("thinking") must
        // still be emitted at the end, and seg("thought") with empty text.
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
        assert!(
            lines.iter().any(|l| l.contains(r#""seg":"thinking""#)),
            "reasoning-only stream must emit thinking seg: {lines:?}"
        );
        assert!(
            lines.iter().any(|l| l.contains(r#""seg":"thought""#)),
            "reasoning-only stream must emit thought seg: {lines:?}"
        );
    }

    #[test]
    fn content_only_stream_no_thinking_seg() {
        // No reasoning deltas — seg("thinking") must NOT appear.
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
        assert!(
            !lines.iter().any(|l| l.contains(r#""seg":"thinking""#)),
            "content-only stream must not emit thinking seg: {lines:?}"
        );
        assert!(
            lines.iter().any(|l| l.contains(r#""seg":"thought""#)),
            "content-only stream must still emit thought seg"
        );
    }

    #[test]
    fn mid_stream_failure_does_not_emit_thinking_seg() {
        // A stream that delivers reasoning deltas but then fails (SSE read
        // error) before completing. No seg("thinking") should leak to the
        // client because the stream never finished successfully.
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
                // First two reads deliver reasoning content.
                let data = b"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"partial\"}}]}\n\n";
                let n = data.len().min(buf.len());
                buf[..n].copy_from_slice(&data[..n]);
                Ok(n)
            }
        }
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let r = parse_openai_stream(FailAfterTwoReads { reads: 0 }, &sink, &cancel);
        assert!(r.is_err(), "mid-stream failure must return error");

        let lines = sink.lines();
        // Deltas may have leaked (first two reads), but seg_content("thinking")
        // must NOT appear because the stream never completed.
        let has_thinking_delta = lines
            .iter()
            .any(|l| l.contains(r#""type":"delta""#) && l.contains(r#""seg":"thinking""#));
        assert!(
            has_thinking_delta,
            "thinking deltas should have reached the sink"
        );
        assert!(
            !lines.iter().any(|l| {
                l.contains(r#""type":"seg_content""#) && l.contains(r#""seg":"thinking""#)
            }),
            "failed stream must not emit thinking seg: {lines:?}"
        );
    }

    #[test]
    fn thinking_seg_content_matches_accumulated_deltas() {
        // The final seg("thinking") content must equal the sum of all
        // delta("thinking") content.
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
        // Find the seg_content line (not delta) for thinking.
        let thinking_seg_line = lines
            .iter()
            .find(|l| {
                l.contains(r#""type":"seg_content""#) && l.contains(r#""seg":"thinking""#)
            })
            .expect("thinking seg missing");
        let seg_json: serde_json::Value =
            serde_json::from_str(thinking_seg_line).expect("invalid JSON");
        let seg_content = seg_json["content"].as_str().unwrap();

        // Extract and concatenate all delta("thinking") content.
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

        assert_eq!(
            seg_content, accumulated_deltas,
            "seg('thinking') content must match sum of all delta('thinking') content"
        );
    }

    #[test]
    fn interleaved_reasoning_content_thinking_seg_still_at_end() {
        // Real-world scenario: reasoning/content chunks interleave.
        // seg("thinking") must still appear at stream completion, not inline.
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
        // Count how many seg_content("thinking") lines appear — must be exactly 1.
        let thinking_seg_count = lines
            .iter()
            .filter(|l| {
                l.contains(r#""type":"seg_content""#) && l.contains(r#""seg":"thinking""#)
            })
            .count();
        assert_eq!(
            thinking_seg_count, 1,
            "exactly one thinking seg at end, got {thinking_seg_count}: {lines:?}"
        );
        // Count seg_content("thought") lines — must be exactly 1.
        let thought_seg_count = lines
            .iter()
            .filter(|l| {
                l.contains(r#""type":"seg_content""#) && l.contains(r#""seg":"thought""#)
            })
            .count();
        assert_eq!(
            thought_seg_count, 1,
            "exactly one thought seg at end, got {thought_seg_count}: {lines:?}"
        );
    }
}

