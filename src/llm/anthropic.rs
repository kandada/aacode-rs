// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Anthropic-compatible streaming client (MiniMax / Claude / DeepSeek-anthropic).
//!
//! Anthropic SSE uses named events: `content_block_start`, `content_block_delta`
//! (`thinking_delta` / `text_delta` / `input_json_delta`), `content_block_stop`,
//! `message_delta` (carries stop_reason). tool_use blocks accumulate their
//! `input` JSON via `input_json_delta`.

use super::sse::SseReader;
use super::types::{ChatMessage, LlmResponse, ToolCall};
use super::LlmClient;
use crate::config::ModelConfig;
use crate::error::{AacodeError, Result};
use crate::stream::EventSink;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};

pub struct AnthropicClient {
    model: ModelConfig,
    agent: ureq::Agent,
}

impl AnthropicClient {
    pub fn new(model: ModelConfig) -> Self {
        AnthropicClient {
            agent: super::openai::llm_agent(model.request_timeout_secs),
            model,
        }
    }

    fn endpoint(&self) -> String {
        let base = adjust_anthropic_base(&self.model.resolved_base_url());
        format!("{}/v1/messages", base.trim_end_matches('/'))
    }

    fn api_key(&self) -> Result<String> {
        self.model
            .api_key
            .clone()
            .filter(|k| !k.trim().is_empty())
            .ok_or_else(|| AacodeError::Config("API key not configured".into()))
    }

    /// Split messages into (system_text, anthropic_messages).
    fn build_messages(messages: &[ChatMessage]) -> (String, Vec<Value>) {
        let mut system = String::new();
        let mut out: Vec<Value> = Vec::new();
        // Accumulator for consecutive tool results, which Anthropic requires to
        // be merged into a single user message with multiple tool_result blocks
        // (one per preceding tool_use). Emitting them as separate user messages
        // triggers: "`tool_use` ids were found without `tool_result` blocks
        // immediately after".
        let mut pending_tool_results: Vec<Value> = Vec::new();

        // Flush accumulated tool_result blocks as one user message.
        fn flush_tool_results(out: &mut Vec<Value>, pending: &mut Vec<Value>) {
            if !pending.is_empty() {
                out.push(json!({"role": "user", "content": std::mem::take(pending)}));
            }
        }

        for m in messages {
            // Any non-tool message ends a run of tool results.
            if m.role != "tool" {
                flush_tool_results(&mut out, &mut pending_tool_results);
            }
            match m.role.as_str() {
                "system" => {
                    if !system.is_empty() {
                        system.push_str("\n\n");
                    }
                    system.push_str(&m.content);
                }
                "tool" => {
                    // Accumulate; will be flushed into a single user message.
                    pending_tool_results.push(json!({
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id.clone().unwrap_or_default(),
                        "content": m.content,
                    }));
                }
                "assistant" => {
                    if let Some(tcs) = &m.tool_calls {
                        let mut blocks: Vec<Value> = Vec::new();
                        if !m.content.is_empty() {
                            blocks.push(json!({"type": "text", "text": m.content}));
                        }
                        for tc in tcs {
                            let input: Value = serde_json::from_str(&tc.arguments)
                                .unwrap_or_else(|_| json!({}));
                            blocks.push(json!({
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": input,
                            }));
                        }
                        out.push(json!({"role": "assistant", "content": blocks}));
                    } else if !m.content.is_empty() {
                        out.push(json!({"role": "assistant", "content": m.content}));
                    }
                }
                _ => {
                    // user
                    if !m.content.is_empty() {
                        out.push(json!({"role": "user", "content": m.content}));
                    }
                }
            }
        }
        // Flush any trailing tool results.
        flush_tool_results(&mut out, &mut pending_tool_results);
        (system, out)
    }

    fn build_body(&self, messages: &[ChatMessage], tools: &[Value], stream: bool) -> Value {
        let (system, msgs) = Self::build_messages(messages);
        let mut body = json!({
            "model": self.model.name,
            "max_tokens": self.model.max_tokens,
            "temperature": self.model.temperature,
            "messages": msgs,
            "stream": stream,
        });
        if !system.is_empty() {
            body["system"] = Value::String(system);
        }
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools.to_vec());
        }
        body
    }
}

/// Anthropic base URLs from OpenAI-style `/v1` suffix need adjustment for
/// providers whose anthropic endpoint lives under `/anthropic`.
fn adjust_anthropic_base(base: &str) -> String {
    let lower = base.to_lowercase();
    let provider_anthropic = lower.contains("minimax")
        || lower.contains("deepseek")
        || lower.contains("moonshot");
    if provider_anthropic {
        if let Some(stripped) = base.strip_suffix("/v1") {
            return format!("{}/anthropic", stripped.trim_end_matches('/'));
        }
        if !base.trim_end_matches('/').ends_with("/anthropic") {
            return format!("{}/anthropic", base.trim_end_matches('/'));
        }
    }
    base.to_string()
}

impl LlmClient for AnthropicClient {
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
            .set("x-api-key", &api_key)
            .set("anthropic-version", "2023-06-01")
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
        parse_anthropic_stream(reader, emitter, cancel)
    }

    fn validate(&self) -> Result<()> {
        let api_key = self.api_key()?;
        let body = json!({
            "model": self.model.name,
            "max_tokens": 4,
            "messages": [{"role": "user", "content": "Hi"}],
        });
        match self
            .agent
            .post(&self.endpoint())
            .set("x-api-key", &api_key)
            .set("anthropic-version", "2023-06-01")
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

#[derive(Default, Clone)]
struct BlockAcc {
    kind: String, // "text" | "thinking" | "tool_use"
    id: String,
    name: String,
    text: String,      // for text/thinking
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

    while let Some(payload) = sse.next_data() {
        if cancel.load(Ordering::SeqCst) {
            return Err(AacodeError::Cancelled);
        }
        let ev: Value = match serde_json::from_str(&payload) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let etype = ev.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match etype {
            // In-stream error event: {"type":"error","error":{"type":..,"message":..}}
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
                    acc.kind = b.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    acc.id = b.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    acc.name = b.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
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

    if !reasoning.is_empty() {
        emitter.seg("thinking", &reasoning);
    }
    emitter.seg("thought", &text);

    // Assemble tool calls.
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

fn truncate(s: &str, n: usize) -> String {
    // Char-boundary-safe truncation (byte slicing would panic on multibyte chars).
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let head: String = s.chars().take(n).collect();
        format!("{head}...")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::CollectingSink;
    use std::io::Cursor;

    #[test]
    fn base_url_adjustment() {
        assert_eq!(
            adjust_anthropic_base("https://api.deepseek.com/v1"),
            "https://api.deepseek.com/anthropic"
        );
        assert_eq!(
            adjust_anthropic_base("https://api.anthropic.com"),
            "https://api.anthropic.com"
        );
    }

    #[test]
    fn split_system_and_tool_result() {
        let msgs = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("hi"),
            ChatMessage::tool_result("c1", "res"),
        ];
        let (sys, out) = AnthropicClient::build_messages(&msgs);
        assert_eq!(sys, "sys");
        assert_eq!(out[0]["role"], "user");
        assert_eq!(out[1]["content"][0]["type"], "tool_result");
        assert_eq!(out[1]["content"][0]["tool_use_id"], "c1");
    }

    /// Full multi-turn tool round-trip into Anthropic wire format:
    /// - system → top-level `system` string
    /// - assistant tool_calls → content:[{type:text}, {type:tool_use}]
    /// - tool_result → user message content:[{type:tool_result, tool_use_id}]
    #[test]
    fn build_messages_full_tool_roundtrip_wire_shape() {
        let msgs = vec![
            ChatMessage::system("be terse"),
            ChatMessage::user("list files"),
            ChatMessage::assistant_with_tools(
                "Listing.",
                vec![ToolCall {
                    id: "toolu_1".into(),
                    name: "run_shell".into(),
                    arguments: "{\"command\":\"ls\"}".into(),
                }],
            ),
            ChatMessage::tool_result("toolu_1", "a.txt"),
            ChatMessage::user("count them"),
        ];
        let (system, out) = AnthropicClient::build_messages(&msgs);
        assert_eq!(system, "be terse");
        // user
        assert_eq!(out[0]["role"], "user");
        assert_eq!(out[0]["content"], "list files");
        // assistant: text block + tool_use block
        assert_eq!(out[1]["role"], "assistant");
        assert_eq!(out[1]["content"][0]["type"], "text");
        assert_eq!(out[1]["content"][0]["text"], "Listing.");
        assert_eq!(out[1]["content"][1]["type"], "tool_use");
        assert_eq!(out[1]["content"][1]["id"], "toolu_1");
        assert_eq!(out[1]["content"][1]["name"], "run_shell");
        // tool_use input must be a parsed OBJECT, not a string
        assert_eq!(out[1]["content"][1]["input"]["command"], "ls");
        // tool result as a user message
        assert_eq!(out[2]["role"], "user");
        assert_eq!(out[2]["content"][0]["type"], "tool_result");
        assert_eq!(out[2]["content"][0]["tool_use_id"], "toolu_1");
        // next user turn
        assert_eq!(out[3]["role"], "user");
        assert!(serde_json::to_string(&out).is_ok());
    }

    /// Multiple system messages must be concatenated into the top-level system.
    #[test]
    fn build_messages_concatenates_system() {
        let msgs = vec![
            ChatMessage::system("first"),
            ChatMessage::system("second"),
            ChatMessage::user("hi"),
        ];
        let (system, _) = AnthropicClient::build_messages(&msgs);
        assert!(system.contains("first"));
        assert!(system.contains("second"));
    }

    /// Parallel tool calls: one assistant message with 2 tool_use blocks must be
    /// followed by ONE user message containing BOTH tool_result blocks (not two
    /// separate user messages). Anthropic rejects the latter with HTTP 400.
    #[test]
    fn build_messages_coalesces_parallel_tool_results() {
        let msgs = vec![
            ChatMessage::user("do two things"),
            ChatMessage::assistant_with_tools(
                "",
                vec![
                    ToolCall { id: "a".into(), name: "run_shell".into(), arguments: "{\"command\":\"cat x\"}".into() },
                    ToolCall { id: "b".into(), name: "run_shell".into(), arguments: "{\"command\":\"python3 x\"}".into() },
                ],
            ),
            ChatMessage::tool_result("a", "content of x"),
            ChatMessage::tool_result("b", "ran x"),
            ChatMessage::user("thanks"),
        ];
        let (_, out) = AnthropicClient::build_messages(&msgs);
        // out[0]=user, out[1]=assistant(2 tool_use), out[2]=user(2 tool_result), out[3]=user
        assert_eq!(out[1]["content"].as_array().unwrap().len(), 2);
        let tool_result_msg = &out[2];
        assert_eq!(tool_result_msg["role"], "user");
        let blocks = tool_result_msg["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 2, "both tool_results must be in one user message");
        assert_eq!(blocks[0]["type"], "tool_result");
        assert_eq!(blocks[0]["tool_use_id"], "a");
        assert_eq!(blocks[1]["tool_use_id"], "b");
        // final user message is separate
        assert_eq!(out[3]["role"], "user");
        assert_eq!(out[3]["content"], "thanks");
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
}
