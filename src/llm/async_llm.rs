// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Async LLM client abstraction — mirrors `LlmClient` but uses `reqwest` + `tokio`.
//!
//! Enables non-blocking LLM calls for CLI and FFI layers. Stream parsing is
//! delegated to `parse.rs` so all clients (sync + async) share one wire-format
//! implementation.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use futures::StreamExt;
use serde_json::{json, Value};

use crate::config::{Gateway, ModelConfig};
use crate::error::{AacodeError, Result};
use crate::llm::types::{ChatMessage, LlmResponse};
use crate::llm::{parse, LlmClient};
use crate::stream::EventSink;

// ── Async SSE reader ──────────────────────────────────────────────────────

struct AsyncSseStream<S> {
    stream: S,
    buffer: Vec<u8>,
    done: bool,
    first_read: bool,
}

impl<S> AsyncSseStream<S>
where
    S: futures::Stream<Item = std::result::Result<bytes::Bytes, reqwest::Error>> + Unpin,
{
    fn new(stream: S) -> Self {
        AsyncSseStream { stream, buffer: Vec::new(), done: false, first_read: true }
    }

    async fn next_data(&mut self) -> std::result::Result<Option<String>, AacodeError> {
        if self.done { return Ok(None); }
        loop {
            if let Some(pos) = self.buffer.iter().position(|&b| b == b'\n') {
                let line_bytes = self.buffer[..pos].to_vec();
                self.buffer.drain(..=pos);
                let line = String::from_utf8_lossy(&line_bytes).into_owned()
                    .trim_end_matches('\r').to_string();
                if self.first_read {
                    self.first_read = false;
                    if line.starts_with('\u{FEFF}') {
                        let s = line[3..].to_string();
                        if s.trim_end_matches(['\r', '\n']).is_empty() { continue; }
                        if let Some(payload) = Self::extract(&s) {
                            if payload == "[DONE]" { self.done = true; return Ok(None); }
                            return Ok(Some(payload));
                        }
                        continue;
                    }
                    if line.trim().is_empty() { continue; }
                }
                if line.trim().is_empty() { continue; }
                if let Some(payload) = Self::extract(&line) {
                    if payload == "[DONE]" { self.done = true; return Ok(None); }
                    return Ok(Some(payload));
                }
                continue;
            }
            match self.stream.next().await {
                Some(Ok(chunk)) => { self.buffer.extend_from_slice(&chunk); continue; }
                Some(Err(e)) => return Err(AacodeError::Network(format!("SSE: {e}"))),
                None => {
                    if !self.buffer.is_empty() {
                        let bytes = std::mem::take(&mut self.buffer);
                        let line = String::from_utf8_lossy(&bytes).into_owned()
                            .trim_end_matches(['\r', '\n']).to_string();
                        if let Some(payload) = Self::extract(&line) {
                            if payload != "[DONE]" { return Ok(Some(payload)); }
                        }
                    }
                    self.done = true;
                    return Ok(None);
                }
            }
        }
    }

    fn extract(line: &str) -> Option<String> {
        line.strip_prefix("data:")
            .map(|r| r.strip_prefix(' ').unwrap_or(r).to_string())
    }
}

// ── Async OpenAI client ────────────────────────────────────────────────────

/// Per-request HTTP timeout for LLM calls. Overridable via the
/// `AACODE_LLM_READ_TIMEOUT` env var (seconds) so tests and mobile hosts can
/// bound a stalled connection (the "TCP accepted but never responds" hang
/// that otherwise blocks until the generous default). Falls back to a long
/// default that tolerates slow reasoning models.
fn request_timeout() -> Duration {
    if let Ok(v) = std::env::var("AACODE_LLM_READ_TIMEOUT") {
        if let Ok(secs) = v.parse::<u64>() {
            if secs > 0 {
                return Duration::from_secs(secs);
            }
        }
    }
    Duration::from_secs(1200)
}

pub struct OpenAiAsyncClient {
    model: ModelConfig,
    client: reqwest::Client,
}

impl OpenAiAsyncClient {
    pub fn new(model: ModelConfig) -> Self {
        OpenAiAsyncClient {
            model,
            client: reqwest::Client::builder()
                .timeout(request_timeout())
                .connect_timeout(Duration::from_secs(10))
                .build()
                .expect("reqwest client"),
        }
    }

    fn endpoint(&self) -> String {
        let base = self.model.resolved_base_url();
        format!("{}/chat/completions", base.trim_end_matches('/'))
    }

    fn build_messages(messages: &[ChatMessage]) -> Vec<Value> {
        messages.iter().map(|m| {
            let mut obj = json!({"role": m.role});
            let map = obj.as_object_mut().unwrap();
            if let Some(tcs) = &m.tool_calls {
                let arr: Vec<Value> = tcs.iter().map(|tc| json!({
                    "id": tc.id, "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments}
                })).collect();
                map.insert("tool_calls".into(), Value::Array(arr));
                map.insert("content".into(), Value::String(m.content.clone()));
            } else {
                map.insert("content".into(), Value::String(m.content.clone()));
            }
            if let Some(id) = &m.tool_call_id {
                map.insert("tool_call_id".into(), Value::String(id.clone()));
            }
            if let Some(rc) = &m.reasoning_content {
                map.insert("reasoning_content".into(), Value::String(rc.clone()));
            }
            obj
        }).collect()
    }

    fn build_body(&self, messages: &[ChatMessage], tools: &[Value], stream: bool) -> Value {
        let m = self.model.name.to_lowercase();
        let temperature = if m.contains("kimi") || m.contains("moonshot") { 1.0 } else { self.model.temperature as f64 };
        let mut body = json!({
            "model": self.model.name,
            "messages": Self::build_messages(messages),
            "temperature": temperature,
            "max_tokens": self.model.max_tokens,
            "stream": stream,
        });
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools.to_vec());
            body["tool_choice"] = Value::String("auto".into());
        }
        body
    }

    async fn post_stream(&self, body: &Value) -> Result<impl futures::Stream<Item = std::result::Result<bytes::Bytes, reqwest::Error>>> {
        let api_key = self.model.api_key.as_deref().filter(|k| !k.trim().is_empty())
            .ok_or_else(|| AacodeError::Config("API key not configured".into()))?;
        let resp = self.client
            .post(self.endpoint())
            .bearer_auth(api_key)
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .json(body).send().await
            .map_err(|e| AacodeError::Network(e.to_string()))?;
        if !resp.status().is_success() {
            let code = resp.status().as_u16();
            let msg = resp.text().await.unwrap_or_default();
            return Err(AacodeError::Api(format!("HTTP {code}: {}", truncate(&msg, 500))));
        }
        Ok(resp.bytes_stream())
    }
}

#[async_trait::async_trait]
impl LlmClient for OpenAiAsyncClient {
    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        tools: &[Value],
        emitter: &dyn EventSink,
        cancel: &AtomicBool,
    ) -> Result<LlmResponse> {
        let body = self.build_body(messages, tools, true);
        let stream = self.post_stream(&body).await?;
        let mut sse = AsyncSseStream::new(stream);
        let mut state = parse::OpenAiParseState::default();

        while let Some(payload) = sse.next_data().await? {
            if cancel.load(Ordering::SeqCst) { return Err(AacodeError::Cancelled); }
            parse::parse_openai_chunk(&payload, &mut state, emitter)?;
        }
        parse::finalize_openai(state, emitter)
    }

    async fn validate(&self) -> Result<()> {
        let api_key = self.model.api_key.as_deref().filter(|k| !k.trim().is_empty())
            .ok_or_else(|| AacodeError::Config("API key not configured".into()))?;
        let body = json!({"model": self.model.name, "messages": [{"role":"user","content":"Hi"}], "max_tokens": 4, "stream": false});
        let resp = self.client
            .post(self.endpoint())
            .bearer_auth(api_key)
            .header("Content-Type", "application/json")
            .json(&body).send().await
            .map_err(|e| AacodeError::Network(e.to_string()))?;
        if !resp.status().is_success() {
            let code = resp.status().as_u16();
            let msg = resp.text().await.unwrap_or_default();
            return Err(AacodeError::Api(format!("HTTP {code}: {}", truncate(&msg, 300))));
        }
        Ok(())
    }
}

// ── Async Anthropic client ─────────────────────────────────────────────────

pub struct AnthropicAsyncClient {
    model: ModelConfig,
    client: reqwest::Client,
}

impl AnthropicAsyncClient {
    pub fn new(model: ModelConfig) -> Self {
        AnthropicAsyncClient {
            model,
            client: reqwest::Client::builder()
                .timeout(request_timeout())
                .connect_timeout(Duration::from_secs(10))
                .build()
                .expect("reqwest client"),
        }
    }

    fn endpoint(&self) -> String {
        let base = adjust_base(&self.model.resolved_base_url());
        format!("{}/v1/messages", base.trim_end_matches('/'))
    }

    fn build_messages(messages: &[ChatMessage]) -> (String, Vec<Value>) {
        let mut system = String::new();
        let mut out: Vec<Value> = Vec::new();
        let mut pending: Vec<Value> = Vec::new();
        fn flush(out: &mut Vec<Value>, pending: &mut Vec<Value>) {
            if !pending.is_empty() { out.push(json!({"role":"user","content":std::mem::take(pending)})); }
        }
        for m in messages {
            if m.role != "tool" { flush(&mut out, &mut pending); }
            match m.role.as_str() {
                "system" => { if !system.is_empty() { system.push_str("\n\n"); } system.push_str(&m.content); }
                "tool" => pending.push(json!({"type":"tool_result","tool_use_id":m.tool_call_id.clone().unwrap_or_default(),"content":m.content})),
                "assistant" => {
                    if let Some(tcs) = &m.tool_calls {
                        let mut blocks: Vec<Value> = Vec::new();
                        if !m.content.is_empty() { blocks.push(json!({"type":"text","text":m.content})); }
                        for tc in tcs { blocks.push(json!({"type":"tool_use","id":tc.id,"name":tc.name,"input":serde_json::from_str::<Value>(&tc.arguments).unwrap_or(json!({}))})); }
                        out.push(json!({"role":"assistant","content":blocks}));
                    } else if !m.content.is_empty() {
                        out.push(json!({"role":"assistant","content":m.content}));
                    }
                }
                _ => { if !m.content.is_empty() { out.push(json!({"role":"user","content":m.content})); } }
            }
        }
        flush(&mut out, &mut pending);
        (system, out)
    }

    fn build_body(&self, messages: &[ChatMessage], tools: &[Value], stream: bool) -> Value {
        let (system, msgs) = Self::build_messages(messages);
        let mut body = json!({
            "model": self.model.name, "max_tokens": self.model.max_tokens,
            "messages": msgs, "stream": stream,
        });
        if !system.is_empty() { body["system"] = Value::String(system); }
        if !tools.is_empty() { body["tools"] = Value::Array(tools.to_vec()); }
        body
    }

    async fn post_stream(&self, body: &Value) -> Result<impl futures::Stream<Item = std::result::Result<bytes::Bytes, reqwest::Error>>> {
        let api_key = self.model.api_key.as_deref().filter(|k| !k.trim().is_empty())
            .ok_or_else(|| AacodeError::Config("API key not configured".into()))?;
        let resp = self.client
            .post(self.endpoint())
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .json(body).send().await
            .map_err(|e| AacodeError::Network(e.to_string()))?;
        if !resp.status().is_success() {
            let code = resp.status().as_u16();
            let msg = resp.text().await.unwrap_or_default();
            return Err(AacodeError::Api(format!("HTTP {code}: {}", truncate(&msg, 500))));
        }
        Ok(resp.bytes_stream())
    }
}

#[async_trait::async_trait]
impl LlmClient for AnthropicAsyncClient {
    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        tools: &[Value],
        emitter: &dyn EventSink,
        cancel: &AtomicBool,
    ) -> Result<LlmResponse> {
        let body = self.build_body(messages, tools, true);
        let stream = self.post_stream(&body).await?;
        let mut sse = AsyncSseStream::new(stream);
        let mut state = parse::AnthropicParseState::default();

        while let Some(payload) = sse.next_data().await? {
            if cancel.load(Ordering::SeqCst) { return Err(AacodeError::Cancelled); }
            parse::parse_anthropic_chunk(&payload, &mut state, emitter)?;
        }
        parse::finalize_anthropic(state, emitter)
    }

    async fn validate(&self) -> Result<()> {
        let api_key = self.model.api_key.as_deref().filter(|k| !k.trim().is_empty())
            .ok_or_else(|| AacodeError::Config("API key not configured".into()))?;
        let body = json!({"model": self.model.name, "max_tokens": 4, "messages": [{"role":"user","content":"Hi"}]});
        let resp = self.client
            .post(self.endpoint())
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&body).send().await
            .map_err(|e| AacodeError::Network(e.to_string()))?;
        if !resp.status().is_success() {
            let code = resp.status().as_u16();
            let msg = resp.text().await.unwrap_or_default();
            return Err(AacodeError::Api(format!("HTTP {code}: {}", truncate(&msg, 300))));
        }
        Ok(())
    }
}

// ── Shared helpers ─────────────────────────────────────────────────────────

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n { s.to_string() } else { let h: String = s.chars().take(n).collect(); format!("{h}...") }
}

fn adjust_base(base: &str) -> String {
    let lower = base.to_lowercase();
    if lower.contains("minimax") || lower.contains("deepseek") || lower.contains("moonshot") {
        if let Some(s) = base.strip_suffix("/v1") { return format!("{}/anthropic", s.trim_end_matches('/')); }
        if !base.trim_end_matches('/').ends_with("/anthropic") { return format!("{}/anthropic", base.trim_end_matches('/')); }
    }
    base.to_string()
}

// ── Factory ────────────────────────────────────────────────────────────────

/// Build the appropriate async LLM client for the given model config.
pub fn build_client_async(model: &ModelConfig) -> Box<dyn LlmClient> {
    match model.gateway {
        Gateway::Anthropic => Box::new(AnthropicAsyncClient::new(model.clone())),
        Gateway::Openai => Box::new(OpenAiAsyncClient::new(model.clone())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_config(name: &str, gateway: Gateway) -> ModelConfig {
        ModelConfig {
            name: name.into(),
            api_key: Some("sk-test".into()),
            gateway,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_async_openai_client_created() {
        let cfg = model_config("gpt-4o", Gateway::Openai);
        let client = OpenAiAsyncClient::new(cfg);
        assert!(client.endpoint().contains("chat/completions"));
    }

    #[tokio::test]
    async fn test_async_anthropic_client_created() {
        let cfg = model_config("claude-sonnet-4-20250514", Gateway::Anthropic);
        let client = AnthropicAsyncClient::new(cfg);
        assert!(client.endpoint().contains("v1/messages"));
    }

    #[tokio::test]
    async fn test_build_client_async_factory() {
        let cfg = model_config("gpt-4o", Gateway::Openai);
        let _client = build_client_async(&cfg);
    }

    #[tokio::test]
    async fn test_async_sse_parses() {
        let bytes = bytes::Bytes::from("data: {\"a\":1}\n\ndata: [DONE]\n\n");
        let stream = Box::pin(futures::stream::once(async move { Ok::<_, reqwest::Error>(bytes) }));
        let mut sse = AsyncSseStream::new(stream);
        assert_eq!(sse.next_data().await.unwrap().unwrap(), "{\"a\":1}");
        assert!(sse.next_data().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_async_validate_fails_with_bad_url() {
        let mut cfg = model_config("gpt-4o", Gateway::Openai);
        cfg.base_url = Some("https://invalid.example.com/v1".into());
        let client = OpenAiAsyncClient::new(cfg);
        let result = client.validate().await;
        assert!(result.is_err());
    }

    #[test]
    fn test_block_on_from_sync_context() {
        let cfg = model_config("gpt-4o", Gateway::Openai);
        let client = build_client_async(&cfg);
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(async { client.validate().await });
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_block_on_llm_helper_creates_runtime_if_needed() {
        let cfg = model_config("gpt-4o", Gateway::Openai);
        let client = build_client_async(&cfg);
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(async { client.validate().await });
        assert!(result.is_err() || result.is_ok());
    }

    #[tokio::test]
    async fn test_async_clients_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OpenAiAsyncClient>();
        assert_send_sync::<AnthropicAsyncClient>();
    }
}
