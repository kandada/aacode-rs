// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! LLM client abstraction + gateway dispatch.

pub mod anthropic;
pub mod openai;
pub mod sse;
pub mod types;

pub use types::{ChatMessage, LlmResponse, ToolCall};

use crate::config::{Gateway, ModelConfig};
use crate::error::Result;
use crate::stream::EventSink;
use std::sync::atomic::AtomicBool;

/// A streaming chat client. Implementations perform one request and stream
/// tokens/segments to the sink, returning the assembled response.
pub trait LlmClient: Send + Sync {
    fn chat_stream(
        &self,
        messages: &[ChatMessage],
        tools: &[serde_json::Value],
        emitter: &dyn EventSink,
        cancel: &AtomicBool,
    ) -> Result<LlmResponse>;

    /// Lightweight probe used by validate_api_key (non-streaming, tiny request).
    fn validate(&self) -> Result<()>;
}

/// Build the appropriate client for the given model config.
pub fn build_client(model: &ModelConfig) -> Box<dyn LlmClient> {
    // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0
    match model.gateway {
        Gateway::Anthropic => Box::new(anthropic::AnthropicClient::new(model.clone())),
        Gateway::Openai => Box::new(openai::OpenAiClient::new(model.clone())),
    }
}

/// Char-boundary-safe truncation (byte slicing would panic on multibyte
/// characters such as Chinese error messages).
pub(crate) fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let head: String = s.chars().take(n).collect();
        format!("{head}...")
    }
}
