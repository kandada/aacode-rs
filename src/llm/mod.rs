// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

pub mod anthropic;
pub mod async_llm;
pub mod openai;
pub mod parse;
pub mod sse;
pub mod thinking;
pub mod types;

pub use types::{ChatMessage, LlmResponse, ToolCall};

use crate::config::ModelConfig;
use crate::error::Result;
use crate::stream::EventSink;
use std::sync::atomic::AtomicBool;

#[async_trait::async_trait]
pub trait LlmClient: Send + Sync {
    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        tools: &[serde_json::Value],
        emitter: &dyn EventSink,
        cancel: &AtomicBool,
    ) -> Result<LlmResponse>;

    async fn validate(&self) -> Result<()>;
}

pub fn build_client(model: &ModelConfig) -> Box<dyn LlmClient> {
    crate::llm::async_llm::build_client_async(model)
}
