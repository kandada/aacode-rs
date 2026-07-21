// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Shared LLM types: chat messages, tool calls, response envelope.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One conversation message. Mirrors the OpenAI chat message shape plus the
/// optional `reasoning_content` used by DeepSeek/Kimi thinking models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String, // system | user | assistant | tool
    #[serde(default)]
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub reasoning_content: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self::simple("system", content)
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self::simple("user", content)
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::simple("assistant", content)
    }
    pub fn simple(role: &str, content: impl Into<String>) -> Self {
        ChatMessage {
            role: role.to_string(),
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }
    /// A tool result message answering a specific tool_call id.
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        ChatMessage {
            role: "tool".to_string(),
            content: content.into(),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            reasoning_content: None,
        }
    }
    /// An assistant message that carries tool_calls.
    pub fn assistant_with_tools(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        ChatMessage {
            role: "assistant".to_string(),
            content: content.into(),
            tool_calls: Some(tool_calls),
            tool_call_id: None,
            reasoning_content: None,
        }
    }
}

/// A single tool/function call requested by the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Raw JSON arguments string (as returned by the API). Parse before use.
    pub arguments: String,
}

impl ToolCall {
    /// Parse the arguments string into a JSON value. Returns an empty object
    /// on parse failure (mirrors the Python fallback of `{}`).
    pub fn parsed_args(&self) -> Value {
        serde_json::from_str(&self.arguments).unwrap_or_else(|_| serde_json::json!({}))
    }
}

/// The result of one streamed model call.
#[derive(Debug, Clone, Default)]
pub struct LlmResponse {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub reasoning_content: Option<String>,
    /// "length"/"max_tokens" indicates truncation.
    pub finish_reason: Option<String>,
}

impl LlmResponse {
    pub fn is_truncated(&self) -> bool {
        matches!(self.finish_reason.as_deref(), Some("length") | Some("max_tokens"))
    }
}
