// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Shared, dependency-free helpers for the fixed-eval and prompt-injection
//! integration tests. These are *fixtures*: they exercise the real ReAct loop,
//! session persistence and tool dispatch without a network model.

#![allow(dead_code)]

use aacode_rs::error::Result;
use aacode_rs::llm::{ChatMessage, LlmClient, LlmResponse, ToolCall};
use aacode_rs::stream::EventSink;
use aacode_rs::tools::registry::Tool;
use aacode_rs::tools::schema::ToolSchema;
use async_trait::async_trait;
use serde_json::Value;
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;

pub type Log = std::sync::Arc<Mutex<Vec<String>>>;
pub fn new_log() -> Log {
    std::sync::Arc::new(Mutex::new(Vec::new()))
}

/// Scripted LLM: returns responses in order and records every request.
pub struct ScriptedLlm {
    pub responses: Mutex<Vec<LlmResponse>>,
    pub seen: Mutex<Vec<Vec<ChatMessage>>>,
}

impl ScriptedLlm {
    pub fn new(responses: Vec<LlmResponse>) -> Self {
        ScriptedLlm {
            responses: Mutex::new(responses),
            seen: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl LlmClient for ScriptedLlm {
    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        _tools: &[Value],
        _emitter: &dyn EventSink,
        _cancel: &AtomicBool,
    ) -> Result<LlmResponse> {
        self.seen.lock().unwrap().push(messages.to_vec());
        let mut queue = self.responses.lock().unwrap();
        if queue.is_empty() {
            return Err(aacode_rs::error::AacodeError::Api(
                "no more scripted responses".into(),
            ));
        }
        Ok(queue.remove(0))
    }

    async fn validate(&self) -> Result<()> {
        Ok(())
    }
}

/// A stub tool that logs its invocation and returns fixed output.
pub struct LogTool {
    pub name: &'static str,
    pub log: Log,
    pub output: String,
}

#[async_trait]
impl Tool for LogTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(self.name, "eval stub", vec![])
    }
    async fn call(&self, _args: &Value, _cancel: &AtomicBool) -> Result<String> {
        self.log.lock().unwrap().push(self.name.to_string());
        Ok(self.output.clone())
    }
}

pub fn tool_response(id: &str, name: &str, args: Value) -> LlmResponse {
    LlmResponse {
        tool_calls: vec![ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments: args.to_string(),
        }],
        ..Default::default()
    }
}

pub fn final_text(text: &str) -> LlmResponse {
    LlmResponse {
        text: text.to_string(),
        ..Default::default()
    }
}

pub fn temp_project(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "aacode_{tag}_{}_{}",
        std::process::id(),
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
