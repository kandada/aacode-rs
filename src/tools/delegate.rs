// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Delegation tool — delegate_task.
//!
//! `delegate_task` runs a specialized sub-agent (code/test/research) inline to
//! completion and returns its summary. The sub-agent uses a registry WITHOUT
//! delegation tools to prevent unbounded recursion.
//!
//! Ported from Python `core/main_agent.py` (delegate_task) + `core/sub_agent.py`.

use super::registry::{Tool, ToolRegistry};
use super::schema::{ParamType, ToolParameter, ToolSchema};
use crate::agent::sub_agent::run_sub_agent;
use crate::config::AgentConfig;
use crate::error::Result;
use crate::llm::LlmClient;
use crate::stream::CollectingSink;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Factory that builds a sub-agent registry (no delegation tools).
pub type SubRegistryFactory = Arc<dyn Fn() -> ToolRegistry + Send + Sync>;

pub struct DelegateTaskTool {
    pub llm: Arc<dyn LlmClient>,
    pub config: AgentConfig,
    pub project_path: PathBuf,
    pub sub_registry: SubRegistryFactory,
}

#[async_trait::async_trait]
impl Tool for DelegateTaskTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "delegate_task",
            "Delegate a task to a specialized sub-agent (agent_type: general|code|test|research). Runs to completion and returns its summary.",
            vec![
                ToolParameter::new("task_description", ParamType::String, true, "Task for the sub-agent", &["task", "description"]),
                ToolParameter::new("agent_type", ParamType::String, false, "general|code|test|research", &["type"]),
            ],
        )
    }

    async fn call(&self, args: &Value, cancel: &AtomicBool) -> Result<String> {
        let task = args
            .get("task_description")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if task.is_empty() {
            return Ok(json!({"success": false, "error": "missing task_description"}).to_string());
        }
        let agent_type = args.get("agent_type").and_then(|v| v.as_str()).unwrap_or("general");
        let registry = (self.sub_registry)();
        // Sub-agent output is collected but not streamed to the main sink to
        // avoid confusing the UI; the returned summary is what matters.
        let sink = CollectingSink::new(false);
        let summary = run_sub_agent(
            agent_type,
            task,
            self.llm.as_ref(),
            &registry,
            &self.config,
            &self.project_path,
            &sink,
            cancel,
        ).await?;
        Ok(json!({
            "success": true,
            "agent_type": agent_type,
            "result": summary,
        })
        .to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{ChatMessage, LlmResponse};
    use crate::stream::EventSink;
    use std::sync::Mutex;

    struct OneShot {
        text: Mutex<Option<String>>,
    }
    #[async_trait::async_trait]
    impl LlmClient for OneShot {
        async fn chat_stream(
            &self,
            _m: &[ChatMessage],
            _t: &[Value],
            _e: &dyn EventSink,
            _c: &AtomicBool,
        ) -> Result<LlmResponse> {
            Ok(LlmResponse {
                text: self.text.lock().unwrap().take().unwrap_or_default(),
                ..Default::default()
            })
        }
        async fn validate(&self) -> Result<()> {
            Ok(())
        }
    }

    fn tmp() -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "aacode_deleg_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn delegate_runs_sub_agent() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
        let llm: Arc<dyn LlmClient> = Arc::new(OneShot {
            text: Mutex::new(Some("sub result".into())),
        });
        let factory: SubRegistryFactory = Arc::new(|| ToolRegistry::new());
        let tool = DelegateTaskTool {
            llm,
            config: AgentConfig::default(),
            project_path: tmp(),
            sub_registry: factory,
        };
        let cancel = AtomicBool::new(false);
        let out = tool
            .call(&json!({"task_description": "do sub", "agent_type": "code"}), &cancel)
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["success"], true);
        assert_eq!(v["result"], "sub result");
        });
    }

    #[tokio::test]
    async fn delegate_missing_task() {
        let llm: Arc<dyn LlmClient> = Arc::new(OneShot {
            text: Mutex::new(None),
        });
        let factory: SubRegistryFactory = Arc::new(|| ToolRegistry::new());
        let tool = DelegateTaskTool {
            llm,
            config: AgentConfig::default(),
            project_path: tmp(),
            sub_registry: factory,
        };
        let cancel = AtomicBool::new(false);
        let out = tool.call(&json!({}), &cancel).await.unwrap();
        assert_eq!(serde_json::from_str::<Value>(&out).unwrap()["success"], false);
    }
}
