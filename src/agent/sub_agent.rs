// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! SubAgent — a focused agent (code/test/research) that runs its own ReAct
//! loop with a specialized system prompt and a shared tool registry.
//!
//! Ported from Python `core/sub_agent.py` + `multi_agent.py`. Runs to
//! completion in-process and returns the final text (used by `delegate_task`).

use crate::agent::prompts::sub_agent_prompt;
use crate::agent::react_loop::{ReactLoop, RunStatus};
use crate::config::{AgentConfig, Gateway};
use crate::error::Result;
use crate::llm::types::ChatMessage;
use crate::llm::LlmClient;
use crate::session::SessionManager;
use crate::stream::EventSink;
use crate::tools::registry::ToolRegistry;
use std::path::Path;
use std::sync::atomic::AtomicBool;

/// Run a delegated sub-task to completion and return its final summary text.
pub fn run_sub_agent(
    agent_type: &str,
    task: &str,
    llm: &dyn LlmClient,
    registry: &ToolRegistry,
    config: &AgentConfig,
    project_path: &Path,
    emitter: &dyn EventSink,
    cancel: &AtomicBool,
) -> Result<String> {
    let system_prompt = sub_agent_prompt(agent_type);
    let native = match config.model.gateway {
        Gateway::Anthropic => registry.anthropic_tools(),
        Gateway::Openai => registry.openai_tools(),
    };

    // Sub-agents get their own ephemeral session (isolated context).
    let mut sub_session = SessionManager::new(project_path);
    sub_session.create_session(task, Some(&format!("sub:{agent_type}")))?;

    let messages = vec![
        ChatMessage::system(system_prompt),
        ChatMessage::user(format!(
            "Task: {task}\n\nFocus on completing this specific task. Output a final text summary (no tool calls) when done."
        )),
    ];

    // Sub-agents use fewer iterations by default.
    let mut sub_cfg = config.clone();
    sub_cfg.max_iterations = config.max_iterations.min(15);

    let loop_ = ReactLoop::new(llm, registry, &sub_cfg, native);
    let result = loop_.run(messages, &mut sub_session, emitter, cancel)?;

    match result.status {
        RunStatus::Completed => Ok(result.final_text),
        RunStatus::MaxIterations => Ok(format!(
            "[sub-agent reached max iterations]\n{}",
            result.final_text
        )),
        RunStatus::Cancelled => Ok("[sub-agent cancelled]".to_string()),
        RunStatus::Error(e) => Ok(format!("[sub-agent error: {e}]")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::LlmResponse;
    use crate::stream::CollectingSink;
    use serde_json::Value;
    use std::sync::Mutex;

    struct OneShotLlm {
        text: Mutex<Option<String>>,
    }
    impl LlmClient for OneShotLlm {
        fn chat_stream(
            &self,
            _m: &[ChatMessage],
            _t: &[Value],
            _e: &dyn EventSink,
            _c: &AtomicBool,
        ) -> Result<LlmResponse> {
            let t = self.text.lock().unwrap().take().unwrap_or_default();
            Ok(LlmResponse {
                text: t,
                ..Default::default()
            })
        }
        fn validate(&self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn sub_agent_returns_summary() {
        let llm = OneShotLlm {
            text: Mutex::new(Some("sub done".into())),
        };
        let reg = ToolRegistry::new();
        let cfg = AgentConfig::default();
        let dir = std::env::temp_dir().join(format!(
            "aacode_sub_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let out =
            run_sub_agent("code", "do x", &llm, &reg, &cfg, &dir, &sink, &cancel).unwrap();
        assert_eq!(out, "sub done");
    }
}
