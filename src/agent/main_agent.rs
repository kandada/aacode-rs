// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! MainAgent — assembles the system prompt, builds the tool registry, and
//! drives the ReAct loop for a task.
//!
//! Ported from Python `core/main_agent.py` (execute + tool assembly).

use crate::agent::prompts::{PLANNING_IN_THOUGHT, SYSTEM_PROMPT_FOR_MAIN_AGENT};
use crate::agent::react_loop::{ReactLoop, RunResult};
use crate::config::{AgentConfig, Gateway};
use crate::context::ContextManager;
use crate::error::Result;
use crate::llm::types::ChatMessage;
use crate::llm::{build_client, LlmClient};
use crate::session::{SessionManager, SessionMessage};
use crate::stream::EventSink;
use crate::tools::registry::ToolRegistry;
use crate::tools::ShellBackend;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

pub struct MainAgent {
    pub config: AgentConfig,
    project_path: PathBuf,
    backend: Arc<dyn ShellBackend>,
    llm: Box<dyn LlmClient>,
    context: ContextManager,
}

impl MainAgent {
    pub fn new(
        mut config: AgentConfig,
        project_path: PathBuf,
        backend: Arc<dyn ShellBackend>,
    ) -> Self {
        // Wire config.timeouts.model_request into the LLM agent's per-request deadline
        if config.model.request_timeout_secs.is_none() {
            config.model.request_timeout_secs = Some(config.timeouts.model_request);
        }
        let llm = build_client(&config.model);
        // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0
        let context = ContextManager::new(&project_path);
        MainAgent {
            config,
            project_path,
            backend,
            llm,
            context,
        }
    }

    /// Build the full tool registry with all concrete tools wired up,
    /// plus the session-id holder for todo store synchronization.
    pub fn build_registry_with_holder(
        &self,
    ) -> (
        ToolRegistry,
        std::sync::Arc<std::sync::Mutex<Option<String>>>,
    ) {
        crate::tools::build_default_registry_with_holder(
            self.backend.clone(),
            self.project_path.clone(),
            &self.config,
        )
    }

    /// Build the full tool registry with all concrete tools wired up.
    pub fn build_registry(&self) -> ToolRegistry {
        self.build_registry_with_holder().0
    }

    /// Assemble the fully-static system prompt (messages[0]).
    /// All content here is configuration that rarely changes — init.md
    /// is project-level rules, skills_list comes from the host, and
    /// Working Directory + PLANNING_IN_THOUGHT are anchored per session.
    /// Byte-identical across execute() calls → provider KV cache stays hot.
    fn build_system_prompt(&self, _registry: &ToolRegistry) -> String {
        let skills_user_dir = self
            .config
            .skills
            .user_dir
            .as_ref()
            .map(std::path::PathBuf::from);
        let skills_list = crate::tools::skills::skills_list_for_prompt(
            &self.project_path,
            skills_user_dir.as_deref(),
            &self.config.skills.extra_builtins,
            self.config.skills.vfs_skills_dir.as_deref(),
        );
        let mut prompt = SYSTEM_PROMPT_FOR_MAIN_AGENT.replace("{skills_list}", &skills_list);
        prompt.push_str(&format!(
            "\n\n## Working Directory\nYour current working directory is: {}\nFor project files, use paths relative to this directory.",
            self.project_path.display()
        ));
        if self.backend.kind() == "fastshell" {
            prompt.push_str(" Paths starting with / (e.g. /skills) are relative to the sandbox root, not the real filesystem. Paths without / are relative to your working directory. avoid /tmp/.");
        }
        // Project init instructions (static config, rarely changes).
        let init = self.context.load_init_instructions();
        prompt.push_str(&format!("\n\n## Project Init Instructions\n\n{init}"));
        // Planning guidance (static).
        prompt.push_str(PLANNING_IN_THOUGHT);
        // Plan-first mode: instruct the model to produce a plan before acting.
        if self.config.plan_first {
            prompt.push_str(
                "\n\nPlan-First Mode: Before taking any action, first produce a concise numbered plan of the steps you will take. Only then begin executing tools.\n",
            );
        }
        prompt
    }

    /// Native tools for the active gateway.
    fn native_tools(&self, registry: &ToolRegistry) -> Vec<serde_json::Value> {
        match self.config.model.gateway {
            Gateway::Anthropic => registry.anthropic_tools(),
            Gateway::Openai => registry.openai_tools(),
        }
    }

    /// Execute a task: create/continue a session, build messages, run the loop.
    pub fn execute(
        &self,
        task: &str,
        session: &mut SessionManager,
        emitter: &dyn EventSink,
        cancel: &AtomicBool,
    ) -> Result<RunResult> {
        emitter.started(task);
        // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0

        // Session: reuse current or create new.
        if session.current_session_id.is_none() {
            let id = session.create_session(task, None)?;
            emitter.session("session_created", &id);
        } else {
            let id = session.current_session_id.clone().unwrap();
            // Ensure the task is recorded as a user message for this turn.
            let already = session
                .messages
                .last()
                .map(|m| m.role == "user" && m.content == task)
                .unwrap_or(false);
            if !already {
                session.add_message(SessionMessage::from_chat(&ChatMessage::user(task)))?;
            }
            emitter.session("session_switched", &id);
        }

        let (registry, sid_holder) = self.build_registry_with_holder();
        // Sync the todo store with the current session.
        {
            let mut sid = sid_holder.lock().unwrap();
            *sid = session.current_session_id.clone();
        }
        let native = self.native_tools(&registry);
        let system_prompt = self.build_system_prompt(&registry);

        // Build the message list: system + prior history + current task.
        let mut messages = vec![ChatMessage::system(system_prompt)];

        // Prior history (excluding the just-added current task to avoid dup).
        let history = session.history_chat();
        // If ended mid-tool previously, inject a recovery hint.
        if session.ended_mid_tool() {
            messages.push(ChatMessage::system(
                "[SYSTEM] Previous turn was interrupted mid tool execution. Continue based on existing results; do not repeat prior tool calls.",
            ));
        }
        // Append history but drop the trailing duplicate current-task user msg.
        let mut hist = history;
        if let Some(last) = hist.last() {
            if last.role == "user" && last.content == task {
                hist.pop();
            }
        }
        messages.extend(hist);

        // Repair the tool_calls/tool pairing before anything reaches the API.
        // A session interrupted between persisting assistant(tool_calls) and
        // the tool results would otherwise poison every subsequent request
        // (HTTP 400 "insufficient tool messages following tool_calls").
        let repaired = crate::agent::sanitize::sanitize_history(&mut messages);
        if repaired > 0 {
            messages.push(ChatMessage::system(format!(
                "[SYSTEM] Repaired {repaired} interrupted tool-call record(s) from the previous turn. Do not repeat prior tool calls; continue from the recorded results."
            )));
        }

        // Current task as the final user turn.
        messages.push(ChatMessage::user(format!(
            "Task: {task}\n\nUse native function calls to execute tools. Output a final text summary (no tool calls) when done."
        )));

        let loop_ = ReactLoop::new(self.llm.as_ref(), &registry, &self.config, native);
        loop_.run(messages, session, emitter, cancel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::backend::{CmdOutput, NativeShell};

    fn make_agent() -> (MainAgent, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "aacode_main_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let backend: Arc<dyn ShellBackend> = Arc::new(NativeShell::new());
        let mut acfg = AgentConfig::default();
        acfg.model.api_key = Some("sk-test".into());
        (MainAgent::new(acfg, dir.clone(), backend), dir)
    }

    #[test]
    fn builds_registry_with_core_tools() {
        let (agent, _) = make_agent();
        let reg = agent.build_registry();
        assert!(reg.contains("run_shell"));
        assert!(reg.contains("add_todo_item"));
        assert!(reg.contains("search_web"));
    }

    #[test]
    fn system_prompt_has_static_content() {
        let (agent, _) = make_agent();
        let reg = agent.build_registry();
        let p = agent.build_system_prompt(&reg);
        assert!(p.contains("Working Directory"));
        assert!(!p.contains("{skills_list}"));
        // Static config belongs in messages[0].
        assert!(p.contains("Project Init Instructions"));
        // Dynamic project analysis is removed — not in messages[0], not appended.
    }

    #[test]
    fn native_tools_switch_by_gateway() {
        let (agent, _) = make_agent();
        let reg = agent.build_registry();
        let openai = agent.native_tools(&reg);
        assert!(openai[0].get("function").is_some());
    }

    #[test]
    fn plan_first_adds_instruction() {
        let (mut agent, _) = make_agent();
        agent.config.plan_first = true;
        let reg = agent.build_registry();
        let p = agent.build_system_prompt(&reg);
        assert!(p.contains("Plan-First Mode"));
    }

    struct MockFastshellBackend;
    impl ShellBackend for MockFastshellBackend {
        fn run(
            &self,
            _command: &str,
            _stdin_input: Option<&str>,
            _timeout_secs: u64,
            _idle_timeout_secs: u64,
            _cwd: &std::path::Path,
        ) -> CmdOutput {
            CmdOutput { stdout: String::new(), stderr: String::new(), exit_code: 0 }
        }
        fn kind(&self) -> &'static str { "fastshell" }
    }

    #[test]
    fn fastshell_backend_adds_sandbox_hint() {
        let dir = std::env::temp_dir().join(format!("aacode_shint_{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let backend: Arc<dyn ShellBackend> = Arc::new(MockFastshellBackend);
        let mut acfg = AgentConfig::default();
        acfg.model.api_key = Some("sk-test".into());
        let agent = MainAgent::new(acfg, dir, backend);
        let reg = agent.build_registry();
        let p = agent.build_system_prompt(&reg);
        assert!(p.contains("sandbox root"), "fastshell prompt must contain sandbox hint");
        assert!(p.contains("avoid /tmp/"));
    }

    #[test]
    fn native_backend_does_not_add_sandbox_hint() {
        let (agent, _) = make_agent();
        let reg = agent.build_registry();
        let p = agent.build_system_prompt(&reg);
        assert!(!p.contains("sandbox root"), "native prompt must not contain sandbox hint");
        assert!(!p.contains("avoid /tmp/"));
    }
}
