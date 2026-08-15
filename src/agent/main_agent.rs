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
        if let Some(mm) = config.multimodal.as_mut() {
            if mm.request_timeout_secs.is_none() {
                mm.request_timeout_secs = Some(config.timeouts.model_request);
            }
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
    fn build_system_prompt(&self, _registry: &ToolRegistry, skills_list: &str) -> String {
        let mut prompt = SYSTEM_PROMPT_FOR_MAIN_AGENT.replace("{skills_list}", skills_list);
        prompt.push_str(&format!(
            "\n\n## Working Directory\nYour current working directory is: {}\nFor project files, use paths relative to this directory.",
            self.project_path.display()
        ));
        if self.backend.kind() == "fastshell" {
            prompt.push_str(" Paths starting with / (e.g. /skills) are relative to the sandbox root, not the real filesystem. Paths without / are relative to your working directory. avoid /tmp/.");
            #[cfg(any(target_os = "android", target_os = "ios"))]
            {
                prompt.push_str(" Python: embedded RustPython (only pure-Python packages work, no C extensions). Test with `python3 -m unittest`; syntax-check: `python3 -c \"import ast; ast.parse(open('f.py').read())\"`. Install missing pure-Python packages with `pip-install <name>`.");
                prompt.push_str(" Built-in extras: `sqlite3` (embedded DB, no install), `node`/`js` (WebView JavaScript), `render` (HTML→screenshot), `jscheck` (JS/TS syntax check); full command list at https://github.com/kandada/fastshell.");
            }
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
    pub async fn execute(
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
            emitter.session_origin(task, &id, true);
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
            emitter.session_origin(task, &id, false);
        }

        let (registry, sid_holder) = self.build_registry_with_holder();
        // Sync the todo store with the current session.
        {
            let mut sid = sid_holder.lock().unwrap();
            *sid = session.current_session_id.clone();
        }
        let native = self.native_tools(&registry);
        let skills_user_dir = self
            .config
            .skills
            .user_dir
            .as_ref()
            .map(std::path::PathBuf::from);
        let skills_list = {
            let project_path = self.project_path.clone();
            let extra_builtins = self.config.skills.extra_builtins.clone();
            let vfs_skills_dir = self.config.skills.vfs_skills_dir.clone();
            tokio::task::spawn_blocking(move || {
                crate::tools::skills::skills_list_for_prompt(
                    &project_path,
                    skills_user_dir.as_deref(),
                    &extra_builtins,
                    vfs_skills_dir.as_deref(),
                )
            })
            .await
        }
        .unwrap_or_else(|e| {
            eprintln!("skills discovery panicked or was cancelled: {e}");
            "(no skills installed)".to_string()
        });
        let system_prompt = self.build_system_prompt(&registry, &skills_list);

        // Build the message list: system + prior history + current task.
        let mut messages = vec![ChatMessage::system(system_prompt)];

        // If ended mid-tool previously, inject a recovery hint.
        if session.ended_mid_tool() {
            messages.push(ChatMessage::system(
                "[SYSTEM] Previous turn was interrupted mid tool execution. Continue based on existing results; do not repeat prior tool calls.",
            ));
        }
        // Append history directly — skip the trailing duplicate current-task
        // user message without allocating an intermediate Vec.
        {
            let history_len = session.messages.len();
            let skip_last = session
                .messages
                .last()
                .map(|m| m.role == "user" && m.content == task)
                .unwrap_or(false);
            let end = if skip_last {
                history_len.saturating_sub(1)
            } else {
                history_len
            };
            messages.reserve(end);
            for m in &session.messages[..end] {
                messages.push(m.to_chat());
            }
        }

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
        let result = loop_.run(messages, session, emitter, cancel).await;
        let _ = session.flush();
        result
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
        let p = agent.build_system_prompt(&reg, "(no skills installed)");
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
        let p = agent.build_system_prompt(&reg, "(no skills installed)");
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
        let p = agent.build_system_prompt(&reg, "(no skills installed)");
        assert!(p.contains("sandbox root"), "fastshell prompt must contain sandbox hint");
        assert!(p.contains("avoid /tmp/"));
    }

    #[test]
    fn native_backend_does_not_add_sandbox_hint() {
        let (agent, _) = make_agent();
        let reg = agent.build_registry();
        let p = agent.build_system_prompt(&reg, "(no skills installed)");
        assert!(!p.contains("sandbox root"), "native prompt must not contain sandbox hint");
        assert!(!p.contains("avoid /tmp/"));
    }

    #[test]
    fn history_dedup_skips_trailing_duplicate_user_message() {
        let dir = std::env::temp_dir().join(format!(
            "aacode_main_hist_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut session = SessionManager::new(&dir);
        let task = "do the thing";
        session.create_session(task, None).unwrap();
        // Append an assistant reply, then a user dup (simulates a follow-up call).
        session
            .add_message(SessionMessage::from_chat(&ChatMessage::assistant("done")))
            .unwrap();
        session
            .add_message(SessionMessage::from_chat(&ChatMessage::user(task)))
            .unwrap();
        session.flush().unwrap();

        // Re-create messages exactly as execute() does (without running the loop).
        let mut messages: Vec<ChatMessage> = vec![ChatMessage::system("sys")];
        {
            let history_len = session.messages.len();
            let skip_last = session
                .messages
                .last()
                .map(|m| m.role == "user" && m.content == task)
                .unwrap_or(false);
            let end = if skip_last {
                history_len.saturating_sub(1)
            } else {
                history_len
            };
            messages.reserve(end);
            for m in &session.messages[..end] {
                messages.push(m.to_chat());
            }
        }

        // Should have: sys + user(task) + assistant("done") = 3 messages
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[0].content, "sys");
        assert_eq!(messages[1].role, "user");
        assert_eq!(messages[1].content, task);
        assert_eq!(messages[2].role, "assistant");
        assert_eq!(messages[2].content, "done");
    }

    #[test]
    fn history_no_dedup_when_last_message_differs() {
        let dir = std::env::temp_dir().join(format!(
            "aacode_main_hist2_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut session = SessionManager::new(&dir);
        session.create_session("task one", None).unwrap();
        session
            .add_message(SessionMessage::from_chat(&ChatMessage::assistant("ok")))
            .unwrap();
        session.flush().unwrap();

        let task = "different task";
        let mut messages: Vec<ChatMessage> = vec![ChatMessage::system("sys")];
        {
            let history_len = session.messages.len();
            let skip_last = session
                .messages
                .last()
                .map(|m| m.role == "user" && m.content == task)
                .unwrap_or(false);
            let end = if skip_last {
                history_len.saturating_sub(1)
            } else {
                history_len
            };
            messages.reserve(end);
            for m in &session.messages[..end] {
                messages.push(m.to_chat());
            }
        }

        // Last is "task one" ≠ "different task" — should NOT be skipped.
        assert_eq!(messages.len(), 3, "should keep all history when last msg != task");
        assert_eq!(messages[1].content, "task one");
        assert_eq!(messages[2].content, "ok");
    }

    #[test]
    fn history_preserves_system_messages() {
        let dir = std::env::temp_dir().join(format!(
            "aacode_main_hist3_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut session = SessionManager::new(&dir);
        session.create_session("task", None).unwrap();
        session
            .add_message(SessionMessage::from_chat(&ChatMessage::system("## Analysis\n2 files")))
            .unwrap();
        session.add_message(SessionMessage::from_chat(&ChatMessage::assistant("done"))).unwrap();
        session.flush().unwrap();

        let mut messages: Vec<ChatMessage> = vec![ChatMessage::system("sys")];
        {
            let history_len = session.messages.len();
            let end = history_len;
            messages.reserve(end);
            for m in &session.messages[..end] {
                messages.push(m.to_chat());
            }
        }

        let sys_msgs: Vec<_> = messages.iter().filter(|m| m.role == "system").collect();
        assert_eq!(sys_msgs.len(), 2, "should preserve both system messages in history");
        assert!(sys_msgs.iter().any(|m| m.content.contains("Analysis")));
    }

    #[test]
    fn history_empty_session_no_messages() {
        let dir = std::env::temp_dir().join(format!(
            "aacode_main_hist4_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let session = SessionManager::new(&dir);

        let mut messages: Vec<ChatMessage> = vec![ChatMessage::system("sys")];
        {
            let history_len = session.messages.len();
            let end = history_len;
            messages.reserve(end);
            for m in &session.messages[..end] {
                messages.push(m.to_chat());
            }
        }

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "system");
    }

    #[test]
    fn history_tool_calls_roundtrip() {
        let dir = std::env::temp_dir().join(format!(
            "aacode_main_hist5_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut session = SessionManager::new(&dir);
        session.create_session("t", None).unwrap();
        let msg = ChatMessage::assistant_with_tools(
            "let me run that",
            vec![crate::llm::types::ToolCall {
                id: "c1".into(),
                name: "run_shell".into(),
                arguments: r#"{"command":"ls"}"#.into(),
            }],
        );
        let mut sm = SessionMessage::from_chat(&msg);
        sm.reasoning_content = Some("think hard".into());
        session.add_message(sm).unwrap();
        session.flush().unwrap();

        let mut messages: Vec<ChatMessage> = vec![ChatMessage::system("sys")];
        {
            let end = session.messages.len();
            messages.reserve(end);
            for m in &session.messages[..end] {
                messages.push(m.to_chat());
            }
        }

        let asst = messages
            .iter()
            .find(|m| m.role == "assistant" && m.tool_calls.is_some())
            .expect("assistant with tool_calls must be preserved");
        assert_eq!(asst.content, "let me run that");
        assert_eq!(asst.reasoning_content.as_deref(), Some("think hard"));
        let tcs = asst.tool_calls.as_ref().unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].name, "run_shell");
        assert_eq!(tcs[0].arguments, r#"{"command":"ls"}"#);
    }
}
