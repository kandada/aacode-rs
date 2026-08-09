// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Tool system: schema, registry, and the concrete tools.

pub mod backend;
pub mod registry;
pub mod schema;
pub mod shell;

pub mod code_tools;
pub mod delegate;
pub mod mcp;
pub mod multimodal;
pub mod session_tools;
pub mod skills;
pub mod todo;
pub mod web;

pub use backend::{BackendKind, ShellBackend};
pub use registry::{Tool, ToolRegistry};
pub use schema::{ParamType, ToolParameter, ToolSchema};
pub use shell::{SharedShell, ShellTool};

use crate::config::AgentConfig;
use crate::llm::{build_client, LlmClient};
use crate::mcp::{McpManager, McpServerSpec};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;

/// Build the sub-agent registry: all tools EXCEPT delegation (prevents
/// recursion) and MCP by default (kept lean).
pub fn build_sub_registry(
    backend: Arc<dyn ShellBackend>,
    project_path: PathBuf,
    config: &AgentConfig,
) -> ToolRegistry {
    let mut reg = ToolRegistry::new();
    // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0
    let _ = register_core_tools(&mut reg, backend, &project_path, config);
    reg
}

/// Build the full default registry with every tool wired up.
pub fn build_default_registry(
    backend: Arc<dyn ShellBackend>,
    project_path: PathBuf,
    config: &AgentConfig,
) -> ToolRegistry {
    // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0
    build_default_registry_with_holder(backend, project_path, config).0
}

/// Build the full default registry and return the session-id holder
/// so callers can keep the todo store in sync with the active session.
pub fn build_default_registry_with_holder(
    backend: Arc<dyn ShellBackend>,
    project_path: PathBuf,
    config: &AgentConfig,
) -> (ToolRegistry, Arc<Mutex<Option<String>>>) {
    let mut reg = ToolRegistry::new();
    let holder = register_core_tools(&mut reg, backend.clone(), &project_path, config);

    // MCP tools.
    let mcp_specs: Vec<McpServerSpec> = Vec::new(); // configured externally in future
    let mcp_mgr = Arc::new(McpManager::new(mcp_specs, config.timeouts.web_request));
    reg.register(Box::new(mcp::ListMcpToolsTool { mgr: mcp_mgr.clone() }));
    reg.register(Box::new(mcp::CallMcpToolTool { mgr: mcp_mgr.clone() }));
    reg.register(Box::new(mcp::McpStatusTool { mgr: mcp_mgr }));

    // Delegation tools (sub-agent uses a delegation-free registry).
    let llm: Arc<dyn LlmClient> = Arc::from(build_client(&config.model));
    let sub_backend = backend.clone();
    let sub_pp = project_path.clone();
    let sub_cfg = config.clone();
    let factory: delegate::SubRegistryFactory = Arc::new(move || {
        build_sub_registry(sub_backend.clone(), sub_pp.clone(), &sub_cfg)
    });
    reg.register(Box::new(delegate::DelegateTaskTool {
        llm,
        config: config.clone(),
        project_path: project_path.clone(),
        sub_registry: factory,
    }));
    reg.register(Box::new(delegate::CreateSubAgentTool));

    (reg, holder)
}

/// Register the core tool set shared by main + sub agents.
/// Returns the shared session-id holder for todo tools.
fn register_core_tools(
    reg: &mut ToolRegistry,
    backend: Arc<dyn ShellBackend>,
    project_path: &PathBuf,
    config: &AgentConfig,
) -> Arc<Mutex<Option<String>>> {
    // Clone backend before ShellTool consumes it — needed by ExecutePythonTool later.
    let backend_for_python = backend.clone();

    // run_shell (native OS shell on desktop, fastshell sandbox on mobile)
    reg.register(Box::new(ShellTool::new(
        backend,
        project_path.clone(),
        config.limits.tool_output_chars,
        config.timeouts.shell_command,
        config.timeouts.shell_idle,
        config.safety.clone(),
    )));

    // Todo tools (shared store + session-id resolver)
    let session_id_holder: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let sid = session_id_holder.clone();
    let todo_store = Arc::new(todo::TodoStore::with_session_resolver(
        project_path,
        Arc::new(move || sid.lock().unwrap().clone()),
    ));
    reg.register(Box::new(todo::AddTodoTool { store: todo_store.clone() }));
    reg.register(Box::new(todo::MarkTodoTool { store: todo_store.clone() }));
    reg.register(Box::new(todo::UpdateTodoTool { store: todo_store.clone() }));
    reg.register(Box::new(todo::TodoSummaryTool { store: todo_store.clone() }));
    reg.register(Box::new(todo::ListTodoFilesTool { store: todo_store }));
    reg.register(Box::new(todo::AddExecutionRecordTool));

    // Web tools
    reg.register(Box::new(web::SearchWebTool::new(config.search.clone(), config.timeouts.web_request)));
    reg.register(Box::new(web::FetchUrlTool {
        project_path: project_path.clone(),
        timeout_secs: config.timeouts.web_request,
    }));
    reg.register(Box::new(web::SearchCodeTool {
        cfg: config.search.clone(),
        timeout_secs: config.timeouts.web_request,
    }));

    // Skills
    reg.register(Box::new(skills::RunSkillsTool {
        project_path: project_path.clone(),
        user_dir: config.skills.user_dir.as_ref().map(PathBuf::from),
        vfs_skills_dir: config.skills.vfs_skills_dir.clone(),
        extra_builtins: config.skills.extra_builtins.clone(),
    }));

    // Session tools
    reg.register(Box::new(session_tools::ListSessionsTool { project_path: project_path.clone() }));
    reg.register(Box::new(session_tools::GetConversationHistoryTool { project_path: project_path.clone() }));
    reg.register(Box::new(session_tools::GetSessionStatsTool { project_path: project_path.clone() }));
    reg.register(Box::new(session_tools::DeleteSessionTool { project_path: project_path.clone() }));
    reg.register(Box::new(session_tools::SwitchSessionTool { project_path: project_path.clone() }));
    reg.register(Box::new(session_tools::NewSessionTool { project_path: project_path.clone() }));
    reg.register(Box::new(session_tools::ContinueSessionTool { project_path: project_path.clone() }));

    // Code tools (execute_python, run_tests, debug_code, analyze_code)
    reg.register(Box::new(code_tools::ExecutePythonTool { project_path: project_path.clone(), backend: backend_for_python, default_timeout_secs: config.timeouts.shell_command }));
    reg.register(Box::new(code_tools::RunTestsTool { project_path: project_path.clone() }));
    reg.register(Box::new(code_tools::DebugCodeTool { project_path: project_path.clone() }));
    reg.register(Box::new(code_tools::AnalyzeCodeTool { project_path: project_path.clone() }));

    // Multimodal tools
    let mm_ctx = multimodal::MultimodalCtx {
        model: config.multimodal.clone(),
        project_path: project_path.clone(),
        timeout_secs: config.timeouts.web_request,
    };
    reg.register(Box::new(multimodal::UnderstandImageTool { ctx: mm_ctx.clone() }));
    reg.register(Box::new(multimodal::UnderstandVideoTool { ctx: mm_ctx.clone() }));
    reg.register(Box::new(multimodal::UnderstandUiDesignTool { ctx: mm_ctx.clone() }));
    reg.register(Box::new(multimodal::ImageConsistencyTool { ctx: mm_ctx }));

    session_id_holder
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::backend::NativeShell;

    fn make_backend() -> (Arc<dyn ShellBackend>, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "aacode_reg_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        (Arc::new(NativeShell::new()), dir)
    }

    #[test]
    fn default_registry_has_all_tools() {
        let (backend, dir) = make_backend();
        let mut cfg = AgentConfig::default();
        cfg.model.api_key = Some("x".into());
        let reg = build_default_registry(backend, dir, &cfg);
        for name in [
            "run_shell",
            "add_todo_item",
            "mark_todo_completed",
            "update_todo_item",
            "get_todo_summary",
            "list_todo_files",
            "add_execution_record",
            "search_web",
            "fetch_url",
            "search_code",
            "run_skills",
            "list_sessions",
            "get_conversation_history",
            "get_session_stats",
            "delete_session",
            "switch_session",
            "new_session",
            "continue_session",
            "understand_image",
            "understand_video",
            "understand_ui_design",
            "analyze_image_consistency",
            "list_mcp_tools",
            "call_mcp_tool",
            "get_mcp_status",
            "delegate_task",
            "create_sub_agent",
        ] {
            assert!(reg.contains(name), "missing tool: {name}");
        }
    }

    #[test]
    fn sub_registry_excludes_delegation() {
        let (backend, dir) = make_backend();
        let cfg = AgentConfig::default();
        let reg = build_sub_registry(backend, dir, &cfg);
        assert!(reg.contains("run_shell"));
        assert!(!reg.contains("delegate_task"));
        assert!(!reg.contains("list_mcp_tools"));
    }

    #[test]
    fn native_tools_export_nonempty() {
        let (backend, dir) = make_backend();
        let mut cfg = AgentConfig::default();
        cfg.model.api_key = Some("x".into());
        let reg = build_default_registry(backend, dir, &cfg);
        assert!(!reg.openai_tools().is_empty());
        assert!(!reg.anthropic_tools().is_empty());
    }
}
