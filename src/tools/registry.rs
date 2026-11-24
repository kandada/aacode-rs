// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Tool registry. Async Tool trait.

use super::schema::ToolSchema;
use crate::error::Result;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicBool;

#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    fn schema(&self) -> ToolSchema;
    async fn call(&self, args: &Value, cancel: &AtomicBool) -> Result<String>;
    fn name(&self) -> &'static str {
        ""
    }
    /// Non-reentrant resource lane (e.g. `"shell"`, `"browser"`, `"python"`).
    ///
    /// Tools sharing a lane must not run concurrently. A batch that contains
    /// any lane tool is executed sequentially (safe and simple); batches with
    /// no lane still run concurrently.
    fn lane(&self) -> Option<&'static str> {
        None
    }
}

/// Conservative operation classification for the read-only gate.
///
/// **Fail-closed**: anything not listed here is treated as mutating/executing.
/// New read-only tools must be added explicitly; unknown tools are blocked in
/// `read_only` mode rather than silently allowed.
pub fn is_read_only(name: &str) -> bool {
    const EXACT: &[&str] = &[
        "snapshot",
        "screenshot",
        "fetch_url",
        "fetch_rendered",
        "browser_tools",
        "mcp_status",
        "run_skills",
        "find_elements",
        "navigate",
        "open",
        "back",
        "forward",
        "reload",
        "stop",
        "scroll",
        "session_export",
        "job_status",
        "job_collect",
        "job_list",
        "list_tool_schemas",
        "get_tool_schema",
        "ui_backends",
        "ui_list_windows",
        "ui_snapshot",
        "ui_find",
        "ui_wait",
    ];
    const PREFIXES: &[&str] = &[
        "get_",
        "list_",
        "search_",
        "read_",
        "extract_",
        "wait_",
        "assert_",
        "understand_",
        "analyze_",
    ];
    EXACT.contains(&name) || PREFIXES.iter().any(|p| name.starts_with(p))
}

#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
    schemas: HashMap<String, ToolSchema>,
    /// When set, every tool observation is sanitized before it reaches the
    /// model/session (see `crate::observation`). Optional so unit tests that
    /// only exercise a single tool keep working with raw output.
    artifact_store: Option<crate::artifacts::ArtifactStore>,
    preview_budget: Option<std::sync::Arc<crate::observation::PreviewBudget>>,
    sanitize_limits: crate::observation::SanitizeLimits,
    /// Active model-facing schemas. `None` = all registered schemas (legacy).
    active: Option<HashSet<String>>,
    /// When true, `call_tool` may only target read-only tools.
    read_only: bool,
}

impl ToolRegistry {
    pub fn new() -> Self {
        ToolRegistry::default()
    }

    /// Enable the observation choke point: all tool output is passed through
    /// `observation::sanitize` (binary → artifact ref + bounded preview).
    pub fn enable_sanitizer(
        &mut self,
        project_path: &std::path::Path,
        limits: crate::observation::SanitizeLimits,
    ) {
        self.preview_budget = Some(std::sync::Arc::new(
            crate::observation::PreviewBudget::default_for(&limits),
        ));
        self.artifact_store = Some(crate::artifacts::ArtifactStore::new(project_path));
        self.sanitize_limits = limits;
    }

    pub fn artifact_store(&self) -> Option<&crate::artifacts::ArtifactStore> {
        self.artifact_store.as_ref()
    }

    fn sanitize_observation(&self, obs: String) -> String {
        match (self.artifact_store.as_ref(), self.preview_budget.as_ref()) {
            (Some(store), Some(budget)) => {
                crate::observation::sanitize(&obs, store, budget, &self.sanitize_limits)
            }
            _ => obs,
        }
    }
    pub fn register(&mut self, tool: Box<dyn Tool>) {
        let schema = tool.schema();
        let name = schema.name.to_string();
        self.schemas.insert(name.clone(), schema);
        self.tools.insert(name, tool);
    }

    /// Install the profile-stable discovery schemas. They have no `Tool`
    /// implementation; `execute()` handles them directly.
    pub fn install_meta_schemas(&mut self) {
        for schema in super::schema::meta_schemas() {
            self.schemas
                .entry(schema.name.to_string())
                .or_insert(schema);
        }
    }

    /// Apply a tool profile and read-only policy. Meta tools stay active.
    pub fn set_profile(&mut self, profile: &str, read_only: bool) {
        let names = self.names();
        let active = select_profile(profile, &names);
        self.active = Some(active);
        self.read_only = read_only;
    }

    pub fn active_names(&self) -> Vec<String> {
        let mut names: Vec<String> = match &self.active {
            Some(active) => self
                .schemas
                .keys()
                .filter(|n| active.contains(n.as_str()))
                .cloned()
                .collect(),
            None => self.schemas.keys().cloned().collect(),
        };
        names.sort();
        names
    }

    pub fn is_active(&self, name: &str) -> bool {
        self.active
            .as_ref()
            .map(|a| a.contains(name))
            .unwrap_or(true)
    }

    fn schema_json(&self, schema: &ToolSchema) -> Value {
        schema.to_openai_tool()["function"].clone()
    }
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// The non-reentrant lane for a tool, if any.
    pub fn lane_of(&self, name: &str) -> Option<&'static str> {
        self.tools.get(name).and_then(|t| t.lane())
    }
    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.schemas.keys().cloned().collect();
        v.sort();
        v
    }
    pub fn schema(&self, name: &str) -> Option<&ToolSchema> {
        self.schemas.get(name)
    }

    /// Async execute for the loop.
    pub async fn execute(&self, name: &str, raw_args: Value, cancel: &AtomicBool) -> String {
        // Profile-stable discovery/proxy tools.
        if name == "list_tool_schemas" || name == "get_tool_schema" || name == "call_tool" {
            return self.execute_meta(name, raw_args, cancel).await;
        }
        if !self.is_active(name) {
            return format!(
                "Error: tool '{name}' is not advertised in this session's profile. \
Use list_tool_schemas/get_tool_schema then call_tool(name, arguments) to invoke it explicitly."
            );
        }
        let tool = match self.tools.get(name) {
            Some(t) => t,
            None => return self.not_found_message(name),
        };
        if self.read_only && !is_read_only(name) {
            return serde_json::json!({
                "success": false,
                "error": format!("read-only mode: tool '{name}' is not allowed"),
                "tool": name,
            })
            .to_string();
        }
        let schema = self.schemas.get(name).unwrap();
        let normalized = schema.normalize_params(&raw_args);
        if let Err(msg) = schema.validate(&normalized) {
            return format!(
                "Parameter validation failed\n\n{msg}\n\nTool docs:\n{}",
                schema.documentation()
            );
        }
        match tool.call(&normalized, cancel).await {
            Ok(obs) => self.sanitize_observation(obs),
            Err(crate::error::AacodeError::Cancelled) => "Execution cancelled".to_string(),
            Err(e) => format!("Execution error: {e}"),
        }
    }

    async fn execute_meta(&self, name: &str, raw_args: Value, cancel: &AtomicBool) -> String {
        let args = raw_args.as_object().cloned().unwrap_or_default();
        match name {
            "list_tool_schemas" => {
                let query = args
                    .get("query")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_lowercase();
                let active_only = args
                    .get("active_only")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let mut out = String::from("Tool schemas:\n");
                for tool_name in self.names() {
                    if active_only && !self.is_active(&tool_name) {
                        continue;
                    }
                    let Some(schema) = self.schemas.get(&tool_name) else {
                        continue;
                    };
                    if !query.is_empty()
                        && !format!("{} {}", tool_name, schema.description)
                            .to_lowercase()
                            .contains(&query)
                    {
                        continue;
                    }
                    let status = if self.is_active(&tool_name) {
                        "active"
                    } else {
                        "inactive"
                    };
                    out.push_str(&format!(
                        "- {tool_name} [{status}]: {}\n",
                        schema.description
                    ));
                }
                out.push_str("\nUse get_tool_schema(name) then call_tool(name, arguments).");
                out
            }
            "get_tool_schema" => {
                let target = args.get("name").and_then(|v| v.as_str()).unwrap_or("");
                match self.schemas.get(target) {
                    Some(schema) => serde_json::to_string_pretty(&self.schema_json(schema))
                        .unwrap_or_else(|_| "schema serialization failed".into()),
                    None => self.not_found_message(target),
                }
            }
            "call_tool" => {
                let target = args.get("name").and_then(|v| v.as_str()).unwrap_or("");
                if target.is_empty() {
                    return "Error: call_tool requires a non-empty name".into();
                }
                if matches!(
                    target,
                    "call_tool" | "list_tool_schemas" | "get_tool_schema"
                ) {
                    return format!("Error: {target} cannot be invoked through call_tool");
                }
                if !self.tools.contains_key(target) {
                    return self.not_found_message(target);
                }
                if self.read_only && !is_read_only(target) {
                    return format!("Error: read-only mode: tool '{target}' is not allowed");
                }
                let arguments = args
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({}));
                let schema = self.schemas.get(target).unwrap();
                let normalized = schema.normalize_params(&arguments);
                if let Err(msg) = schema.validate(&normalized) {
                    return format!(
                        "Parameter validation failed\n\n{msg}\n\nTool docs:\n{}",
                        schema.documentation()
                    );
                }
                match self
                    .tools
                    .get(target)
                    .unwrap()
                    .call(&normalized, cancel)
                    .await
                {
                    Ok(obs) => self.sanitize_observation(obs),
                    Err(crate::error::AacodeError::Cancelled) => "Execution cancelled".to_string(),
                    Err(e) => format!("Execution error: {e}"),
                }
            }
            _ => self.not_found_message(name),
        }
    }

    pub fn openai_tools(&self) -> Vec<Value> {
        let mut names: Vec<&String> = self.schemas.keys().collect();
        names.sort();
        names
            .iter()
            .filter(|n| self.is_active(n))
            .filter_map(|n| self.schemas.get(*n))
            .map(|s| s.to_openai_tool())
            .collect()
    }
    pub fn anthropic_tools(&self) -> Vec<Value> {
        let mut names: Vec<&String> = self.schemas.keys().collect();
        names.sort();
        names
            .iter()
            .filter(|n| self.is_active(n))
            .filter_map(|n| self.schemas.get(*n))
            .map(|s| s.to_anthropic_tool())
            .collect()
    }

    fn not_found_message(&self, name: &str) -> String {
        let mut msg = format!("Error: Unknown tool '{name}'\n\n");
        let suggestions = self.suggest_similar(name, 3);
        if !suggestions.is_empty() {
            msg.push_str("Did you mean:\n");
            for s in &suggestions {
                msg.push_str(&format!("  - {s}\n"));
            }
            msg.push('\n');
        }
        msg.push_str("Available tools:\n");
        for n in self.names() {
            if let Some(s) = self.schemas.get(&n) {
                let desc: String = s.description.chars().take(60).collect();
                msg.push_str(&format!("  - {n}: {desc}\n"));
            }
        }
        msg
    }

    fn suggest_similar(&self, name: &str, max: usize) -> Vec<String> {
        let mut scored: Vec<(usize, &str)> = self
            .tools
            .keys()
            .map(|k| (levenshtein(name, k), k.as_str()))
            .filter(|(d, _)| *d <= 4)
            .collect();
        scored.sort_by_key(|(d, _)| *d);
        scored
            .iter()
            .take(max)
            .map(|(_, k)| k.to_string())
            .collect()
    }
}

fn select_profile(profile: &str, available: &[String]) -> HashSet<String> {
    let mut active = HashSet::new();
    let lower = profile.trim().to_lowercase();
    // Keep the default surface lean, matching Python's core/auto split:
    // shell + skills + delegate + todos + history + a few web tools.
    // Jobs, code, session management, UI and media are discoverable through
    // list_tool_schemas/get_tool_schema/call_tool instead of being advertised
    // on every request.
    let is_core = |name: &str| {
        matches!(
            name,
            "run_shell"
                | "run_skills"
                | "delegate_task"
                | "add_todo_item"
                | "mark_todo_completed"
                | "update_todo_item"
                | "get_todo_summary"
                | "list_todo_files"
                | "search_history"
                | "read_history"
        )
    };
    let is_auto_web = |name: &str| matches!(name, "search_web" | "fetch_url" | "fetch_rendered");
    let is_web = |name: &str| {
        matches!(
            name,
            "search_web" | "fetch_url" | "search_code" | "fetch_rendered"
        )
    };
    let is_browser = |name: &str| {
        matches!(
            name,
            "browser_call" | "browser_tools" | "ax_act" | "fetch_rendered"
        )
    };
    let is_mcp = |name: &str| {
        name.starts_with("mcp_")
            || matches!(name, "list_mcp_tools" | "call_mcp_tool" | "get_mcp_status")
    };
    let is_media =
        |name: &str| name.starts_with("understand_") || name.starts_with("analyze_image");
    for name in available {
        let meta = matches!(
            name.as_str(),
            "list_tool_schemas" | "get_tool_schema" | "call_tool"
        );
        let include = if meta {
            true
        } else {
            match lower.as_str() {
                "core" => is_core(name),
                "auto" => is_core(name) || is_auto_web(name),
                "web" => is_core(name) || is_web(name) || is_browser(name) || is_mcp(name),
                "full" => true,
                // Unknown/empty profile falls back to lean auto.
                _ => is_core(name) || is_auto_web(name),
            }
        };
        if include {
            active.insert(name.clone());
        }
    }
    // `web`/`full` should not accidentally drop the media category in core.
    if lower == "full" {
        for name in available {
            if is_media(name) || is_mcp(name) || is_browser(name) {
                active.insert(name.clone());
            }
        }
    }
    active
}

fn levenshtein(a: &str, b: &str) -> usize {
    let (a, b) = if a.len() > b.len() { (b, a) } else { (a, b) };
    let n = a.len();
    let m = b.len();
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];
    for j in 1..=m {
        curr[0] = j;
        for i in 1..=n {
            curr[i] = if a.as_bytes()[i - 1] == b.as_bytes()[j - 1] {
                prev[i - 1]
            } else {
                1 + prev[i - 1].min(prev[i]).min(curr[i - 1])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

#[cfg(test)]
mod tests {
    use super::super::schema::{ParamType, ToolParameter};
    use super::*;
    use serde_json::json;

    #[test]
    fn read_only_classification_is_fail_closed() {
        assert!(is_read_only("get_page_text"));
        assert!(is_read_only("list_tabs"));
        assert!(is_read_only("wait_for_element"));
        assert!(is_read_only("search_history"));
        assert!(is_read_only("navigate"));
        assert!(is_read_only("snapshot"));
        assert!(is_read_only("job_status"));
        assert!(is_read_only("job_collect"));
        assert!(!is_read_only("job_start"));
        assert!(!is_read_only("job_cancel"));
        assert!(!is_read_only("run_shell"));
        assert!(!is_read_only("execute_python"));
        assert!(!is_read_only("browser_call"));
        assert!(!is_read_only("click"));
        assert!(!is_read_only("delegate_task"));
        // Unknown tools are blocked in read-only mode (fail-closed).
        assert!(!is_read_only("some_future_mutating_tool"));
    }

    struct BlobTool;
    #[async_trait::async_trait]
    impl Tool for BlobTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema::new("blob", "returns a big base64 blob", vec![])
        }
        async fn call(&self, _a: &Value, _c: &AtomicBool) -> Result<String> {
            use base64::Engine;
            let raw: Vec<u8> = (0..200_000u32).map(|i| (i % 256) as u8).collect();
            let b64 = base64::engine::general_purpose::STANDARD.encode(&raw);
            Ok(json!({"result": {"base64": b64, "width": 400, "height": 800}}).to_string())
        }
    }

    struct EchoTool(&'static str);
    #[async_trait::async_trait]
    impl Tool for EchoTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema::new(
                self.0,
                "test echo tool",
                vec![ToolParameter::new(
                    "text",
                    ParamType::String,
                    false,
                    "text",
                    &[],
                )],
            )
        }
        async fn call(&self, args: &Value, _cancel: &AtomicBool) -> Result<String> {
            Ok(format!(
                "echo:{}",
                args.get("text").and_then(|v| v.as_str()).unwrap_or("")
            ))
        }
    }

    fn dummy_registry() -> ToolRegistry {
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(EchoTool("run_shell")));
        reg.register(Box::new(EchoTool("search_web")));
        reg.register(Box::new(EchoTool("understand_image")));
        reg.register(Box::new(EchoTool("browser_call")));
        reg.install_meta_schemas();
        reg
    }

    #[test]
    fn profiles_filter_schemas_but_keep_meta_tools() {
        let mut reg = dummy_registry();
        reg.set_profile("auto", false);
        let active = reg.active_names();
        assert!(active.contains(&"run_shell".into()));
        assert!(active.contains(&"search_web".into()));
        assert!(!active.contains(&"understand_image".into()));
        assert!(!active.contains(&"browser_call".into()));
        for meta in ["list_tool_schemas", "get_tool_schema", "call_tool"] {
            assert!(active.contains(&meta.into()), "missing meta {meta}");
        }
        reg.set_profile("full", false);
        assert!(reg.active_names().contains(&"understand_image".into()));
        assert!(reg.active_names().contains(&"browser_call".into()));
    }

    #[tokio::test]
    async fn call_tool_invokes_a_profile_hidden_tool() {
        let mut reg = dummy_registry();
        reg.set_profile("auto", false);
        let out = reg
            .execute(
                "call_tool",
                json!({"name": "understand_image", "arguments": {"text": "hi"}}),
                &AtomicBool::new(false),
            )
            .await;
        assert_eq!(out, "echo:hi");
        let listing = reg
            .execute(
                "list_tool_schemas",
                json!({"query": "understand"}),
                &AtomicBool::new(false),
            )
            .await;
        assert!(listing.contains("understand_image"));
        assert!(listing.contains("inactive"));
    }

    #[tokio::test]
    async fn read_only_call_tool_blocks_mutating_targets() {
        let mut reg = dummy_registry();
        reg.set_profile("auto", true);
        let allowed = reg
            .execute(
                "call_tool",
                json!({"name": "search_history", "arguments": {}}),
                &AtomicBool::new(false),
            )
            .await;
        // search_history isn't registered in this toy registry, so it must fail cleanly.
        assert!(allowed.contains("Unknown tool") || allowed.contains("not found"));
        let blocked = reg
            .execute(
                "call_tool",
                json!({"name": "run_shell", "arguments": {"text": "x"}}),
                &AtomicBool::new(false),
            )
            .await;
        assert!(blocked.contains("read-only mode"));
        let schema = reg
            .execute(
                "get_tool_schema",
                json!({"name": "run_shell"}),
                &AtomicBool::new(false),
            )
            .await;
        assert!(schema.contains("run_shell"));
    }

    /// The choke point must bound *any* tool output by default: a large base64
    /// payload comes back as an artifact ref + tiny preview, never the bulk.
    #[tokio::test]
    async fn execute_sanitizes_bulk_binary_by_default() {
        let dir = std::env::temp_dir().join(format!("aacode_reg_san_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut reg = ToolRegistry::new();
        reg.enable_sanitizer(&dir, crate::observation::SanitizeLimits::default());
        reg.register(Box::new(BlobTool));
        let out = reg
            .execute("blob", json!({}), &AtomicBool::new(false))
            .await;
        assert!(out.len() < 2_000, "output not bounded: {} chars", out.len());
        assert!(out.contains("artifact"), "no artifact ref: {out}");
        assert!(
            out.contains(".aacode/artifacts/"),
            "no artifact path: {out}"
        );
        assert!(!out.contains(&"A".repeat(200)), "bulk base64 leaked");
    }

    /// Without the sanitizer (single-tool unit tests) raw output is preserved.
    #[tokio::test]
    async fn execute_without_sanitizer_is_raw() {
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(BlobTool));
        let out = reg
            .execute("blob", json!({}), &AtomicBool::new(false))
            .await;
        assert!(out.len() > 2_000, "raw output expected without sanitizer");
    }

    struct Echo;
    #[async_trait::async_trait]
    impl Tool for Echo {
        fn schema(&self) -> ToolSchema {
            ToolSchema::new(
                "echo",
                "echo text",
                vec![ToolParameter::new(
                    "text",
                    ParamType::String,
                    true,
                    "text",
                    &["msg"],
                )],
            )
        }
        async fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
            Ok(args["text"].as_str().unwrap_or("").to_string())
        }
    }

    fn reg() -> ToolRegistry {
        let mut r = ToolRegistry::new();
        r.register(Box::new(Echo));
        r
    }

    #[tokio::test]
    async fn executes_with_alias() {
        let r = reg();
        let cancel = AtomicBool::new(false);
        let out = r.execute("echo", json!({"msg": "hi"}), &cancel).await;
        assert_eq!(out, "hi");
    }

    #[tokio::test]
    async fn validation_error_for_missing() {
        let r = reg();
        let cancel = AtomicBool::new(false);
        let out = r.execute("echo", json!({}), &cancel).await;
        assert!(out.contains("validation failed"));
        assert!(out.contains("text"));
    }

    #[tokio::test]
    async fn unknown_tool_suggests() {
        let r = reg();
        let cancel = AtomicBool::new(false);
        let out = r.execute("ecko", json!({}), &cancel).await;
        assert!(out.contains("Unknown tool 'ecko'"));
        assert!(out.contains("echo"));
    }

    #[test]
    fn levenshtein_basic() {
        assert_eq!(levenshtein("cat", "bat"), 1);
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("echo", "echo"), 0);
    }

    #[test]
    fn exports_native_tools() {
        assert!(!reg().openai_tools().is_empty());
    }

    fn profile_registry() -> ToolRegistry {
        let mut reg = ToolRegistry::new();
        for name in [
            "run_shell",
            "search_web",
            "fetch_url",
            "browser_call",
            "understand_image",
            "mcp_demo",
            "custom_widget",
        ] {
            reg.register(Box::new(EchoTool(name)));
        }
        reg.install_meta_schemas();
        reg
    }

    #[test]
    fn core_profile_hides_web_browser_mcp_and_media() {
        let mut reg = profile_registry();
        reg.set_profile("core", false);
        let active = reg.active_names();
        assert!(active.contains(&"run_shell".into()));
        for hidden in [
            "search_web",
            "fetch_url",
            "browser_call",
            "understand_image",
            "mcp_demo",
            "custom_widget",
        ] {
            assert!(
                !active.contains(&hidden.into()),
                "{hidden} should be hidden"
            );
        }
        for meta in ["list_tool_schemas", "get_tool_schema", "call_tool"] {
            assert!(active.contains(&meta.into()), "missing meta {meta}");
        }
    }

    #[test]
    fn web_profile_includes_web_browser_and_mcp_but_not_media() {
        let mut reg = profile_registry();
        reg.set_profile("web", false);
        let active = reg.active_names();
        for included in [
            "run_shell",
            "search_web",
            "fetch_url",
            "browser_call",
            "mcp_demo",
        ] {
            assert!(active.contains(&included.into()), "missing {included}");
        }
        assert!(!active.contains(&"understand_image".into()));
        assert!(!active.contains(&"custom_widget".into()));
    }

    #[test]
    fn auto_profile_is_lean_and_leaves_heavy_tools_for_discovery() {
        let mut reg = ToolRegistry::new();
        for name in [
            "run_shell",
            "run_skills",
            "delegate_task",
            "add_todo_item",
            "mark_todo_completed",
            "update_todo_item",
            "get_todo_summary",
            "list_todo_files",
            "search_history",
            "read_history",
            "search_web",
            "fetch_url",
            "fetch_rendered",
            "search_code",
            "job_start",
            "execute_python",
            "run_tests",
            "debug_code",
            "analyze_code",
            "list_sessions",
            "get_conversation_history",
            "ui_snapshot",
            "browser_call",
            "mcp_demo",
            "understand_image",
        ] {
            reg.register(Box::new(EchoTool(name)));
        }
        reg.install_meta_schemas();
        reg.set_profile("auto", false);
        let active = reg.active_names();
        for included in [
            "run_shell",
            "run_skills",
            "delegate_task",
            "add_todo_item",
            "search_history",
            "read_history",
            "search_web",
            "fetch_url",
            "fetch_rendered",
            "list_tool_schemas",
            "get_tool_schema",
            "call_tool",
        ] {
            assert!(
                active.contains(&included.into()),
                "missing auto tool {included}"
            );
        }
        for hidden in [
            "search_code",
            "job_start",
            "execute_python",
            "run_tests",
            "debug_code",
            "analyze_code",
            "list_sessions",
            "get_conversation_history",
            "ui_snapshot",
            "browser_call",
            "mcp_demo",
            "understand_image",
        ] {
            assert!(!active.contains(&hidden.into()), "auto must hide {hidden}");
        }
        assert!(
            active.len() <= 16,
            "auto schema surface is too large: {}",
            active.len()
        );
    }

    #[test]
    fn full_profile_includes_every_registered_schema() {
        let mut reg = profile_registry();
        reg.set_profile("full", false);
        let names = reg.names();
        assert_eq!(reg.active_names(), names);
    }

    #[tokio::test]
    async fn inactive_tool_direct_execute_is_rejected() {
        let mut reg = profile_registry();
        reg.set_profile("auto", false);
        let out = reg
            .execute("browser_call", json!({}), &AtomicBool::new(false))
            .await;
        assert!(out.contains("not advertised"), "got: {out}");
    }

    #[tokio::test]
    async fn call_tool_rejects_meta_recursion_and_empty_name() {
        let reg = profile_registry();
        let out = reg
            .execute(
                "call_tool",
                json!({"name": "call_tool", "arguments": {}}),
                &AtomicBool::new(false),
            )
            .await;
        assert!(out.contains("cannot be invoked"), "got: {out}");
        let out = reg
            .execute("call_tool", json!({"name": ""}), &AtomicBool::new(false))
            .await;
        assert!(out.contains("non-empty name"), "got: {out}");
        let out = reg
            .execute(
                "call_tool",
                json!({"name": "missing"}),
                &AtomicBool::new(false),
            )
            .await;
        assert!(out.contains("Unknown tool"), "got: {out}");
    }

    #[tokio::test]
    async fn call_tool_normalizes_nested_argument_aliases() {
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(Echo));
        reg.install_meta_schemas();
        let out = reg
            .execute(
                "call_tool",
                json!({"name": "echo", "arguments": {"msg": "hi"}}),
                &AtomicBool::new(false),
            )
            .await;
        assert_eq!(out, "hi");
    }

    #[tokio::test]
    async fn call_tool_allows_registered_read_only_targets_in_read_only_mode() {
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(EchoTool("search_history")));
        reg.install_meta_schemas();
        reg.set_profile("auto", true);
        let out = reg
            .execute(
                "call_tool",
                json!({"name": "search_history", "arguments": {"text": "q"}}),
                &AtomicBool::new(false),
            )
            .await;
        assert_eq!(out, "echo:q");
    }

    #[tokio::test]
    async fn list_active_only_hides_profile_hidden_tools() {
        let mut reg = profile_registry();
        reg.set_profile("auto", false);
        let out = reg
            .execute(
                "list_tool_schemas",
                json!({"active_only": true, "query": "image"}),
                &AtomicBool::new(false),
            )
            .await;
        assert!(!out.contains("understand_image"), "got: {out}");
        let out = reg
            .execute(
                "list_tool_schemas",
                json!({"query": "image"}),
                &AtomicBool::new(false),
            )
            .await;
        assert!(out.contains("understand_image"));
        assert!(out.contains("inactive"));
    }

    #[test]
    fn profile_names_are_case_insensitive_and_unknown_falls_back_to_auto() {
        let mut reg = profile_registry();
        reg.set_profile("  WEB  ", false);
        let web = reg.active_names();
        assert!(web.contains(&"search_web".into()));
        assert!(web.contains(&"browser_call".into()));
        assert!(!web.contains(&"understand_image".into()));

        reg.set_profile("nonsense", false);
        let fallback = reg.active_names();
        assert!(fallback.contains(&"search_web".into()));
        assert!(!fallback.contains(&"browser_call".into()));
        assert!(!fallback.contains(&"understand_image".into()));
    }

    #[tokio::test]
    async fn get_tool_schema_returns_hidden_tool_json() {
        let mut reg = profile_registry();
        reg.set_profile("auto", false);
        let out = reg
            .execute(
                "get_tool_schema",
                json!({"name": "understand_image"}),
                &AtomicBool::new(false),
            )
            .await;
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["name"], "understand_image");
        assert_eq!(v["parameters"]["type"], "object");

        let missing = reg
            .execute(
                "get_tool_schema",
                json!({"name": "nope"}),
                &AtomicBool::new(false),
            )
            .await;
        assert!(missing.contains("Unknown tool"), "got: {missing}");
    }

    #[test]
    fn openai_tools_follow_profile_and_are_sorted() {
        let mut reg = profile_registry();
        reg.set_profile("web", false);
        let tools = reg.openai_tools();
        let names: Vec<String> = tools
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap().to_string())
            .collect();
        assert!(names.contains(&"search_web".to_string()));
        assert!(!names.contains(&"understand_image".to_string()));
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
    }
}
