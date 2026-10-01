// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Tool registry. Async Tool trait.

use super::schema::ToolSchema;
use crate::error::Result;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;

#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    fn schema(&self) -> ToolSchema;
    async fn call(&self, args: &Value, cancel: &AtomicBool) -> Result<String>;
    fn name(&self) -> &'static str {
        ""
    }
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
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }
    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.tools.keys().cloned().collect();
        v.sort();
        v
    }
    pub fn schema(&self, name: &str) -> Option<&ToolSchema> {
        self.schemas.get(name)
    }

    /// Async execute for the loop.
    pub async fn execute(&self, name: &str, raw_args: Value, cancel: &AtomicBool) -> String {
        let tool = match self.tools.get(name) {
            Some(t) => t,
            None => return self.not_found_message(name),
        };
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

    pub fn openai_tools(&self) -> Vec<Value> {
        let mut names: Vec<&String> = self.schemas.keys().collect();
        names.sort();
        names
            .iter()
            .filter_map(|n| self.schemas.get(*n))
            .map(|s| s.to_openai_tool())
            .collect()
    }
    pub fn anthropic_tools(&self) -> Vec<Value> {
        let mut names: Vec<&String> = self.schemas.keys().collect();
        names.sort();
        names
            .iter()
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
}
