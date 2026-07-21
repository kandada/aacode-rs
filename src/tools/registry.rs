// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Tool registry: registration, schema-driven validation, alias normalization,
//! native-tools export, and friendly not-found suggestions.
//!
//! Ported from Python `utils/tool_registry.py`.

use super::schema::ToolSchema;
use crate::error::Result;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;

/// A callable tool. `call` receives already-normalized+validated arguments and
/// returns the observation string that goes back to the model.
pub trait Tool: Send + Sync {
    fn schema(&self) -> ToolSchema;
    fn call(&self, args: &Value, cancel: &AtomicBool) -> Result<String>;
    fn name(&self) -> &'static str {
        // Default derives from schema; tools may override to avoid allocation.
        // Note: schema() returns owned; callers should prefer schema().name.
        ""
    }
}

/// Registry of tools keyed by name.
#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
    schemas: HashMap<String, ToolSchema>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        ToolRegistry::default()
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

    /// Export all schemas as OpenAI native tools.
    pub fn openai_tools(&self) -> Vec<Value> {
        let mut names = self.names();
        names.sort();
        names
            .iter()
            .filter_map(|n| self.schemas.get(n))
            .map(|s| s.to_openai_tool())
            .collect()
    }

    /// Export all schemas as Anthropic native tools.
    pub fn anthropic_tools(&self) -> Vec<Value> {
        let mut names = self.names();
        names.sort();
        names
            .iter()
            .filter_map(|n| self.schemas.get(n))
            .map(|s| s.to_anthropic_tool())
            .collect()
    }

    /// Execute a tool by name with raw model-provided args. Handles unknown
    /// tool, alias normalization, validation, then dispatch. Errors are
    /// returned as `Ok(observation_string)` (never panics the loop) except for
    /// cancellation which propagates.
    pub fn execute(&self, name: &str, raw_args: Value, cancel: &AtomicBool) -> String {
        let tool = match self.tools.get(name) {
            Some(t) => t,
            None => return self.not_found_message(name),
        };
        let schema = self.schemas.get(name).unwrap();
        let normalized = schema.normalize_params(&raw_args);
        if let Err(msg) = schema.validate(&normalized) {
            let mut out =
                format!("❌ Parameter validation failed\n\n{msg}\n\n📖 Tool docs:\n");
            out.push_str(&schema.documentation());
            return out;
        }
        match tool.call(&normalized, cancel) {
            Ok(obs) => obs,
            Err(crate::error::AacodeError::Cancelled) => "❌ Execution cancelled".to_string(),
            Err(e) => format!("❌ Execution error: {e}"),
        }
    }

    /// Build the "unknown tool" message with similar-name suggestions.
    fn not_found_message(&self, name: &str) -> String {
        let mut msg = format!("Error: Unknown tool '{name}'\n\n");
        let suggestions = self.suggest_similar(name, 3);
        if !suggestions.is_empty() {
            msg.push_str("Did you mean one of these tools?\n");
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

    /// Simple edit-distance based similar-name suggestions.
    fn suggest_similar(&self, name: &str, max: usize) -> Vec<String> {
        let mut scored: Vec<(usize, String)> = self
            .names()
            .into_iter()
            .map(|n| (levenshtein(name, &n), n))
            .collect();
        scored.sort_by_key(|(d, _)| *d);
        scored
            .into_iter()
            .filter(|(d, n)| *d <= (n.len().max(name.len()) / 2).max(2))
            .take(max)
            .map(|(_, n)| n)
            .collect()
    }
}

/// Classic Levenshtein distance.
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::super::schema::{ParamType, ToolParameter};
    use super::*;
    use serde_json::json;

    struct Echo;
    impl Tool for Echo {
        fn schema(&self) -> ToolSchema {
            ToolSchema::new(
                "echo",
                "echo text",
                vec![ToolParameter::new(
                    "text",
                    ParamType::String,
                    true,
                    "text to echo",
                    &["msg", "message"],
                )],
            )
        }
        fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
            Ok(args["text"].as_str().unwrap_or("").to_string())
        }
    }

    fn reg() -> ToolRegistry {
        let mut r = ToolRegistry::new();
        r.register(Box::new(Echo));
        r
    }

    #[test]
    fn executes_with_alias() {
        let r = reg();
        let cancel = AtomicBool::new(false);
        let out = r.execute("echo", json!({"msg": "hi"}), &cancel);
        assert_eq!(out, "hi");
    }

    #[test]
    fn validation_error_for_missing() {
        let r = reg();
        let cancel = AtomicBool::new(false);
        let out = r.execute("echo", json!({}), &cancel);
        assert!(out.contains("validation failed"));
        assert!(out.contains("text"));
    }

    #[test]
    fn unknown_tool_suggests() {
        let r = reg();
        let cancel = AtomicBool::new(false);
        let out = r.execute("ecko", json!({}), &cancel);
        assert!(out.contains("Unknown tool 'ecko'"));
        assert!(out.contains("echo"));
    }

    #[test]
    fn exports_native_tools() {
        let r = reg();
        let o = r.openai_tools();
        assert_eq!(o.len(), 1);
        assert_eq!(o[0]["function"]["name"], "echo");
        let a = r.anthropic_tools();
        assert_eq!(a[0]["name"], "echo");
    }

    #[test]
    fn levenshtein_basic() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("echo", "echo"), 0);
    }
}
