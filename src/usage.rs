// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Append-only provider-usage recording.
//!
//! Writes one JSONL line per model call under `.aacode/usage/usage.jsonl`.
//! Only token counts and request metadata are stored — never prompt or
//! response text. Telemetry must never break a task, so all errors are
//! swallowed.
//!
//! Env: `AACODE_USAGE_LOG=0` disables; `AACODE_USAGE_VERBOSE=1` prints one
//! compact line per call.

use serde_json::Value;
use std::io::Write;
use std::path::{Path, PathBuf};

pub struct UsageTracker {
    project_path: PathBuf,
    enabled: bool,
    pub verbose: bool,
}

impl UsageTracker {
    pub fn new(project_path: &Path) -> Self {
        let enabled = std::env::var("AACODE_USAGE_LOG")
            .map(|v| v != "0")
            .unwrap_or(true);
        let verbose = std::env::var("AACODE_USAGE_VERBOSE")
            .map(|v| v == "1")
            .unwrap_or(false);
        UsageTracker {
            project_path: project_path.to_path_buf(),
            enabled,
            verbose,
        }
    }

    /// Append one event (a `ts` field is added). Returns false when disabled or
    /// on any I/O error.
    pub fn record(&self, event: &Value) -> bool {
        if !self.enabled {
            return false;
        }
        let dir = self.project_path.join(".aacode").join("usage");
        if std::fs::create_dir_all(&dir).is_err() {
            return false;
        }
        let path = dir.join("usage.jsonl");
        let mut line = event.clone();
        if let Some(obj) = line.as_object_mut() {
            obj.insert("ts".into(), Value::String(crate::session::now_iso_ms()));
        }
        let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        else {
            return false;
        };
        f.write_all(line.to_string().as_bytes()).is_ok() && f.write_all(b"\n").is_ok()
    }
}

/// Build the JSONL event for one model call. `usage` is the normalized usage
/// (if the provider reported it); pass `None` when missing.
#[allow(clippy::too_many_arguments)]
pub fn usage_event(
    session_id: &str,
    run_id: &str,
    iteration: u32,
    gateway: &str,
    model: &str,
    estimated_input_tokens: usize,
    messages_count: usize,
    compacted: bool,
    usage: Option<&crate::llm::Usage>,
) -> Value {
    let mut event = serde_json::json!({
        "source": "main",
        "session_id": session_id,
        "run_id": run_id,
        "iteration": iteration,
        "gateway": gateway,
        "model": model,
        "stream": true,
        "status": "ok",
        "usage_source": if usage.is_some() { "provider" } else { "missing" },
        "estimated_input_tokens": estimated_input_tokens,
        "context_tokens_estimate": estimated_input_tokens,
        "messages_count": messages_count,
        "compacted": compacted,
    });
    if let Some(u) = usage {
        event["input_tokens"] = serde_json::json!(u.input_tokens);
        event["output_tokens"] = serde_json::json!(u.output_tokens);
        event["total_tokens"] = serde_json::json!(u.total_tokens);
        event["cache_read_tokens"] = serde_json::json!(u.cache_read_tokens);
        event["cache_write_tokens"] = serde_json::json!(u.cache_write_tokens);
        event["cache_miss_tokens"] = serde_json::json!(u.cache_miss_tokens);
        event["reasoning_tokens"] = serde_json::json!(u.reasoning_tokens);
        event["cache_hit_rate"] = serde_json::json!(u.cache_hit_rate());
    }
    event
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Usage;

    #[test]
    fn event_includes_cache_hit_rate() {
        let u = Usage {
            input_tokens: 1000,
            output_tokens: 100,
            total_tokens: 1100,
            cache_read_tokens: 900,
            cache_write_tokens: 0,
            cache_miss_tokens: 100,
            reasoning_tokens: 5,
        };
        let ev = usage_event("s", "r", 1, "openai", "m", 999, 10, false, Some(&u));
        assert_eq!(ev["usage_source"], "provider");
        assert!((ev["cache_hit_rate"].as_f64().unwrap() - 0.9).abs() < 1e-9);
        assert_eq!(ev["input_tokens"], 1000);
    }

    #[test]
    fn event_without_usage_is_marked_missing() {
        let ev = usage_event("s", "r", 1, "openai", "m", 10, 2, false, None);
        assert_eq!(ev["usage_source"], "missing");
        assert!(ev.get("cache_hit_rate").is_none());
    }

    #[test]
    fn records_jsonl_line() {
        let dir = std::env::temp_dir().join(format!(
            "aacode_usage_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let tracker = UsageTracker::new(&dir);
        let ev = usage_event("s", "r", 1, "openai", "m", 10, 2, false, None);
        assert!(tracker.record(&ev));
        let text =
            std::fs::read_to_string(dir.join(".aacode").join("usage").join("usage.jsonl")).unwrap();
        assert!(text.contains("\"usage_source\":\"missing\""), "{text}");
        assert!(text.contains("\"ts\""), "{text}");
        assert_eq!(text.lines().count(), 1);
    }
}
