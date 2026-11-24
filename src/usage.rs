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
pub fn usage_cost(usage: Option<&crate::llm::Usage>, pricing: Option<&Value>) -> Option<f64> {
    let usage = usage?;
    let pricing = pricing?;
    let per = 1_000_000.0f64;
    let val = |name: &str| {
        pricing
            .get(name)
            .and_then(|v| {
                v.as_f64()
                    .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
            })
            .unwrap_or(0.0)
    };
    let input_price = if val("input") > 0.0 {
        val("input")
    } else {
        val("prompt")
    };
    let cache_read_price = if pricing.get("cache_read").is_some() {
        val("cache_read")
    } else {
        input_price * 0.1
    };
    let cache_write_price = if pricing.get("cache_write").is_some() {
        val("cache_write")
    } else {
        input_price
    };
    let output_price = if pricing.get("output").is_some() {
        val("output")
    } else {
        input_price * 4.0
    };
    let cost = (usage.cache_miss_tokens as f64) * input_price
        + (usage.cache_read_tokens as f64) * cache_read_price
        + (usage.cache_write_tokens as f64) * cache_write_price
        + (usage.output_tokens as f64) * output_price;
    Some(cost / per)
}

pub fn usage_event(
    session_id: &str,
    run_id: &str,
    iteration: u32,
    gateway: &str,
    model: &str,
    estimated_input_tokens: usize,
    messages_count: usize,
    compacted: bool,
    pricing: Option<&Value>,
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
        event["prompt_total"] = serde_json::json!(u.input_tokens);
        event["uncached_input"] = serde_json::json!(u.cache_miss_tokens);
        event["cache_read"] = serde_json::json!(u.cache_read_tokens);
        event["cache_write"] = serde_json::json!(u.cache_write_tokens);
        event["output"] = serde_json::json!(u.output_tokens);
        event["reasoning"] = serde_json::json!(u.reasoning_tokens);
        event["total"] = serde_json::json!(u.total_tokens);
        if let Some(cost) = usage_cost(Some(u), pricing) {
            event["cost"] = serde_json::json!(cost);
        }
    }
    event
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Usage;

    #[test]
    fn event_includes_canonical_fields_and_cost() {
        let u = Usage {
            input_tokens: 1000,
            output_tokens: 100,
            total_tokens: 1100,
            cache_read_tokens: 900,
            cache_write_tokens: 0,
            cache_miss_tokens: 100,
            reasoning_tokens: 5,
        };
        let pricing = serde_json::json!({"input": 1.0, "cache_read": 0.1, "output": 2.0});
        let ev = usage_event(
            "s",
            "r",
            1,
            "openai",
            "m",
            999,
            10,
            false,
            Some(&pricing),
            Some(&u),
        );
        assert_eq!(ev["prompt_total"], 1000);
        assert_eq!(ev["uncached_input"], 100);
        assert_eq!(ev["cache_read"], 900);
        assert_eq!(ev["output"], 100);
        assert!((ev["cost"].as_f64().unwrap() - 0.00039).abs() < 1e-9);
    }

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
        let ev = usage_event("s", "r", 1, "openai", "m", 999, 10, false, None, Some(&u));
        assert_eq!(ev["usage_source"], "provider");
        assert!((ev["cache_hit_rate"].as_f64().unwrap() - 0.9).abs() < 1e-9);
        assert_eq!(ev["input_tokens"], 1000);
    }

    #[test]
    fn event_without_usage_is_marked_missing() {
        let ev = usage_event("s", "r", 1, "openai", "m", 10, 2, false, None, None);
        assert_eq!(ev["usage_source"], "missing");
        assert!(ev.get("cache_hit_rate").is_none());
    }

    #[test]
    fn cost_uses_prompt_alias_and_default_rates() {
        let u = Usage {
            input_tokens: 4_000_000,
            output_tokens: 1_000_000,
            total_tokens: 5_000_000,
            cache_read_tokens: 1_000_000,
            cache_write_tokens: 1_000_000,
            cache_miss_tokens: 1_000_000,
            reasoning_tokens: 0,
        };
        // Omitted cache_read/cache_write/output fall back to 10% / 100% / 4x.
        let cost = usage_cost(Some(&u), Some(&serde_json::json!({"prompt": 2.0}))).unwrap();
        assert!((cost - (2.0 + 0.2 + 2.0 + 8.0)).abs() < 1e-9, "cost={cost}");
    }

    #[test]
    fn cost_accepts_string_prices_and_explicit_zero_cache_write() {
        let u = Usage {
            input_tokens: 1_000_000,
            output_tokens: 0,
            total_tokens: 1_000_000,
            cache_read_tokens: 1_000_000,
            cache_write_tokens: 1_000_000,
            cache_miss_tokens: 1_000_000,
            reasoning_tokens: 0,
        };
        let cost = usage_cost(
            Some(&u),
            Some(&serde_json::json!({
                "input": "1.5",
                "cache_read": "0.15",
                "cache_write": 0,
            })),
        )
        .unwrap();
        // cache_write explicitly 0 must not fall back to input price.
        assert!((cost - (1.5 + 0.15)).abs() < 1e-9, "cost={cost}");
    }

    #[test]
    fn cost_is_none_without_usage_or_pricing() {
        let u = Usage::default();
        let pricing = serde_json::json!({"input": 1.0});
        assert!(usage_cost(None, Some(&pricing)).is_none());
        assert!(usage_cost(Some(&u), None).is_none());
    }

    #[test]
    fn cost_zero_tokens_is_zero() {
        let cost = usage_cost(
            Some(&Usage::default()),
            Some(&serde_json::json!({"input": 10.0, "output": 20.0})),
        )
        .unwrap();
        assert_eq!(cost, 0.0);
    }

    #[test]
    fn event_with_zero_prompt_reports_zero_cache_rate() {
        let u = Usage {
            input_tokens: 0,
            output_tokens: 5,
            total_tokens: 5,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            cache_miss_tokens: 0,
            reasoning_tokens: 0,
        };
        let ev = usage_event("s", "r", 1, "openai", "m", 0, 0, false, None, Some(&u));
        assert_eq!(ev["cache_hit_rate"], 0.0);
        assert_eq!(ev["output"], 5);
        assert_eq!(ev["prompt_total"], 0);
    }

    #[test]
    fn event_missing_usage_keeps_estimates_and_marks_missing() {
        let ev = usage_event(
            "s",
            "r",
            2,
            "anthropic",
            "claude",
            1234,
            7,
            true,
            None,
            None,
        );
        assert_eq!(ev["usage_source"], "missing");
        assert_eq!(ev["estimated_input_tokens"], 1234);
        assert_eq!(ev["messages_count"], 7);
        assert_eq!(ev["compacted"], true);
        assert!(ev.get("cost").is_none());
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
        let ev = usage_event("s", "r", 1, "openai", "m", 10, 2, false, None, None);
        assert!(tracker.record(&ev));
        let text =
            std::fs::read_to_string(dir.join(".aacode").join("usage").join("usage.jsonl")).unwrap();
        assert!(text.contains("\"usage_source\":\"missing\""), "{text}");
        assert!(text.contains("\"ts\""), "{text}");
        assert_eq!(text.lines().count(), 1);
    }
}
