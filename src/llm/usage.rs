// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Provider-usage normalization.
//!
//! OpenAI-compatible gateways (DeepSeek, Kimi, OpenAI) and Anthropic report
//! token counts with different field names. These helpers map both shapes to
//! the crate-wide [`Usage`] so callers never deal with provider-specific
//! fields.

use super::types::Usage;
use serde_json::Value;

fn as_u64(v: Option<&Value>) -> u64 {
    v.and_then(|v| {
        v.as_u64()
            .or_else(|| v.as_f64().map(|f| f.max(0.0) as u64))
            .or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
    })
    .unwrap_or(0)
}

fn field<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    v.as_object().and_then(|o| o.get(key))
}

/// Normalize an OpenAI-compatible `usage` object, including DeepSeek's
/// `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` and the
/// `prompt_tokens_details.cached_tokens` convention.
pub fn normalize_openai_usage(raw: &Value) -> Option<Usage> {
    let obj = raw.as_object()?;
    if obj.is_empty() {
        return None;
    }
    let input = as_u64(field(raw, "prompt_tokens"));
    let output = as_u64(field(raw, "completion_tokens"));
    let total = {
        let t = as_u64(field(raw, "total_tokens"));
        if t == 0 {
            input + output
        } else {
            t
        }
    };
    let cache_read = {
        let hit = as_u64(field(raw, "prompt_cache_hit_tokens"));
        if hit == 0 {
            as_u64(field(raw, "prompt_tokens_details").and_then(|d| field(d, "cached_tokens")))
        } else {
            hit
        }
    };
    let cache_miss = {
        let miss = as_u64(field(raw, "prompt_cache_miss_tokens"));
        if miss == 0 {
            input.saturating_sub(cache_read)
        } else {
            miss
        }
    };
    let reasoning =
        as_u64(field(raw, "completion_tokens_details").and_then(|d| field(d, "reasoning_tokens")));
    Some(Usage {
        input_tokens: input,
        output_tokens: output,
        total_tokens: total,
        cache_read_tokens: cache_read,
        cache_write_tokens: 0,
        cache_miss_tokens: cache_miss,
        reasoning_tokens: reasoning,
    })
}

/// Normalize an Anthropic `usage` object, folding cache reads/writes into the
/// reported input total (they are billed as input tokens).
pub fn normalize_anthropic_usage(raw: &Value) -> Option<Usage> {
    let obj = raw.as_object()?;
    if obj.is_empty() {
        return None;
    }
    let uncached = as_u64(field(raw, "input_tokens"));
    let cache_read = as_u64(field(raw, "cache_read_input_tokens"));
    let cache_write = as_u64(field(raw, "cache_creation_input_tokens"));
    let input = uncached + cache_read + cache_write;
    let output = as_u64(field(raw, "output_tokens"));
    Some(Usage {
        input_tokens: input,
        output_tokens: output,
        total_tokens: input + output,
        cache_read_tokens: cache_read,
        cache_write_tokens: cache_write,
        cache_miss_tokens: uncached,
        reasoning_tokens: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn openai_deepseek_cache_fields() {
        let u = normalize_openai_usage(&json!({
            "prompt_tokens": 1001,
            "completion_tokens": 50,
            "total_tokens": 1051,
            "prompt_cache_hit_tokens": 768,
            "prompt_cache_miss_tokens": 233
        }))
        .unwrap();
        assert_eq!(u.input_tokens, 1001);
        assert_eq!(u.output_tokens, 50);
        assert_eq!(u.cache_read_tokens, 768);
        assert_eq!(u.cache_miss_tokens, 233);
        assert!((u.cache_hit_rate() - 768.0 / 1001.0).abs() < 1e-9);
    }

    #[test]
    fn openai_details_fallback() {
        let u = normalize_openai_usage(&json!({
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 40},
            "completion_tokens_details": {"reasoning_tokens": 7}
        }))
        .unwrap();
        assert_eq!(u.cache_read_tokens, 40);
        assert_eq!(u.cache_miss_tokens, 60);
        assert_eq!(u.reasoning_tokens, 7);
        assert_eq!(u.total_tokens, 110);
    }

    #[test]
    fn anthropic_folds_cache_into_input() {
        let u = normalize_anthropic_usage(&json!({
            "input_tokens": 20,
            "cache_read_input_tokens": 700,
            "cache_creation_input_tokens": 30,
            "output_tokens": 12
        }))
        .unwrap();
        assert_eq!(u.input_tokens, 750);
        assert_eq!(u.cache_read_tokens, 700);
        assert_eq!(u.cache_write_tokens, 30);
        assert_eq!(u.cache_miss_tokens, 20);
        assert_eq!(u.total_tokens, 762);
    }

    #[test]
    fn empty_usage_is_none() {
        assert!(normalize_openai_usage(&json!({})).is_none());
        assert!(normalize_anthropic_usage(&json!({})).is_none());
        assert!(normalize_openai_usage(&json!(null)).is_none());
    }
}
