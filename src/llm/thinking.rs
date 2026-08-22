// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Anthropic 网关 thinking 启用的状态机控制
//!
//! 背景：Anthropic 协议的 thinking 配置有两种 type：
//! - `adaptive`：无需 budget_tokens，Claude 4.6+ 主流，未来兼容
//! - `enabled`：必须有 budget_tokens (≥1024 且 < max_tokens)，Claude 4.5 及更早
//!
//! 不同 Anthropic 兼容代理/底层模型对二者的支持差异较大。通过 strict 的
//! 错误判定实现"乐观尝试 + 失败降级"的状态机，**零模型硬编码**。
//!
//! 策略顺序：`adaptive` → `enabled` → `none`，命中后该模型会话内
//! 不再尝试更高优先级模式。

use std::sync::{Mutex, OnceLock};
use std::collections::HashMap;

/// thinking 模式三态（按推荐度排序）
pub const THINKING_MODE_ADAPTIVE: &str = "adaptive"; // 无需 budget_tokens
pub const THINKING_MODE_ENABLED: &str = "enabled"; // 需 budget_tokens
pub const THINKING_MODE_NONE: &str = "none"; // 不带 thinking

pub const THINKING_MODE_ORDER: &[&str] = &[THINKING_MODE_ADAPTIVE, THINKING_MODE_ENABLED, THINKING_MODE_NONE];

/// 默认 budget_tokens（仅 enabled 模式使用）
/// 30k 平衡了 thinking 深度与响应空间，且避开 Anthropic 文档警告的 32k 警戒线。
pub const DEFAULT_BUDGET_TOKENS: u64 = 30000;

/// 根据模式返回 `thinking=` 请求参数字。
///
/// - `adaptive`: `{"type": "adaptive"}`
/// - `enabled`: `{"type": "enabled", "budget_tokens": 30000}`
/// - `none` 或未知: `None`（调用方据此跳过 thinking= 字段）
pub fn thinking_kw_for_mode(mode: &str) -> Option<serde_json::Value> {
    match mode {
        THINKING_MODE_ADAPTIVE => Some(serde_json::json!({"type": "adaptive"})),
        THINKING_MODE_ENABLED => Some(serde_json::json!({
            "type": "enabled",
            "budget_tokens": DEFAULT_BUDGET_TOKENS,
        })),
        _ => None,
    }
}

/// 状态机的下一步。
///
/// Returns:
/// - 下个模式（`adaptive` → `enabled` → `none`）
/// - `None` 表示已是终态（`none`）或未知模式
pub fn next_mode(current_mode: &str) -> Option<&'static str> {
    match current_mode {
        THINKING_MODE_ADAPTIVE => Some(THINKING_MODE_ENABLED),
        THINKING_MODE_ENABLED => Some(THINKING_MODE_NONE),
        _ => None,
    }
}

/// 严格判断：API 是否因该 thinking 模式不被支持而拒绝。
///
/// 仅当错误信息明确指向 thinking 配置时才返回 true，避免误吞：
/// - 401 authentication_error（认证错误）
/// - 404 model_not_found（模型不存在）
/// - 429 rate_limit_exceeded（配额错误）
/// - 其他通用 400
///
/// 关键边界：若错误信息含其他具体 mode 名（如 `"thinking.type.adaptive"` 被拒
/// 而我们正在尝试 `enabled`），不应判为"当前 mode 被拒"——那是别的 mode 的事。
pub fn is_thinking_mode_rejected(error_msg: &str, mode: &str) -> bool {
    let msg = error_msg.to_lowercase();

    // 1. Anthropic 官方 precise 错误格式（最可靠）
    let needle = format!("\"thinking.type.{}\"", mode);
    if msg.contains(&needle) {
        return true;
    }

    // 2. 检查是否提到其他具体 mode 名（错误指向其他 mode，不是当前）
    for &other_mode in THINKING_MODE_ORDER {
        if other_mode == mode {
            continue;
        }
        let other_needle = format!("\"thinking.type.{}\"", other_mode);
        if msg.contains(&other_needle) {
            return false;
        }
    }

    // 3. 第三方代理 unknown field（不含具体 mode 名 → 整个 thinking 被拒）
    if msg.contains("unknown") && msg.contains("thinking") {
        return true;
    }

    // 4. 通用兜底：not supported + thinking
    if msg.contains("not supported") && msg.contains("thinking") {
        return true;
    }

    false
}

/// 会话级 per-model thinking 模式缓存。
///
/// 用 `Mutex<HashMap>` 保护并发访问。进程级生命周期——同一进程内所有 client 共享。
/// 重启后清空，最坏代价：每个模型第一次请求一次重试。
static THINKING_CACHE: OnceLock<Mutex<HashMap<String, &'static str>>> = OnceLock::new();

fn cache() -> &'static Mutex<HashMap<String, &'static str>> {
    THINKING_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// 读取模型的首选 thinking 模式（命中缓存返回缓存值，否则返回 adaptive）。
pub fn get_cached_mode(model_name: &str) -> &'static str {
    let m = cache().lock().unwrap_or_else(|e| e.into_inner());
    m.get(model_name).copied().unwrap_or(THINKING_MODE_ADAPTIVE)
}

/// 将模型的成功模式写入缓存。
pub fn cache_mode(model_name: &str, mode: &'static str) {
    let mut m = cache().lock().unwrap_or_else(|e| e.into_inner());
    m.insert(model_name.to_string(), mode);
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── thinking_kw_for_mode ───

    #[test]
    fn kw_adaptive_has_no_budget() {
        let kw = thinking_kw_for_mode(THINKING_MODE_ADAPTIVE).unwrap();
        assert_eq!(kw["type"], "adaptive");
        assert!(kw.get("budget_tokens").is_none());
    }

    #[test]
    fn kw_enabled_has_budget() {
        let kw = thinking_kw_for_mode(THINKING_MODE_ENABLED).unwrap();
        assert_eq!(kw["type"], "enabled");
        assert_eq!(kw["budget_tokens"], DEFAULT_BUDGET_TOKENS);
        assert!(kw["budget_tokens"].as_u64().unwrap() >= 1024);
    }

    #[test]
    fn kw_none_is_none() {
        assert!(thinking_kw_for_mode(THINKING_MODE_NONE).is_none());
    }

    #[test]
    fn kw_unknown_is_none() {
        assert!(thinking_kw_for_mode("garbage").is_none());
        assert!(thinking_kw_for_mode("").is_none());
    }

    #[test]
    fn budget_tokens_default_is_30000() {
        assert_eq!(DEFAULT_BUDGET_TOKENS, 30000);
        assert!(DEFAULT_BUDGET_TOKENS < 32000); // 避开 Anthropic 32k 警戒线
        assert!(DEFAULT_BUDGET_TOKENS < 96000); // 远低于 max_tokens 默认
    }

    // ─── next_mode ───

    #[test]
    fn next_mode_adaptive_to_enabled() {
        assert_eq!(next_mode(THINKING_MODE_ADAPTIVE), Some(THINKING_MODE_ENABLED));
    }

    #[test]
    fn next_mode_enabled_to_none() {
        assert_eq!(next_mode(THINKING_MODE_ENABLED), Some(THINKING_MODE_NONE));
    }

    #[test]
    fn next_mode_none_is_terminal() {
        assert_eq!(next_mode(THINKING_MODE_NONE), None);
    }

    #[test]
    fn next_mode_unknown_is_none() {
        assert_eq!(next_mode("foo"), None);
        assert_eq!(next_mode(""), None);
    }

    #[test]
    fn next_mode_full_progression() {
        let mut mode = THINKING_MODE_ADAPTIVE;
        let mut steps = vec![mode];
        loop {
            match next_mode(mode) {
                Some(n) => { steps.push(n); mode = n; }
                None => break,
            }
        }
        assert_eq!(steps, vec![THINKING_MODE_ADAPTIVE, THINKING_MODE_ENABLED, THINKING_MODE_NONE]);
    }

    #[test]
    fn next_mode_no_infinite_loop() {
        let mut mode = THINKING_MODE_ADAPTIVE;
        for _ in 0..100 {
            match next_mode(mode) {
                Some(n) => mode = n,
                None => break,
            }
        }
        assert_eq!(mode, THINKING_MODE_NONE);
    }

    // ─── is_thinking_mode_rejected (正向) ───

    #[test]
    fn reject_official_adaptive() {
        let err = r#"Error code: 400 - {"type":"error","error":{"type":"invalid_request_error","message":"\"thinking.type.adaptive\" is not supported"}}"#;
        assert!(is_thinking_mode_rejected(err, THINKING_MODE_ADAPTIVE));
    }

    #[test]
    fn reject_official_enabled() {
        let err = r#"400 "thinking.type.enabled" is not supported"#;
        assert!(is_thinking_mode_rejected(err, THINKING_MODE_ENABLED));
    }

    #[test]
    fn reject_third_party_unknown_field() {
        let err = "400 Bad Request: unknown field: thinking";
        assert!(is_thinking_mode_rejected(err, THINKING_MODE_ADAPTIVE));
        assert!(is_thinking_mode_rejected(err, THINKING_MODE_ENABLED));
    }

    #[test]
    fn reject_generic_not_supported() {
        let err = "thinking is not supported on this model";
        assert!(is_thinking_mode_rejected(err, THINKING_MODE_ADAPTIVE));
    }

    #[test]
    fn reject_case_insensitive() {
        let err = "THINKING.TYPE.ENABLED IS NOT SUPPORTED";
        assert!(is_thinking_mode_rejected(err, THINKING_MODE_ENABLED));
    }

    #[test]
    fn reject_specific_mode_must_match() {
        // adaptive 错误信息不应被 enabled 模式误判
        let err = r#"400 "thinking.type.adaptive" is not supported"#;
        assert!(!is_thinking_mode_rejected(err, THINKING_MODE_ENABLED));
        // 反向亦然
        let err = r#"400 "thinking.type.enabled" is not supported"#;
        assert!(!is_thinking_mode_rejected(err, THINKING_MODE_ADAPTIVE));
    }

    // ─── is_thinking_mode_rejected (反向，避免误吞) ───

    #[test]
    fn not_reject_auth_error() {
        let err = "401 authentication_error: invalid api key";
        assert!(!is_thinking_mode_rejected(err, THINKING_MODE_ADAPTIVE));
    }

    #[test]
    fn not_reject_model_not_found() {
        let err = "404 model_not_found: unknown model";
        assert!(!is_thinking_mode_rejected(err, THINKING_MODE_ADAPTIVE));
    }

    #[test]
    fn not_reject_rate_limit() {
        let err = "429 rate_limit_exceeded: too many requests";
        assert!(!is_thinking_mode_rejected(err, THINKING_MODE_ADAPTIVE));
    }

    #[test]
    fn not_reject_quota() {
        let err = "insufficient_quota: please check your plan";
        assert!(!is_thinking_mode_rejected(err, THINKING_MODE_ADAPTIVE));
    }

    #[test]
    fn not_reject_generic_400() {
        let err = "400 Bad Request: invalid parameter: temperature";
        assert!(!is_thinking_mode_rejected(err, THINKING_MODE_ADAPTIVE));
    }

    #[test]
    fn not_reject_network_error() {
        let err = "ConnectionError: connection reset";
        assert!(!is_thinking_mode_rejected(err, THINKING_MODE_ADAPTIVE));
    }

    #[test]
    fn not_reject_empty_message() {
        let err = "";
        assert!(!is_thinking_mode_rejected(err, THINKING_MODE_ADAPTIVE));
    }

    #[test]
    fn not_reject_thinking_mentioned_but_supported() {
        let err = "thinking stream interrupted";
        assert!(!is_thinking_mode_rejected(err, THINKING_MODE_ADAPTIVE));
    }

    #[test]
    fn not_reject_unrelated_thinking_word() {
        let err = "I was thinking about the problem and got 500";
        assert!(!is_thinking_mode_rejected(err, THINKING_MODE_ADAPTIVE));
    }

    // ─── 缓存 API ───

    #[test]
    fn cache_default_adaptive() {
        assert_eq!(get_cached_mode("never-seen-model"), THINKING_MODE_ADAPTIVE);
    }

    #[test]
    fn cache_round_trip() {
        cache_mode("test-model-cache-rt", THINKING_MODE_ENABLED);
        assert_eq!(get_cached_mode("test-model-cache-rt"), THINKING_MODE_ENABLED);
        cache_mode("test-model-cache-rt", THINKING_MODE_NONE);
        assert_eq!(get_cached_mode("test-model-cache-rt"), THINKING_MODE_NONE);
    }

    // ─── 模块 sanity ───

    #[test]
    fn mode_order_is_correct() {
        assert_eq!(
            THINKING_MODE_ORDER,
            &[THINKING_MODE_ADAPTIVE, THINKING_MODE_ENABLED, THINKING_MODE_NONE]
        );
    }

    #[test]
    fn mode_constants_are_unique() {
        use std::collections::HashSet;
        let set: HashSet<&str> = THINKING_MODE_ORDER.iter().copied().collect();
        assert_eq!(set.len(), 3);
    }
}