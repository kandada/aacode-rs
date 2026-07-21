// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Context compaction — round-aware compact view.
//!
//! Ported (simplified) from Python `utils/message_utils.py` +
//! `react_loop._compact_context`. A "round" starts at a user message and runs
//! until the next user message. When token budget is exceeded we keep the
//! protected first rounds + the last rounds, and replace the middle with a
//! single summary system message. tool_calls/tool pairs are never split
//! because they always live inside one round.

use crate::config::ContextConfig;
use crate::llm::types::ChatMessage;
use crate::session::estimate_tokens;

/// Estimate tokens of a whole message list.
pub fn estimate_messages_tokens(messages: &[ChatMessage]) -> usize {
    messages
        .iter()
        .map(|m| {
            let mut t = estimate_tokens(&m.content);
            if let Some(rc) = &m.reasoning_content {
                t += estimate_tokens(rc);
            }
            if let Some(tcs) = &m.tool_calls {
                for tc in tcs {
                    t += estimate_tokens(&tc.arguments) + estimate_tokens(&tc.name);
                }
            }
            t + 4 // per-message overhead
        })
        .sum()
}

/// Split messages into rounds. The leading system message(s) form round 0's
/// prefix; each subsequent `user` message begins a new round.
///
/// Returns (system_prefix, rounds) where rounds is a Vec of message slices.
fn split_into_rounds(messages: &[ChatMessage]) -> (Vec<ChatMessage>, Vec<Vec<ChatMessage>>) {
    let mut system_prefix = Vec::new();
    let mut rounds: Vec<Vec<ChatMessage>> = Vec::new();
    let mut idx = 0;

    // Leading system messages belong to the prefix.
    while idx < messages.len() && messages[idx].role == "system" {
        system_prefix.push(messages[idx].clone());
        idx += 1;
    }

    let mut current: Vec<ChatMessage> = Vec::new();
    for m in &messages[idx..] {
        if m.role == "user" && !current.is_empty() {
            rounds.push(std::mem::take(&mut current));
        }
        current.push(m.clone());
    }
    if !current.is_empty() {
        rounds.push(current);
    }
    (system_prefix, rounds)
}

/// Count how many rounds from the end contain a real user message, returning
/// the index of the round that is the Nth-from-last user round.
fn last_n_user_round_index(rounds: &[Vec<ChatMessage>], n: usize) -> Option<usize> {
    if n == 0 {
        return None;
    }
    let mut seen = 0;
    for i in (0..rounds.len()).rev() {
        if rounds[i].iter().any(|m| m.role == "user") {
            seen += 1;
            if seen == n {
                return Some(i);
            }
        }
    }
    None
}

/// Cached compaction state — keeps the compact-view **prefix byte-stable**
/// across iterations so the provider's KV/prefix cache keeps hitting.
///
/// Without this, every iteration would re-derive the middle boundary and
/// re-generate the summary from a shifting round list: the request prefix
/// changes every turn and the entire KV cache is invalidated exactly in the
/// long-context scenario where it matters most. (Mirrors the Python
/// `_compact_summary` / `_compact_summary_msg_count` caching.)
#[derive(Debug, Clone)]
pub struct CompactCache {
    /// The frozen summary text of the compacted middle.
    pub summary: String,
    /// Round index (in the stable round numbering) where the tail begins —
    /// rounds `[protect_first, tail_start)` are covered by `summary`.
    pub tail_start: usize,
    /// Re-compact (recompute boundary + summary, accepting one cache miss)
    /// when the view grows beyond this many tokens again.
    pub recompact_at: usize,
}

/// Build a compact view using (and maintaining) a [`CompactCache`] so the
/// view prefix stays stable across iterations. Returns
/// (view, was_compacted, token_count).
pub fn build_compact_view_cached(
    messages: &[ChatMessage],
    cfg: &ContextConfig,
    cache: &mut Option<CompactCache>,
) -> (Vec<ChatMessage>, bool, usize) {
    let total = estimate_messages_tokens(messages);

    // Fast path: under budget and no compaction has happened yet.
    if cache.is_none() && total <= cfg.compact_trigger_tokens {
        return (messages.to_vec(), false, total);
    }

    // Reuse the frozen boundary + summary while the compact view fits.
    if let Some(c) = cache.as_ref() {
        let (system_prefix, rounds) = split_into_rounds(messages);
        if c.tail_start <= rounds.len() {
            let mut view: Vec<ChatMessage> = Vec::new();
            view.extend(system_prefix.iter().cloned());
            for r in rounds.iter().take(cfg.protect_first_rounds) {
                view.extend(r.iter().cloned());
            }
            view.push(summary_message(&c.summary));
            for r in rounds.iter().skip(c.tail_start) {
                view.extend(r.iter().cloned());
            }
            let view_total = estimate_messages_tokens(&view);
            if view_total <= c.recompact_at {
                return (view, true, view_total);
            }
            // Tail grew too large again → fall through to re-compact
            // (one intentional cache miss, then stable again).
        }
    }

    // (Re)compute the boundary + summary and freeze them.
    let (view, compacted, tokens) = build_compact_view(messages, cfg, None);
    if compacted {
        // Recover the frozen boundary from the freshly built view: the
        // summary sits right after system prefix + protected rounds.
        let (_, rounds) = split_into_rounds(messages);
        let tail_start = tail_start_for(messages, cfg).unwrap_or(rounds.len());
        let summary = view
            .iter()
            .find(|m| m.role == "system" && m.content.starts_with("## 🧠 History Summary"))
            .map(|m| {
                m.content
                    .trim_start_matches("## 🧠 History Summary (compacted)\n")
                    .trim_end_matches("\n\n(Older turns compacted; continue based on recent context.)")
                    .to_string()
            })
            .unwrap_or_default();
        *cache = Some(CompactCache {
            summary,
            tail_start,
            // Re-freeze when the frozen view grows 50% beyond its size at
            // freeze time (relative to the actual view, not the trigger —
            // the compacted view may legitimately exceed a small trigger).
            recompact_at: (tokens + tokens / 2).max(cfg.compact_trigger_tokens),
        });
    }
    (view, compacted, tokens)
}

/// The summary system message (byte-stable formatting).
fn summary_message(summary: &str) -> ChatMessage {
    ChatMessage::system(format!(
        "## 🧠 History Summary (compacted)\n{summary}\n\n(Older turns compacted; continue based on recent context.)"
    ))
}

/// Where the tail would start for the current messages (same logic as
/// build_compact_view's boundary selection).
fn tail_start_for(messages: &[ChatMessage], cfg: &ContextConfig) -> Option<usize> {
    let (_, rounds) = split_into_rounds(messages);
    match last_n_user_round_index(&rounds, cfg.protect_last_user_rounds) {
        Some(user_idx) if user_idx > cfg.protect_first_rounds => Some(user_idx),
        Some(_) => None,
        None => {
            if rounds.len() <= cfg.protect_first_rounds + cfg.keep_last_rounds {
                None
            } else {
                Some(rounds.len().saturating_sub(cfg.keep_last_rounds))
            }
        }
    }
}

/// Build a compact view of the messages if over budget. Returns
/// (view, was_compacted, token_count). Does not mutate the input.
///
/// `summary` (if provided) is the cached AI-generated summary of the middle.
pub fn build_compact_view(
    messages: &[ChatMessage],
    cfg: &ContextConfig,
    summary: Option<&str>,
) -> (Vec<ChatMessage>, bool, usize) {
    let total = estimate_messages_tokens(messages);
    if total <= cfg.compact_trigger_tokens {
        return (messages.to_vec(), false, total);
    }

    let (system_prefix, rounds) = split_into_rounds(messages);
    let protect_first = cfg.protect_first_rounds;
    let keep_last = cfg.keep_last_rounds;
    let protect_user = cfg.protect_last_user_rounds;

    // Determine the boundary: summarize rounds before the last-N-user round.
    let cut_end = last_n_user_round_index(&rounds, protect_user);

    let (middle_range, tail_start) = match cut_end {
        Some(user_idx) => {
            // summarize rounds [protect_first, user_idx)
            if user_idx <= protect_first {
                // nothing to compact
                return (messages.to_vec(), false, total);
            }
            ((protect_first, user_idx), user_idx)
        }
        None => {
            if rounds.len() <= protect_first + keep_last {
                return (messages.to_vec(), false, total);
            }
            let start = protect_first;
            let end = rounds.len().saturating_sub(keep_last);
            ((start, end), end)
        }
    };

    let mut view: Vec<ChatMessage> = Vec::new();
    view.extend(system_prefix.iter().cloned());
    // protected first rounds
    for r in rounds.iter().take(protect_first) {
        view.extend(r.iter().cloned());
    }
    // summary system message for the middle
    let (ms, me) = middle_range;
    if me > ms {
        let summarized = summary.map(|s| s.to_string()).unwrap_or_else(|| {
            heuristic_summary(&rounds[ms..me])
        });
        view.push(ChatMessage::system(format!(
            "## 🧠 History Summary (compacted)\n{summarized}\n\n(Older turns compacted; continue based on recent context.)"
        )));
    }
    // tail rounds
    for r in rounds.iter().skip(tail_start) {
        view.extend(r.iter().cloned());
    }

    let new_total = estimate_messages_tokens(&view);
    (view, true, new_total)
}

/// A cheap fallback summary of the middle rounds (used when no AI summary is
/// cached): lists tool names invoked and truncated user asks.
pub fn heuristic_summary(middle: &[Vec<ChatMessage>]) -> String {
    let mut tools = Vec::new();
    let mut asks = Vec::new();
    for round in middle {
        for m in round {
            if m.role == "user" {
                let a: String = m.content.chars().take(80).collect();
                if !a.trim().is_empty() {
                    asks.push(a);
                }
            }
            if let Some(tcs) = &m.tool_calls {
                for tc in tcs {
                    tools.push(tc.name.clone());
                }
            }
        }
    }
    let mut out = String::new();
    if !asks.is_empty() {
        out.push_str("Earlier requests:\n");
        for a in asks.iter().take(10) {
            out.push_str(&format!("- {a}\n"));
        }
    }
    if !tools.is_empty() {
        out.push_str(&format!("Tools used: {}\n", dedup_join(&tools)));
    }
    if out.is_empty() {
        out.push_str("(prior conversation compacted)");
    }
    out
}

fn dedup_join(items: &[String]) -> String {
    let mut seen = std::collections::BTreeMap::new();
    for i in items {
        *seen.entry(i.clone()).or_insert(0usize) += 1;
    }
    seen.into_iter()
        .map(|(k, v)| format!("{k}×{v}"))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::ToolCall;

    fn cfg(trigger: usize) -> ContextConfig {
        ContextConfig {
            compact_trigger_tokens: trigger,
            protect_first_rounds: 1,
            keep_last_rounds: 2,
            protect_last_user_rounds: 1,
            ..Default::default()
        }
    }

    fn convo(rounds: usize) -> Vec<ChatMessage> {
        let mut v = vec![ChatMessage::system("SYS")];
        for i in 0..rounds {
            v.push(ChatMessage::user(format!("ask {i} {}", "x".repeat(200))));
            v.push(ChatMessage::assistant_with_tools(
                "",
                vec![ToolCall {
                    id: format!("c{i}"),
                    name: "run_shell".into(),
                    arguments: "{}".into(),
                }],
            ));
            v.push(ChatMessage::tool_result(format!("c{i}"), "y".repeat(200)));
        }
        v
    }

    #[test]
    fn no_compaction_below_trigger() {
        let msgs = convo(2);
        let (view, compacted, _) = build_compact_view(&msgs, &cfg(1_000_000), None);
        assert!(!compacted);
        assert_eq!(view.len(), msgs.len());
    }

    #[test]
    fn compaction_reduces_tokens() {
        let msgs = convo(10);
        let before = estimate_messages_tokens(&msgs);
        let (view, compacted, after) = build_compact_view(&msgs, &cfg(100), None);
        assert!(compacted);
        assert!(after < before);
        // system prefix preserved
        assert_eq!(view[0].role, "system");
        assert_eq!(view[0].content, "SYS");
    }

    #[test]
    fn tool_pairs_not_split() {
        let msgs = convo(10);
        let (view, _, _) = build_compact_view(&msgs, &cfg(100), None);
        // every assistant-with-tool_calls must be followed by a tool result
        for i in 0..view.len() {
            if view[i].tool_calls.is_some() {
                let has_result = view
                    .iter()
                    .skip(i + 1)
                    .any(|m| m.role == "tool");
                assert!(has_result, "tool_calls without following tool result");
            }
        }
    }

    #[test]
    fn cached_summary_used() {
        let msgs = convo(10);
        let (view, _, _) = build_compact_view(&msgs, &cfg(100), Some("MY SUMMARY"));
        assert!(view.iter().any(|m| m.content.contains("MY SUMMARY")));
    }

    #[test]
    fn heuristic_summary_lists_tools() {
        let msgs = convo(4);
        let (_, rounds) = split_into_rounds(&msgs);
        let s = heuristic_summary(&rounds);
        assert!(s.contains("run_shell"));
    }

    // ───────── KV/prefix-cache stability (provider prefix caching) ─────────

    /// Serialize a view for byte-stable comparison.
    fn fingerprint(view: &[ChatMessage]) -> Vec<String> {
        view.iter()
            .map(|m| {
                format!(
                    "{}|{}|{:?}|{:?}",
                    m.role,
                    m.content,
                    m.tool_calls.as_ref().map(|t| t.len()),
                    m.tool_call_id
                )
            })
            .collect()
    }

    #[test]
    fn kv_prefix_stable_without_compaction() {
        // Below the trigger the view is a passthrough: each iteration's
        // request must be a prefix-extension of the previous one.
        let mut cache = None;
        let mut msgs = convo(2);
        let (v1, c1, _) = build_compact_view_cached(&msgs, &cfg(1_000_000), &mut cache);
        assert!(!c1);
        msgs.push(ChatMessage::user("next question"));
        let (v2, _, _) = build_compact_view_cached(&msgs, &cfg(1_000_000), &mut cache);
        assert_eq!(
            fingerprint(&v1),
            fingerprint(&v2[..v1.len()]),
            "prefix must be stable"
        );
    }

    #[test]
    fn kv_prefix_stable_after_compaction() {
        // THE regression this cache exists for: once compaction triggers,
        // the summary + boundary must FREEZE so later iterations only append
        // — otherwise every long-context request is a full KV-cache miss.
        let mut cache = None;
        let mut msgs = convo(12);
        let (v1, c1, _) = build_compact_view_cached(&msgs, &cfg(10), &mut cache);
        assert!(c1, "should compact");
        assert!(cache.is_some(), "cache must be populated");

        // Simulate two more agent iterations (append-only growth).
        msgs.push(ChatMessage::user("follow-up 1"));
        let (v2, _, _) = build_compact_view_cached(&msgs, &cfg(10), &mut cache);
        assert!(v2.len() > v1.len());
        assert_eq!(
            fingerprint(&v1),
            fingerprint(&v2[..v1.len()]),
            "compacted view prefix must be byte-stable"
        );

        msgs.push(ChatMessage::user("follow-up 2"));
        let (v3, _, _) = build_compact_view_cached(&msgs, &cfg(10), &mut cache);
        assert_eq!(
            fingerprint(&v2),
            fingerprint(&v3[..v2.len()]),
            "second append must also extend the prefix"
        );
    }

    #[test]
    fn recompacts_when_tail_overflows() {
        // When the tail outgrows recompact_at, the boundary re-freezes (one
        // intentional cache miss) instead of letting the view grow unbounded.
        let mut cache = None;
        let mut msgs = convo(12);
        let (_, c1, _) = build_compact_view_cached(&msgs, &cfg(10), &mut cache);
        assert!(c1);
        let first_tail = cache.as_ref().unwrap().tail_start;

        // Grow the tail far beyond recompact_at (trigger=10 → recompact_at=15).
        for i in 0..30 {
            msgs.push(ChatMessage::user(format!("grow {i} {}", "z".repeat(300))));
        }
        let (view, c2, tokens) = build_compact_view_cached(&msgs, &cfg(10), &mut cache);
        assert!(c2);
        let new_tail = cache.as_ref().unwrap().tail_start;
        assert!(
            new_tail > first_tail,
            "boundary must advance on re-compaction ({first_tail} → {new_tail})"
        );
        // And the re-frozen view must again be smaller than the raw messages.
        assert!(tokens < estimate_messages_tokens(&msgs));
        assert!(view.iter().any(|m| m.content.contains("History Summary")));
    }

    #[test]
    fn tool_pairs_not_split_in_cached_view() {
        let mut cache = None;
        let mut msgs = convo(12);
        let _ = build_compact_view_cached(&msgs, &cfg(10), &mut cache);
        msgs.push(ChatMessage::user("more"));
        let (view, _, _) = build_compact_view_cached(&msgs, &cfg(10), &mut cache);
        // Every assistant-with-tool_calls must be directly followed by its
        // tool result (sanitize invariant holds through cached compaction).
        for i in 0..view.len() {
            if view[i].tool_calls.is_some() {
                assert!(
                    i + 1 < view.len() && view[i + 1].role == "tool",
                    "tool pair split at index {i}"
                );
            }
        }
    }
}
