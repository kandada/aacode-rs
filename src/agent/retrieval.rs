// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Retrieval-style context control (replaces summary compaction).
//!
//! Instead of summarizing and deleting history, the full `messages` list is
//! kept intact. The model is sent a *view* that starts at a frozen **cutoff**;
//! older messages stay retrievable through `search_history` / `read_history`.
//! The cutoff only moves forward and only when the view exceeds the trigger,
//! which keeps the request prefix byte-stable for provider prompt caching.

use crate::agent::compact::estimate_messages_tokens;
use crate::config::ContextConfig;
use crate::llm::types::ChatMessage;
use crate::session::now_iso;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

pub const CHECKPOINT_OPEN: &str = "<conversation-checkpoint>";
pub const CHECKPOINT_CLOSE: &str = "</conversation-checkpoint>";

/// Persisted model-view cutoff for one session.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ContextCheckpoint {
    pub cutoff_index: usize,
    #[serde(default)]
    pub stub: String,
    #[serde(default)]
    pub session_id: String,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub keep_tokens: usize,
}

/// A real user turn (excludes host-injected messages and checkpoint stubs).
pub fn is_real_user_message(m: &ChatMessage) -> bool {
    m.role == "user"
        && !m.content.starts_with("[System]")
        && !m.content.starts_with(CHECKPOINT_OPEN)
}

fn real_user_indices(messages: &[ChatMessage]) -> Vec<usize> {
    messages
        .iter()
        .enumerate()
        .filter(|(_, m)| is_real_user_message(m))
        .map(|(i, _)| i)
        .collect()
}

/// A round starts at the system prompt or any non-tool message.
fn round_start_indices(messages: &[ChatMessage]) -> Vec<usize> {
    messages
        .iter()
        .enumerate()
        .filter(|(i, m)| *i == 0 || m.role != "tool")
        .map(|(i, _)| i)
        .collect()
}

fn next_index(indices: &[usize], value: usize) -> Option<usize> {
    indices.iter().copied().find(|i| *i > value)
}

fn truncate_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn estimate_view_tokens(
    messages: &[ChatMessage],
    cutoff: usize,
    latest_user: Option<usize>,
    stub_tokens: usize,
) -> usize {
    if cutoff <= 0 {
        return estimate_messages_tokens(messages);
    }
    let mut view: Vec<ChatMessage> = Vec::new();
    if let Some(first) = messages.first() {
        view.push(first.clone());
    }
    if stub_tokens > 0 {
        view.push(ChatMessage::user("x".repeat(stub_tokens * 4)));
    }
    if let Some(lu) = latest_user {
        if cutoff > lu && lu < messages.len() {
            view.push(messages[lu].clone());
            let mut i = lu + 1;
            while i < cutoff && messages.get(i).map(|m| m.role.as_str()) == Some("system") {
                view.push(messages[i].clone());
                i += 1;
            }
        }
    }
    view.extend_from_slice(&messages[cutoff.min(messages.len())..]);
    estimate_messages_tokens(&view)
}

/// Choose where the model view should start. Never moves the cutoff backwards.
///
/// Returns `(cutoff, latest_user_index)`.
pub fn compute_retrieval_cutoff(
    messages: &[ChatMessage],
    current_cutoff: usize,
    keep_tokens: usize,
    keep_recent_user_messages: usize,
    keep_min_rounds: usize,
) -> (usize, Option<usize>) {
    if messages.is_empty() {
        return (0, None);
    }
    let count = messages.len();
    let current = current_cutoff.min(count);
    let users = real_user_indices(messages);
    let starts = round_start_indices(messages);
    let latest_user = users.last().copied();

    let Some(latest_user) = latest_user else {
        let candidate = if starts.len() >= keep_min_rounds {
            starts[starts.len() - keep_min_rounds]
        } else {
            0
        };
        return (current.max(candidate), None);
    };

    let mut candidate = if users.len() >= keep_recent_user_messages {
        users[users.len() - keep_recent_user_messages]
    } else {
        users[0]
    };
    candidate = candidate.max(current);

    if starts.len() >= keep_min_rounds {
        candidate = current.max(candidate.min(starts[starts.len() - keep_min_rounds]));
    }

    // Drop whole older user rounds until the view fits the budget.
    while candidate < latest_user {
        let tokens = estimate_view_tokens(messages, candidate, Some(latest_user), 2000);
        if tokens <= keep_tokens {
            break;
        }
        candidate = next_index(&users, candidate).unwrap_or(latest_user);
    }

    if candidate < latest_user {
        let tokens = estimate_view_tokens(messages, candidate, Some(latest_user), 2000);
        if tokens <= keep_tokens {
            return (candidate, Some(latest_user));
        }
    }

    // The current task alone is too large: keep the latest request plus the
    // newest rounds, and make the middle retrievable history.
    candidate = candidate.max(latest_user);
    let tail_starts: Vec<usize> = starts
        .iter()
        .copied()
        .filter(|i| *i > latest_user)
        .collect();
    let mut chosen_tail = latest_user;
    for start in tail_starts.iter().rev() {
        let tokens = estimate_view_tokens(messages, *start, Some(latest_user), 2000);
        if tokens <= keep_tokens {
            chosen_tail = *start;
        } else {
            break;
        }
    }
    if starts.len() >= keep_min_rounds && starts[starts.len() - keep_min_rounds] > latest_user {
        let min_start = starts[starts.len() - keep_min_rounds];
        if chosen_tail > latest_user {
            chosen_tail = chosen_tail.min(min_start);
        }
    }

    (candidate.max(chosen_tail), Some(latest_user))
}

/// Render the model view without modifying the stored messages.
pub fn render_retrieval_view(
    messages: &[ChatMessage],
    cutoff: usize,
    latest_user: Option<usize>,
    stub: &str,
) -> Vec<ChatMessage> {
    let mut cutoff = cutoff.min(messages.len());
    // Never start a view with an orphaned tool result.
    while cutoff < messages.len() && messages[cutoff].role == "tool" {
        cutoff += 1;
    }

    let mut view: Vec<ChatMessage> = Vec::new();
    if let Some(first) = messages.first() {
        view.push(first.clone());
    }
    view.push(ChatMessage::user(stub));

    if let Some(lu) = latest_user {
        if cutoff > lu && lu < messages.len() {
            view.push(messages[lu].clone());
            let mut i = lu + 1;
            while i < cutoff && messages.get(i).map(|m| m.role.as_str()) == Some("system") {
                view.push(messages[i].clone());
                i += 1;
            }
        }
    }
    view.extend_from_slice(&messages[cutoff..]);
    view
}

fn pending_todos(project: &Path, session_id: &str) -> Vec<String> {
    let dir = project.join(".aacode").join("todos");
    let candidates = [
        dir.join(format!("todo_{session_id}.md")),
        dir.join("todo.md"),
    ];
    for path in candidates {
        if let Ok(text) = std::fs::read_to_string(&path) {
            let items: Vec<String> = text
                .lines()
                .filter_map(|line| {
                    line.trim()
                        .strip_prefix("- [ ]")
                        .map(|rest| truncate_chars(rest.trim(), 120))
                })
                .take(10)
                .collect();
            if !items.is_empty() {
                return items;
            }
        }
    }
    Vec::new()
}

fn last_error(messages: &[ChatMessage]) -> Option<String> {
    const MARKERS: [&str; 6] = [
        "error",
        "failed",
        "traceback",
        "exception",
        "❌",
        "execution error",
    ];
    for m in messages.iter().rev() {
        if m.role != "tool" && m.role != "assistant" {
            continue;
        }
        let lower = m.content.to_lowercase();
        if MARKERS.iter().any(|k| lower.contains(k)) {
            return Some(truncate_chars(
                &m.content.replace('\n', " ").trim().to_string(),
                300,
            ));
        }
    }
    None
}

/// Build the compact checkpoint stub that replaces omitted history.
pub fn build_checkpoint_stub(
    messages: &[ChatMessage],
    cutoff: usize,
    latest_user: Option<usize>,
    project: &Path,
    session_id: &str,
) -> String {
    let cutoff = cutoff.min(messages.len());
    let mut lines = vec![CHECKPOINT_OPEN.to_string()];
    if cutoff > 0 {
        lines.push(format!("Older messages omitted: messages 1-{cutoff}."));
    }
    lines.push(
        "Full history is persisted. Use search_history / read_history to retrieve older context. \
         Never cat the session JSON."
            .to_string(),
    );
    if !session_id.is_empty() {
        lines.push(format!("Session: {session_id}"));
    }

    let todos = pending_todos(project, session_id);
    if !todos.is_empty() {
        lines.push("Unfinished todos:".to_string());
        lines.extend(todos.into_iter().map(|t| format!("- {t}")));
    }

    if let Some(err) = last_error(&messages[..cutoff]) {
        lines.push(format!("Last error: {err}"));
    }

    let mut earlier = Vec::new();
    for (index, message) in messages[..cutoff].iter().enumerate() {
        if !is_real_user_message(message) {
            continue;
        }
        if Some(index) == latest_user {
            continue;
        }
        let text = message.content.replace('\n', " ");
        earlier.push(format!("- [{index}] {}", truncate_chars(text.trim(), 80)));
    }
    if !earlier.is_empty() {
        lines.push("Earlier user messages:".to_string());
        let start = earlier.len().saturating_sub(10);
        lines.extend(earlier[start..].iter().cloned());
    }

    lines.push(CHECKPOINT_CLOSE.to_string());
    let stub = lines.join("\n");
    if stub.chars().count() > 6000 {
        format!("{}\n[truncated]", truncate_chars(&stub, 6000))
    } else {
        stub
    }
}

/// In-memory retrieval state carried across iterations of one run.
#[derive(Debug, Clone, Default)]
pub struct RetrievalState {
    pub checkpoint: Option<ContextCheckpoint>,
}

impl RetrievalState {
    pub fn from_value(value: Option<Value>) -> Self {
        RetrievalState {
            checkpoint: value.and_then(|v| serde_json::from_value(v).ok()),
        }
    }

    pub fn to_value(&self) -> Option<Value> {
        self.checkpoint
            .as_ref()
            .and_then(|c| serde_json::to_value(c).ok())
    }

    /// Build the model view, advancing the cutoff only when necessary.
    ///
    /// Returns `(view, compacted, tokens, changed)`; persist when `changed`.
    pub fn model_view(
        &mut self,
        messages: &[ChatMessage],
        cfg: &ContextConfig,
        project: &Path,
        session_id: &str,
    ) -> (Vec<ChatMessage>, bool, usize, bool) {
        let trigger = cfg.retrieval_trigger_tokens.max(1);
        let keep = cfg.retrieval_keep_tokens.max(1);
        let recent = cfg.retrieval_keep_recent_user_messages.max(1);
        let min_rounds = cfg.retrieval_keep_min_rounds.max(1);
        let count = messages.len();

        let cutoff = self
            .checkpoint
            .as_ref()
            .map(|c| c.cutoff_index)
            .unwrap_or(0)
            .min(count);
        let latest_user = real_user_indices(messages).last().copied();
        let mut changed = false;

        // A restored checkpoint without a stub (e.g. legacy file) is rebuilt
        // once; it stays byte-stable afterwards until the cutoff advances.
        if cutoff > 0
            && self
                .checkpoint
                .as_ref()
                .map(|c| c.stub.is_empty())
                .unwrap_or(true)
        {
            let stub = build_checkpoint_stub(messages, cutoff, latest_user, project, session_id);
            self.checkpoint = Some(ContextCheckpoint {
                cutoff_index: cutoff,
                stub,
                session_id: session_id.to_string(),
                created_at: now_iso(),
                keep_tokens: keep,
            });
            changed = true;
        }

        let view = if cutoff > 0 {
            let stub = self
                .checkpoint
                .as_ref()
                .map(|c| c.stub.as_str())
                .unwrap_or("");
            render_retrieval_view(messages, cutoff, latest_user, stub)
        } else {
            messages.to_vec()
        };
        let tokens = estimate_messages_tokens(&view);
        if tokens <= trigger {
            return (view, cutoff > 0, tokens, changed);
        }

        let (new_cutoff, new_latest) =
            compute_retrieval_cutoff(messages, cutoff, keep, recent, min_rounds);
        if new_cutoff <= cutoff {
            return (view, cutoff > 0, tokens, changed);
        }

        let stub = build_checkpoint_stub(messages, new_cutoff, new_latest, project, session_id);
        let new_view = render_retrieval_view(messages, new_cutoff, new_latest, &stub);
        let new_tokens = estimate_messages_tokens(&new_view);
        // Moving the cutoff changes the request prefix and costs one cache
        // miss. Only do it when the saving is meaningful (>= 10%) or the view
        // exceeds the hard context limit.
        if !should_advance(tokens, new_tokens, cfg.max_context_tokens) {
            return (view, cutoff > 0, tokens, changed);
        }
        self.checkpoint = Some(ContextCheckpoint {
            cutoff_index: new_cutoff,
            stub,
            session_id: session_id.to_string(),
            created_at: now_iso(),
            keep_tokens: keep,
        });
        (new_view, true, new_tokens, true)
    }
}

/// Is advancing the cutoff worth the one-time prefix-cache miss?
///
/// Yes when it saves at least 10% of the current view, or when the current
/// view is over the hard context limit (any reduction helps).
fn should_advance(current_tokens: usize, new_tokens: usize, hard_limit: usize) -> bool {
    if new_tokens >= current_tokens {
        return false;
    }
    if current_tokens > hard_limit {
        return true;
    }
    let saved = current_tokens - new_tokens;
    saved.saturating_mul(10) >= current_tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ContextConfig;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage::simple(role, content)
    }

    fn cfg(trigger: usize, keep: usize) -> ContextConfig {
        ContextConfig {
            retrieval_trigger_tokens: trigger,
            retrieval_keep_tokens: keep,
            ..Default::default()
        }
    }

    #[test]
    fn real_user_excludes_injected_messages() {
        assert!(is_real_user_message(&msg("user", "hello")));
        assert!(!is_real_user_message(&msg("user", "[System] hint")));
        assert!(!is_real_user_message(&msg(
            "user",
            "<conversation-checkpoint>x"
        )));
        assert!(!is_real_user_message(&msg("assistant", "hi")));
    }

    #[test]
    fn view_starts_with_system_and_stub_no_orphan_tool() {
        let mut messages = vec![msg("system", "SYS")];
        for i in 0..10 {
            messages.push(msg("user", &format!("task {i}")));
            messages.push(msg("assistant", &format!("a {i}")));
            messages.push(ChatMessage::tool_result(format!("c{i}"), "obs"));
        }
        // Cutoff landing on a tool message must advance past it.
        let view = render_retrieval_view(&messages, 2, Some(9), "STUB");
        assert_eq!(view[0].role, "system");
        assert_eq!(view[1].role, "user");
        assert_eq!(view[1].content, "STUB");
        assert_ne!(view[2].role, "tool");
    }

    #[test]
    fn cutoff_never_moves_backwards_and_keeps_latest_user() {
        let mut messages = vec![msg("system", "SYS")];
        for i in 0..20 {
            messages.push(msg("user", &format!("task {i} {}", "x".repeat(400))));
            messages.push(ChatMessage::assistant("ok"));
        }
        let (c1, _) = compute_retrieval_cutoff(&messages, 0, 200, 2, 2);
        let (c2, _) = compute_retrieval_cutoff(&messages, c1, 200, 2, 2);
        assert!(c2 >= c1, "cutoff must not move backwards");
        let (c3, latest) = compute_retrieval_cutoff(&messages, 0, 1, 2, 2);
        assert!(c3 <= latest.unwrap());
    }

    #[test]
    fn no_trigger_below_threshold() {
        let mut state = RetrievalState::default();
        let messages = vec![msg("system", "SYS"), msg("user", "small task")];
        let (view, compacted, _, changed) =
            state.model_view(&messages, &cfg(100_000, 20_000), Path::new("."), "s1");
        assert!(!compacted);
        assert!(!changed);
        assert_eq!(view.len(), messages.len());
    }

    #[test]
    fn triggers_cutoff_and_freezes_while_under_trigger() {
        let mut messages = vec![msg("system", "SYS")];
        for i in 0..40 {
            messages.push(msg("user", &format!("task {i} {}", "x".repeat(500))));
            messages.push(ChatMessage::assistant("ok"));
        }
        let project = std::env::temp_dir();
        let mut state = RetrievalState::default();
        let (view1, compacted1, _, changed1) =
            state.model_view(&messages, &cfg(1000, 500), &project, "s1");
        assert!(compacted1 && changed1, "should advance the cutoff");
        assert!(view1.len() < messages.len());

        // Appending more messages must not change the frozen cutoff.
        messages.push(msg("user", "follow-up"));
        let before = state.checkpoint.as_ref().unwrap().cutoff_index;
        let (_, _, _, changed2) = state.model_view(&messages, &cfg(1000, 500), &project, "s1");
        let after = state.checkpoint.as_ref().unwrap().cutoff_index;
        assert_eq!(before, after, "cutoff must stay frozen while under trigger");
        assert!(!changed2);
    }

    #[test]
    fn advance_requires_meaningful_saving_or_hard_limit() {
        // 20% saving → advance.
        assert!(should_advance(1000, 800, 100_000));
        // 5% saving → skip (not worth a cache miss).
        assert!(!should_advance(1000, 950, 100_000));
        // Over the hard limit: any reduction helps.
        assert!(should_advance(1000, 990, 900));
        // No reduction → never.
        assert!(!should_advance(1000, 1000, 100_000));
        assert!(!should_advance(1000, 1200, 100_000));
    }

    #[test]
    fn checkpoint_survives_serialization() {
        let cp = ContextCheckpoint {
            cutoff_index: 5,
            stub: "STUB".into(),
            session_id: "s1".into(),
            created_at: "now".into(),
            keep_tokens: 100,
        };
        let v = serde_json::to_value(&cp).unwrap();
        let restored = RetrievalState::from_value(Some(v));
        assert_eq!(restored.checkpoint.unwrap().cutoff_index, 5);
    }
}
