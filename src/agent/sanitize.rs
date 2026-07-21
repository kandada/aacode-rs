// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Message-history sanitizer: guarantees the OpenAI tool-calling invariant
//! *"an assistant message with `tool_calls` must be followed by tool messages
//! responding to each `tool_call_id`"* before anything is sent to the API.
//!
//! Histories get broken in the real world: a task is cancelled or times out
//! between persisting the assistant(tool_calls) message and persisting the
//! tool results; the process dies mid-execution; an old/buggy session file is
//! resumed. Without repair, such a session is **permanently poisoned** — every
//! subsequent request replays the dangling `tool_calls` and the provider
//! rejects it with HTTP 400 (`insufficient tool messages following
//! tool_calls`).
//!
//! Strategy (repair, don't reject — mirrors and extends the Python
//! `validate_tool_call_integrity`, which only detects):
//!   1. Every `tool_call_id` without a following tool result gets a
//!      **synthetic tool message** explaining the interruption, so the model
//!      knows the call never completed and the pairing is valid again.
//!   2. **Orphaned tool messages** (no preceding assistant tool_call with a
//!      matching id) are dropped — providers reject those too.
//!   3. Tool results are kept **adjacent** to their assistant message (any
//!      interleaved non-tool messages are moved after the tool block).

use crate::llm::types::ChatMessage;
use std::collections::HashSet;

/// Placeholder content injected for a tool call that never produced a result.
pub const INTERRUPTED_RESULT: &str =
    "[tool execution was interrupted before a result was recorded (task \
     cancelled, timeout, or crash); treat this call as not executed]";

/// Repairs `messages` in place so the tool_calls/tool pairing invariant holds.
/// Returns the number of repairs performed (0 = history was already valid).
pub fn sanitize_history(messages: &mut Vec<ChatMessage>) -> usize {
    let mut repairs = 0usize;
    let mut out: Vec<ChatMessage> = Vec::with_capacity(messages.len());
    let mut i = 0;

    while i < messages.len() {
        let msg = messages[i].clone();

        // Orphaned tool message: tool role appearing here means it did NOT
        // directly follow an assistant tool_calls block (those are consumed
        // in the branch below). Drop it.
        if msg.role == "tool" {
            repairs += 1;
            i += 1;
            continue;
        }

        let has_tool_calls = msg
            .tool_calls
            .as_ref()
            .map(|t| !t.is_empty())
            .unwrap_or(false);

        if !has_tool_calls {
            out.push(msg);
            i += 1;
            continue;
        }

        // assistant with tool_calls: gather the responses that follow.
        // Providers require tool messages to be adjacent; tolerate (and fix)
        // interleaved non-tool messages by collecting until the next message
        // that clearly starts a new round (user/assistant).
        let expected: Vec<String> = msg
            .tool_calls
            .as_ref()
            .unwrap()
            .iter()
            .map(|tc| tc.id.clone())
            .collect();
        let expected_set: HashSet<&str> = expected.iter().map(|s| s.as_str()).collect();

        out.push(msg);
        i += 1;

        let mut answered: HashSet<String> = HashSet::new();
        let mut displaced: Vec<ChatMessage> = Vec::new();
        let mut tool_block: Vec<ChatMessage> = Vec::new();

        while i < messages.len() {
            let next = &messages[i];
            match next.role.as_str() {
                "tool" => {
                    let id = next.tool_call_id.clone().unwrap_or_default();
                    if expected_set.contains(id.as_str()) && !answered.contains(&id) {
                        answered.insert(id);
                        tool_block.push(next.clone());
                    } else {
                        // duplicate or foreign tool result → drop
                        repairs += 1;
                    }
                    i += 1;
                }
                // System notes may legally be interleaved by our own loop;
                // move them after the tool block to preserve adjacency.
                "system" if answered.len() < expected.len() => {
                    displaced.push(next.clone());
                    i += 1;
                }
                _ => break, // user/assistant → the round ended
            }
        }

        // Inject synthetic results for every unanswered tool_call_id,
        // preserving the original call order.
        for id in &expected {
            if !answered.contains(id) {
                repairs += 1;
                tool_block.push(ChatMessage::tool_result(
                    id.clone(),
                    INTERRUPTED_RESULT.to_string(),
                ));
            }
        }

        // Reorder synthetic/real results to match the tool_calls order (some
        // providers validate order as well).
        tool_block.sort_by_key(|t| {
            let id = t.tool_call_id.as_deref().unwrap_or("");
            expected.iter().position(|e| e == id).unwrap_or(usize::MAX)
        });

        out.extend(tool_block);
        out.extend(displaced);
    }

    *messages = out;
    repairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::ToolCall;

    fn asst_with_tools(ids: &[&str]) -> ChatMessage {
        ChatMessage::assistant_with_tools(
            String::new(),
            ids.iter()
                .map(|id| ToolCall {
                    id: id.to_string(),
                    name: "run_shell".into(),
                    arguments: "{}".into(),
                })
                .collect(),
        )
    }

    fn tool(id: &str) -> ChatMessage {
        ChatMessage::tool_result(id.to_string(), "ok".to_string())
    }

    #[test]
    fn valid_history_untouched() {
        let mut m = vec![
            ChatMessage::system("s"),
            ChatMessage::user("u"),
            asst_with_tools(&["a", "b"]),
            tool("a"),
            tool("b"),
            ChatMessage {
                role: "assistant".into(),
                content: "done".into(),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];
        let before = m.clone();
        assert_eq!(sanitize_history(&mut m), 0);
        assert_eq!(m.len(), before.len());
    }

    #[test]
    fn dangling_tool_calls_get_synthetic_results() {
        // The exact HTTP-400 scenario: assistant(tool_calls) persisted, task
        // cancelled before any tool result was written.
        let mut m = vec![
            ChatMessage::user("u"),
            asst_with_tools(&["call_1", "call_2"]),
            // (no tool messages — session interrupted)
            ChatMessage::user("continue please"),
        ];
        let repairs = sanitize_history(&mut m);
        assert_eq!(repairs, 2);
        assert_eq!(m[2].role, "tool");
        assert_eq!(m[2].tool_call_id.as_deref(), Some("call_1"));
        assert!(m[2].content.contains("interrupted"));
        assert_eq!(m[3].tool_call_id.as_deref(), Some("call_2"));
        assert_eq!(m[4].role, "user");
    }

    #[test]
    fn partially_answered_tool_calls_completed() {
        // First tool ran, cancel hit before the second → only the missing one
        // gets a synthetic result, order preserved.
        let mut m = vec![
            asst_with_tools(&["a", "b", "c"]),
            tool("a"),
            ChatMessage::user("next"),
        ];
        let repairs = sanitize_history(&mut m);
        assert_eq!(repairs, 2);
        let ids: Vec<_> = m
            .iter()
            .filter(|x| x.role == "tool")
            .map(|x| x.tool_call_id.clone().unwrap())
            .collect();
        assert_eq!(ids, ["a", "b", "c"]);
        assert_eq!(m.last().unwrap().role, "user");
    }

    #[test]
    fn orphaned_tool_messages_dropped() {
        let mut m = vec![
            ChatMessage::user("u"),
            tool("ghost"), // no assistant tool_calls before it
            ChatMessage::user("v"),
        ];
        let repairs = sanitize_history(&mut m);
        assert_eq!(repairs, 1);
        assert!(m.iter().all(|x| x.role != "tool"));
    }

    #[test]
    fn interleaved_system_note_moved_after_tool_block() {
        // Our loop can insert [SYSTEM WARNING] notes; ensure adjacency of
        // assistant(tool_calls) → tool results survives.
        let mut m = vec![
            asst_with_tools(&["a"]),
            ChatMessage::system("[SYSTEM WARNING]: stale"),
            tool("a"),
        ];
        sanitize_history(&mut m);
        assert_eq!(m[0].role, "assistant");
        assert_eq!(m[1].role, "tool");
        assert_eq!(m[2].role, "system");
    }

    #[test]
    fn duplicate_tool_result_dropped() {
        let mut m = vec![asst_with_tools(&["a"]), tool("a"), tool("a")];
        let repairs = sanitize_history(&mut m);
        assert_eq!(repairs, 1);
        assert_eq!(m.iter().filter(|x| x.role == "tool").count(), 1);
    }

    #[test]
    fn foreign_tool_id_dropped_and_missing_injected() {
        let mut m = vec![asst_with_tools(&["a"]), tool("zzz")];
        let repairs = sanitize_history(&mut m);
        assert_eq!(repairs, 2); // dropped foreign + injected missing "a"
        let tools: Vec<_> = m.iter().filter(|x| x.role == "tool").collect();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].tool_call_id.as_deref(), Some("a"));
    }
}
