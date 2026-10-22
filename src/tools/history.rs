// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Read-only history retrieval tools.
//!
//! When the retrieval cutoff omits older messages from the model view, the
//! agent can inspect them with `search_history` / `read_history` — without
//! `cat`-ing multi-megabyte session JSON.

use super::registry::Tool;
use super::schema::{ParamType, ToolParameter, ToolSchema};
use crate::error::{AacodeError, Result};
use crate::session::{SessionManager, SessionMessage};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

fn clamp_u64(args: &Value, key: &str, default: u64, min: u64, max: u64) -> u64 {
    args.get(key)
        .and_then(|v| v.as_u64())
        .unwrap_or(default)
        .clamp(min, max)
}

fn take_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn tool_calls_text(message: &SessionMessage, limit: usize) -> String {
    let Some(calls) = &message.tool_calls else {
        return String::new();
    };
    calls
        .iter()
        .map(|c| format!("{}({})", c.name, take_chars(&c.arguments, limit)))
        .collect::<Vec<_>>()
        .join("; ")
}

fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn ceil_char_boundary(s: &str, mut i: usize) -> usize {
    let n = s.len();
    while i < n && !s.is_char_boundary(i) {
        i += 1;
    }
    i.min(n)
}

fn snippet(text: &str, query: &str, radius: usize) -> String {
    let flat = text.replace('\n', " ");
    let lower = flat.to_lowercase();
    let Some(pos) = lower.find(&query.to_lowercase()) else {
        return take_chars(&flat, radius * 2);
    };
    // `pos` comes from the lowercased copy; clamp the byte offsets to valid
    // char boundaries in the original so slicing can never panic.
    let start = floor_char_boundary(&flat, pos.saturating_sub(radius));
    let end = ceil_char_boundary(&flat, (pos + query.len() + radius).min(flat.len()));
    let mut out = String::new();
    if start > 0 {
        out.push_str("...");
    }
    out.push_str(&flat[start..end]);
    if end < flat.len() {
        out.push_str("...");
    }
    out
}

fn current_id(holder: &Arc<Mutex<Option<String>>>) -> Option<String> {
    holder.lock().ok().and_then(|g| g.clone())
}

pub struct SearchHistoryTool {
    pub project_path: PathBuf,
    pub active_session_id: Arc<Mutex<Option<String>>>,
}

#[async_trait::async_trait]
impl Tool for SearchHistoryTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "search_history",
            "Search persisted conversation messages that may have been omitted from the model context. \
             Use this instead of cat-ing session JSON. Searches message content and tool-call arguments; \
             reasoning content is excluded. Default scope is the current session; use scope=project for \
             other sessions (use list_sessions to discover session IDs).",
            vec![
                ToolParameter::new(
                    "query",
                    ParamType::String,
                    true,
                    "Case-insensitive text to search for (error, path, requirement, ...).",
                    &["q", "text", "keyword"],
                ),
                ToolParameter::new(
                    "session_id",
                    ParamType::String,
                    false,
                    "Specific session id; defaults to the current session.",
                    &["session"],
                ),
                ToolParameter::new(
                    "scope",
                    ParamType::String,
                    false,
                    "current (default) or project (all sessions).",
                    &["range"],
                ),
                ToolParameter::new(
                    "role",
                    ParamType::String,
                    false,
                    "Optional role filter: user, assistant, tool, system.",
                    &["role_filter"],
                ),
                ToolParameter::new(
                    "limit",
                    ParamType::Integer,
                    false,
                    "Maximum matches to return (1-50, default 20).",
                    &["max_results", "count"],
                ),
                ToolParameter::new(
                    "offset",
                    ParamType::Integer,
                    false,
                    "Pagination offset (default 0).",
                    &["skip"],
                ),
            ],
        )
    }

    async fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if query.is_empty() {
            return Ok("search_history requires a non-empty query.".to_string());
        }
        let session_id = args
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let scope = args
            .get("scope")
            .and_then(|v| v.as_str())
            .unwrap_or("current")
            .to_string();
        let role = args
            .get("role")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty());
        if let Some(r) = &role {
            if !["user", "assistant", "tool", "system"].contains(&r.as_str()) {
                return Ok("role must be one of: user, assistant, tool, system.".to_string());
            }
        }
        let limit = clamp_u64(args, "limit", 20, 1, 50) as usize;
        let offset = clamp_u64(args, "offset", 0, 0, u64::MAX) as usize;
        let current = current_id(&self.active_session_id);
        let project = self.project_path.clone();

        tokio::task::spawn_blocking(move || -> Result<String> {
            let sm = SessionManager::new(&project);
            let session_ids: Vec<String> = if let Some(sid) = session_id {
                vec![sid]
            } else if scope == "project" {
                sm.list_sessions()
                    .into_iter()
                    .take(50)
                    .map(|s| s.session_id)
                    .collect()
            } else {
                current.into_iter().collect()
            };

            let mut matches: Vec<(String, usize, String, String, String)> = Vec::new();
            for sid in session_ids {
                for (index, message) in sm.read_session_messages(&sid).into_iter().enumerate() {
                    if let Some(r) = &role {
                        if &message.role != r {
                            continue;
                        }
                    }
                    let calls = tool_calls_text(&message, 300);
                    let haystack = format!("{}\n{}", message.content, calls);
                    if !haystack.to_lowercase().contains(&query.to_lowercase()) {
                        continue;
                    }
                    matches.push((
                        sid.clone(),
                        index,
                        message.role.clone(),
                        message.timestamp.clone(),
                        snippet(&haystack, &query, 320),
                    ));
                }
            }

            if matches.is_empty() {
                return Ok(format!("No matches for {query:?} in {scope} scope."));
            }
            let total = matches.len();
            let sliced: Vec<_> = matches.into_iter().skip(offset).take(limit).collect();
            let mut lines = vec![format!(
                "Matches ({total} total, showing {}-{}):",
                offset + 1,
                offset + sliced.len()
            )];
            for (sid, index, role, timestamp, snip) in sliced {
                lines.push(format!("[{sid}][{index}] {role} {timestamp}: {snip}"));
            }
            if total > offset + limit {
                lines.push(format!("[More matches: offset={}]", offset + limit));
            }
            Ok(lines.join("\n"))
        })
        .await
        .map_err(|e| AacodeError::Other(e.to_string()))?
    }
}

pub struct ReadHistoryTool {
    pub project_path: PathBuf,
    pub active_session_id: Arc<Mutex<Option<String>>>,
}

#[async_trait::async_trait]
impl Tool for ReadHistoryTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "read_history",
            "Read a bounded range of persisted conversation messages by index. Use search_history first \
             to find the relevant index. Never cat the session JSON. Returns role, timestamp, tool calls \
             and truncated content.",
            vec![
                ToolParameter::new(
                    "session_id",
                    ParamType::String,
                    false,
                    "Session id; defaults to the current session.",
                    &["session"],
                ),
                ToolParameter::new(
                    "start_index",
                    ParamType::Integer,
                    false,
                    "First message index to read (0-based, default 0).",
                    &["start", "from"],
                ),
                ToolParameter::new(
                    "max_messages",
                    ParamType::Integer,
                    false,
                    "Maximum messages to return (1-50, default 20).",
                    &["limit", "count"],
                ),
                ToolParameter::new(
                    "max_chars",
                    ParamType::Integer,
                    false,
                    "Maximum total characters (1000-30000, default 12000).",
                    &["max_output", "chars"],
                ),
            ],
        )
    }

    async fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let session_id = args
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .or_else(|| current_id(&self.active_session_id));
        let Some(sid) = session_id else {
            return Ok("No active session.".to_string());
        };
        let start_index = clamp_u64(args, "start_index", 0, 0, u64::MAX) as usize;
        let max_messages = clamp_u64(args, "max_messages", 20, 1, 50) as usize;
        let max_chars = clamp_u64(args, "max_chars", 12_000, 1_000, 30_000) as usize;
        let project = self.project_path.clone();

        tokio::task::spawn_blocking(move || -> Result<String> {
            let sm = SessionManager::new(&project);
            let messages = sm.read_session_messages(&sid);
            if messages.is_empty() {
                return Ok(format!("Session {sid} has no messages, or does not exist."));
            }
            let start = start_index.min(messages.len());
            let end = (start + max_messages).min(messages.len());
            let per_message = (max_chars / max_messages.max(1)).max(500);

            let mut lines = vec![format!(
                "Session {sid} | messages {start}-{} of {}:",
                end.saturating_sub(1),
                messages.len()
            )];
            let mut used = lines[0].chars().count();
            let mut next_index = start;
            for (index, message) in messages.iter().enumerate().take(end).skip(start) {
                let mut content = message.content.clone();
                if content.chars().count() > per_message {
                    let total = content.chars().count();
                    content = format!(
                        "{}... [{total} chars total]",
                        take_chars(&content, per_message)
                    );
                }
                let mut block = vec![format!("[{index}] {} {}", message.role, message.timestamp)];
                if let Some(id) = &message.tool_call_id {
                    block.push(format!("tool_call_id: {id}"));
                }
                let calls = tool_calls_text(message, 300);
                if !calls.is_empty() {
                    block.push(format!("tool_calls: {calls}"));
                }
                block.push(content);
                let text = block.join("\n");
                let cost = text.chars().count();
                if used + cost > max_chars && index > start {
                    break;
                }
                lines.push(text);
                used += cost;
                next_index = index + 1;
            }
            if next_index < messages.len() {
                lines.push(format!("[Continue with start_index={next_index}]"));
            }
            Ok(lines.join("\n\n"))
        })
        .await
        .map_err(|e| AacodeError::Other(e.to_string()))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::ChatMessage;
    use serde_json::json;

    fn seed(project: &std::path::Path) -> String {
        let mut sm = SessionManager::new(project);
        let sid = sm.create_session("fix the cache bug", None).unwrap();
        sm.add_message(SessionMessage::from_chat(&ChatMessage::user(
            "please fix E_CACHE_MISS in loader.rs",
        )))
        .unwrap();
        sm.add_message(SessionMessage::from_chat(&ChatMessage::assistant(
            "I will inspect loader.rs",
        )))
        .unwrap();
        sm.flush().unwrap();
        sid
    }

    fn holder(sid: &str) -> Arc<Mutex<Option<String>>> {
        Arc::new(Mutex::new(Some(sid.to_string())))
    }

    #[tokio::test]
    async fn search_finds_and_limits() {
        let dir = std::env::temp_dir().join(format!(
            "aacode_hist_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let sid = seed(&dir);
        let tool = SearchHistoryTool {
            project_path: dir.clone(),
            active_session_id: holder(&sid),
        };
        let out = tool
            .call(&json!({"query": "E_CACHE_MISS"}), &AtomicBool::new(false))
            .await
            .unwrap();
        assert!(out.contains("Matches"), "{out}");
        assert!(out.contains("E_CACHE_MISS"), "{out}");

        let none = tool
            .call(
                &json!({"query": "zzz-not-present"}),
                &AtomicBool::new(false),
            )
            .await
            .unwrap();
        assert!(none.contains("No matches"), "{none}");
    }

    #[tokio::test]
    async fn read_returns_bounded_transcript() {
        let dir = std::env::temp_dir().join(format!(
            "aacode_hist2_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let sid = seed(&dir);
        let tool = ReadHistoryTool {
            project_path: dir,
            active_session_id: holder(&sid),
        };
        let out = tool
            .call(
                &json!({"start_index": 0, "max_messages": 2}),
                &AtomicBool::new(false),
            )
            .await
            .unwrap();
        assert!(out.contains("messages 0-1"), "{out}");
        assert!(out.contains("E_CACHE_MISS"), "{out}");
    }
}
