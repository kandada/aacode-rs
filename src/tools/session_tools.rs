// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Session management tools — list/switch/delete/history/stats/new/continue.
//!
//! These operate on the on-disk session store via a fresh `SessionManager`
//! bound to the project path. Switching the *active* session is coordinated at
//! the FFI/CLI layer; these tools report information the model can act on.
//!
//! Ported from Python `core/main_agent.py` session tools.

use super::registry::Tool;
use super::schema::{ParamType, ToolParameter, ToolSchema};
use crate::error::Result;
use crate::session::SessionManager;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

pub struct ListSessionsTool {
    pub project_path: PathBuf,
}
impl Tool for ListSessionsTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("list_sessions", "List all conversation sessions.", vec![])
    }
    fn call(&self, _args: &Value, _c: &AtomicBool) -> Result<String> {
        let sm = SessionManager::new(&self.project_path);
        let sessions: Vec<Value> = sm
            .list_sessions()
            .into_iter()
            .map(|s| {
                json!({
                    "session_id": s.session_id,
                    "title": s.title,
                    "created_at": s.created_at,
                    "last_activity": s.last_activity,
                    "total_messages": s.total_messages,
                    "status": s.status,
                })
            })
            .collect();
        Ok(json!({"success": true, "count": sessions.len(), "sessions": sessions}).to_string())
    }
}

pub struct GetConversationHistoryTool {
    pub project_path: PathBuf,
}
impl Tool for GetConversationHistoryTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "get_conversation_history",
            "Get recent conversation history of a session (or the latest).",
            vec![
                ToolParameter::new("session_id", ParamType::String, false, "Session id", &["id"]),
                ToolParameter::new("max_length", ParamType::Integer, false, "Max messages", &["limit"]),
            ],
        )
    }
    fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let sm = SessionManager::new(&self.project_path);
        let id = args.get("session_id").and_then(|v| v.as_str()).map(|s| s.to_string());
        let max = args.get("max_length").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
        let id = id.or_else(|| sm.list_sessions().first().map(|s| s.session_id.clone()));
        let id = match id {
            Some(i) => i,
            None => return Ok(json!({"success": false, "error": "no sessions"}).to_string()),
        };
        let msgs = sm.read_session_messages(&id);
        let recent: Vec<Value> = msgs
            .iter()
            .rev()
            .take(max)
            .rev()
            .map(|m| json!({"role": m.role, "content": preview(&m.content, 200)}))
            .collect();
        Ok(json!({"success": true, "session_id": id, "history": recent}).to_string())
    }
}

pub struct GetSessionStatsTool {
    pub project_path: PathBuf,
}
impl Tool for GetSessionStatsTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("get_session_stats", "Get statistics for sessions.", vec![])
    }
    fn call(&self, _args: &Value, _c: &AtomicBool) -> Result<String> {
        let sm = SessionManager::new(&self.project_path);
        let sessions = sm.list_sessions();
        let total_msgs: usize = sessions.iter().map(|s| s.total_messages).sum();
        let total_tokens: usize = sessions.iter().map(|s| s.total_tokens).sum();
        Ok(json!({
            "success": true,
            "stats": {
                "session_count": sessions.len(),
                "total_messages": total_msgs,
                "total_tokens": total_tokens,
            }
        })
        .to_string())
    }
}

pub struct DeleteSessionTool {
    pub project_path: PathBuf,
}
impl Tool for DeleteSessionTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "delete_session",
            "Delete a conversation session by id.",
            vec![ToolParameter::new("session_id", ParamType::String, true, "Session id", &["id"])],
        )
    }
    fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let id = args.get("session_id").and_then(|v| v.as_str()).unwrap_or("");
        let mut sm = SessionManager::new(&self.project_path);
        let ok = sm.delete_session(id).unwrap_or(false);
        Ok(json!({"success": ok, "session_id": id}).to_string())
    }
}

/// Informational stubs for new/continue/switch. Actual active-session change is
/// coordinated by the caller (CLI/FFI). These acknowledge the request.
pub struct NewSessionTool {
    pub project_path: PathBuf,
}
impl Tool for NewSessionTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "new_session",
            "Request a new conversation session for the next task.",
            vec![ToolParameter::new("task", ParamType::String, false, "Initial task", &[])],
        )
    }
    fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let task = args.get("task").and_then(|v| v.as_str()).unwrap_or("New task");
        let title = args.get("title").and_then(|v| v.as_str());
        let mut sm = SessionManager::new(&self.project_path);
        match sm.create_session(task, title) {
            Ok(id) => Ok(json!({"success": true, "session_id": id, "message": "New session created"}).to_string()),
            Err(e) => Ok(json!({"success": false, "error": e.to_string()}).to_string()),
        }
    }
}

pub struct ContinueSessionTool {
    pub project_path: PathBuf,
}
impl Tool for ContinueSessionTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "continue_session",
            "Add a follow-up message to the current or specified session.",
            vec![
                ToolParameter::new("message", ParamType::String, true, "Follow-up message", &[]),
                ToolParameter::new("session_id", ParamType::String, false, "Target session id", &["id"]),
            ],
        )
    }
    fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let msg = args.get("message").and_then(|v| v.as_str()).unwrap_or("");
        if msg.is_empty() {
            return Ok(json!({"success": false, "error": "message required"}).to_string());
        }
        let sid = args.get("session_id").and_then(|v| v.as_str()).map(|s| s.to_string());
        let mut sm = SessionManager::new(&self.project_path);
        if let Some(ref id) = sid {
            if !sm.switch_session(id).unwrap_or(false) {
                return Ok(json!({"success": false, "error": format!("session {id} not found")}).to_string());
            }
        }
        if sm.current_session_id.is_none() {
            let id = sm.create_session(&msg, None)?;
            return Ok(json!({"success": true, "session_id": id}).to_string());
        }
        sm.add_message(crate::session::SessionMessage::from_chat(
            &crate::llm::ChatMessage::user(msg),
        ))?;
        let id = sm.current_session_id.clone().unwrap_or_default();
        Ok(json!({"success": true, "session_id": id}).to_string())
    }
}

pub struct SwitchSessionTool {
    pub project_path: PathBuf,
}
impl Tool for SwitchSessionTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "switch_session",
            "Switch to a specified conversation session.",
            vec![ToolParameter::new("session_id", ParamType::String, true, "Session id", &["id"])],
        )
    }
    fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let id = args.get("session_id").and_then(|v| v.as_str()).unwrap_or("");
        let mut sm = SessionManager::new(&self.project_path);
        let ok = sm.switch_session(id).unwrap_or(false);
        Ok(json!({"success": ok, "session_id": id}).to_string())
    }
}

fn preview(s: &str, n: usize) -> String {
    let t: String = s.chars().take(n).collect();
    if s.chars().count() > n {
        format!("{t}...")
    } else {
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SessionManager;

    fn tmp() -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "aacode_sesstool_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn list_and_stats() {
        let d = tmp();
        {
            let mut sm = SessionManager::new(&d);
            sm.create_session("task one", None).unwrap();
        }
        let cancel = AtomicBool::new(false);
        let out = ListSessionsTool { project_path: d.clone() }
            .call(&json!({}), &cancel)
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["count"].as_u64().unwrap() >= 1);

        let out = GetSessionStatsTool { project_path: d.clone() }
            .call(&json!({}), &cancel)
            .unwrap();
        assert!(serde_json::from_str::<Value>(&out).unwrap()["success"] == true);
    }

    #[test]
    fn delete_session() {
        let d = tmp();
        let id = {
            let mut sm = SessionManager::new(&d);
            sm.create_session("x", None).unwrap()
        };
        let cancel = AtomicBool::new(false);
        let out = DeleteSessionTool { project_path: d.clone() }
            .call(&json!({"session_id": id}), &cancel)
            .unwrap();
        assert_eq!(serde_json::from_str::<Value>(&out).unwrap()["success"], true);
    }

    #[test]
    fn history_no_sessions() {
        let d = tmp();
        let cancel = AtomicBool::new(false);
        let out = GetConversationHistoryTool { project_path: d }
            .call(&json!({}), &cancel)
            .unwrap();
        assert_eq!(serde_json::from_str::<Value>(&out).unwrap()["success"], false);
    }

    #[test]
    fn new_and_continue_ack() {
        let d = tmp();
        let cancel = AtomicBool::new(false);
        let nt = NewSessionTool { project_path: d.clone() };
        let out = nt.call(&json!({"task": "test"}), &cancel).unwrap();
        assert!(out.contains("New session created"));
        let ct = ContinueSessionTool { project_path: d };
        let out = ct.call(&json!({"message": "followup"}), &cancel).unwrap();
        assert!(out.contains("true"));
    }
}
