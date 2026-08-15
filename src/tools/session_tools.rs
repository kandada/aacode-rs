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
use crate::error::{AacodeError, Result};
use crate::session::SessionManager;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

pub struct ListSessionsTool {
    pub project_path: PathBuf,
}
#[async_trait::async_trait]
impl Tool for ListSessionsTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("list_sessions", "List all conversation sessions.", vec![])
    }
    async fn call(&self, _args: &Value, _c: &AtomicBool) -> Result<String> {
        let project_path = self.project_path.clone();
        tokio::task::spawn_blocking(move || -> Result<String> {
            let sm = SessionManager::new(&project_path);
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
        })
        .await
        .map_err(|e| AacodeError::Other(e.to_string()))?
    }
}

pub struct GetConversationHistoryTool {
    pub project_path: PathBuf,
    pub active_session_id: Arc<Mutex<Option<String>>>,
}
#[async_trait::async_trait]
impl Tool for GetConversationHistoryTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "get_conversation_history",
            "Read messages from a session. Returns role, content, timestamp, tool_calls, tool_call_id, and reasoning_content for each message. Fields longer than max_content_chars chars are truncated with a length marker.",
            vec![
                ToolParameter::new("session_id", ParamType::String, false, "Session id (defaults to current)", &["id"]),
                ToolParameter::new("range_from", ParamType::Integer, false, "Start index (0 = earliest message)", &["from"]),
                ToolParameter::new("range_to", ParamType::Integer, false, "End index exclusive (default = latest)", &["to"]),
                ToolParameter::new("max_content_chars", ParamType::Integer, false, "Max chars per field before truncation", &["limit"]),
            ],
        )
    }
    async fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let id = args.get("session_id").and_then(|v| v.as_str()).map(|s| s.to_string());
        let range_from = args.get("range_from").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let range_to = args.get("range_to").and_then(|v| v.as_u64()).map(|n| n as usize);
        let max_chars = args.get("max_content_chars").and_then(|v| v.as_u64()).unwrap_or(100) as usize;
        let project_path = self.project_path.clone();
        let active_id = self.active_session_id.lock().unwrap().clone();
        tokio::task::spawn_blocking(move || -> Result<String> {
            let sm = SessionManager::new(&project_path);
            let id = id
                .or(active_id)
                .or_else(|| sm.list_sessions().first().map(|s| s.session_id.clone()));
            let id = match id {
                Some(i) => i,
                None => return Ok(json!({"success": false, "error": "no sessions"}).to_string()),
            };
            let msgs = sm.read_session_messages(&id);
            let total = msgs.len();
            let to = range_to.unwrap_or(total).min(total);
            let from = range_from.min(to);
            let sliced = &msgs[from..to];

            let history: Vec<Value> = sliced
                .iter()
                .map(|m| {
                    let mut obj = json!({
                        "role": m.role,
                        "content": truncate_str(&m.content, max_chars),
                        "timestamp": m.timestamp,
                    });
                    if let Some(ref tcs) = m.tool_calls {
                        let calls: Vec<Value> = tcs.iter().map(|tc| json!({
                            "id": tc.id,
                            "name": tc.name,
                            "arguments": truncate_str(&tc.arguments, max_chars),
                        })).collect();
                        obj["tool_calls"] = json!(calls);
                    }
                    if let Some(ref tci) = m.tool_call_id {
                        obj["tool_call_id"] = json!(tci);
                    }
                    if let Some(ref rc) = m.reasoning_content {
                        obj["reasoning_content"] = json!(truncate_str(rc, max_chars));
                    }
                    obj
                })
                .collect();

            Ok(json!({
                "success": true,
                "session_id": id,
                "total_messages": total,
                "history": history,
            }).to_string())
        })
        .await
        .map_err(|e| AacodeError::Other(e.to_string()))?
    }
}

fn truncate_str(s: &str, max_chars: usize) -> String {
    if max_chars == 0 || s.chars().count() <= max_chars {
        return s.to_string();
    }
    let head: String = s.chars().take(max_chars).collect();
    format!("{}…({} chars total)", head, s.chars().count())
}

pub struct GetSessionStatsTool {
    pub project_path: PathBuf,
}
#[async_trait::async_trait]
impl Tool for GetSessionStatsTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("get_session_stats", "Get statistics for sessions.", vec![])
    }
    async fn call(&self, _args: &Value, _c: &AtomicBool) -> Result<String> {
        let project_path = self.project_path.clone();
        tokio::task::spawn_blocking(move || -> Result<String> {
            let sm = SessionManager::new(&project_path);
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
        })
        .await
        .map_err(|e| AacodeError::Other(e.to_string()))?
    }
}

pub struct DeleteSessionTool {
    pub project_path: PathBuf,
}
#[async_trait::async_trait]
impl Tool for DeleteSessionTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "delete_session",
            "Delete a conversation session by id.",
            vec![ToolParameter::new("session_id", ParamType::String, true, "Session id", &["id"])],
        )
    }
    async fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let id = args.get("session_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let project_path = self.project_path.clone();
        tokio::task::spawn_blocking(move || -> Result<String> {
            let mut sm = SessionManager::new(&project_path);
            let ok = sm.delete_session(&id).unwrap_or(false);
            Ok(json!({"success": ok, "session_id": id}).to_string())
        })
        .await
        .map_err(|e| AacodeError::Other(e.to_string()))?
    }
}

/// Informational stubs for new/continue/switch. Actual active-session change is
/// coordinated by the caller (CLI/FFI). These acknowledge the request.
pub struct NewSessionTool {
    pub project_path: PathBuf,
}
#[async_trait::async_trait]
impl Tool for NewSessionTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "new_session",
            "Request a new conversation session for the next task.",
            vec![ToolParameter::new("task", ParamType::String, false, "Initial task", &[])],
        )
    }
    async fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let task = args.get("task").and_then(|v| v.as_str()).unwrap_or("New task").to_string();
        let title = args.get("title").and_then(|v| v.as_str()).map(|s| s.to_string());
        let project_path = self.project_path.clone();
        tokio::task::spawn_blocking(move || -> Result<String> {
            let mut sm = SessionManager::new(&project_path);
            match sm.create_session(&task, title.as_deref()) {
                Ok(id) => Ok(json!({"success": true, "session_id": id, "message": "New session created"}).to_string()),
                Err(e) => Ok(json!({"success": false, "error": e.to_string()}).to_string()),
            }
        })
        .await
        .map_err(|e| AacodeError::Other(e.to_string()))?
    }
}

pub struct ContinueSessionTool {
    pub project_path: PathBuf,
}
#[async_trait::async_trait]
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
    async fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let msg = args.get("message").and_then(|v| v.as_str()).unwrap_or("").to_string();
        if msg.is_empty() {
            return Ok(json!({"success": false, "error": "message required"}).to_string());
        }
        let sid = args.get("session_id").and_then(|v| v.as_str()).map(|s| s.to_string());
        let project_path = self.project_path.clone();
        tokio::task::spawn_blocking(move || -> Result<String> {
            let mut sm = SessionManager::new(&project_path);
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
        })
        .await
        .map_err(|e| AacodeError::Other(e.to_string()))?
    }
}

pub struct SwitchSessionTool {
    pub project_path: PathBuf,
}
#[async_trait::async_trait]
impl Tool for SwitchSessionTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "switch_session",
            "Switch to a specified conversation session.",
            vec![ToolParameter::new("session_id", ParamType::String, true, "Session id", &["id"])],
        )
    }
    async fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let id = args.get("session_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let project_path = self.project_path.clone();
        tokio::task::spawn_blocking(move || -> Result<String> {
            let mut sm = SessionManager::new(&project_path);
            let ok = sm.switch_session(&id).unwrap_or(false);
            Ok(json!({"success": ok, "session_id": id}).to_string())
        })
        .await
        .map_err(|e| AacodeError::Other(e.to_string()))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::ToolCall;
    use crate::llm::ChatMessage;
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

    fn history_tool(path: PathBuf) -> GetConversationHistoryTool {
        GetConversationHistoryTool {
            project_path: path,
            active_session_id: Arc::new(Mutex::new(None)),
        }
    }

    #[tokio::test]
    async fn list_and_stats() {
        let d = tmp();
        {
            let mut sm = SessionManager::new(&d);
            sm.create_session("task one", None).unwrap();
        }
        let cancel = AtomicBool::new(false);
        let out = ListSessionsTool { project_path: d.clone() }
            .call(&json!({}), &cancel)
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["count"].as_u64().unwrap() >= 1);

        let out = GetSessionStatsTool { project_path: d.clone() }
            .call(&json!({}), &cancel)
            .await
            .unwrap();
        assert!(serde_json::from_str::<Value>(&out).unwrap()["success"] == true);
    }

    #[tokio::test]
    async fn delete_session() {
        let d = tmp();
        let id = {
            let mut sm = SessionManager::new(&d);
            sm.create_session("x", None).unwrap()
        };
        let cancel = AtomicBool::new(false);
        let out = DeleteSessionTool { project_path: d.clone() }
            .call(&json!({"session_id": id}), &cancel)
            .await
            .unwrap();
        assert_eq!(serde_json::from_str::<Value>(&out).unwrap()["success"], true);
    }

    #[tokio::test]
    async fn history_no_sessions() {
        let d = tmp();
        let cancel = AtomicBool::new(false);
        let out = history_tool(d)
            .call(&json!({}), &cancel)
            .await
            .unwrap();
        assert_eq!(serde_json::from_str::<Value>(&out).unwrap()["success"], false);
    }

    #[tokio::test]
    async fn history_includes_tool_calls_and_reasoning() {
        let d = tmp();
        let id = {
            let mut sm = SessionManager::new(&d);
            sm.create_session("task", None).unwrap();
            let assistant = ChatMessage::assistant_with_tools(
                "I will run a command",
                vec![ToolCall {
                    id: "call_abc".into(),
                    name: "run_shell".into(),
                    arguments: r#"{"command":"ls"}"#.into(),
                }],
            );
            let mut msg = crate::session::SessionMessage::from_chat(&assistant);
            msg.reasoning_content = Some("I should check the directory".into());
            sm.add_message(msg).unwrap();

            sm.add_message(crate::session::SessionMessage::from_chat(
                &ChatMessage::tool_result("call_abc", "file1.txt\nfile2.txt\n"),
            ))
            .unwrap();
            sm.flush().unwrap();
            sm.current_session_id.clone().unwrap()
        };

        let cancel = AtomicBool::new(false);
        let out = history_tool(d.clone())
            .call(&json!({"session_id": id}), &cancel)
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["success"], true);
        let history = v["history"].as_array().unwrap();

        let asst = history
            .iter()
            .find(|m| m["role"] == "assistant" && m["tool_calls"].is_array())
            .expect("assistant with tool_calls should be in history");

        assert_eq!(asst["content"], "I will run a command");
        assert_eq!(asst["timestamp"].as_str().map(|s| !s.is_empty()), Some(true));
        let tcs = asst["tool_calls"].as_array().unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0]["name"], "run_shell");
        assert_eq!(tcs[0]["arguments"], r#"{"command":"ls"}"#);
        assert_eq!(asst["reasoning_content"], "I should check the directory");

        let tool = history
            .iter()
            .find(|m| m["role"] == "tool")
            .expect("tool result should be in history");
        assert_eq!(tool["content"], "file1.txt\nfile2.txt\n");
        assert_eq!(tool["tool_call_id"], "call_abc");
    }

    #[tokio::test]
    async fn history_truncates_long_content() {
        let d = tmp();
        let id = {
            let mut sm = SessionManager::new(&d);
            sm.create_session("task", None).unwrap();
            let content = "A".repeat(500);
            sm.add_message(crate::session::SessionMessage::from_chat(
                &ChatMessage::assistant(content.clone()),
            ))
            .unwrap();
            sm.flush().unwrap();
            sm.current_session_id.clone().unwrap()
        };

        let cancel = AtomicBool::new(false);
        let out = history_tool(d.clone())
            .call(&json!({"session_id": id, "max_content_chars": 100}), &cancel)
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        let history = v["history"].as_array().unwrap();
        let asst = history.iter().find(|m| m["role"] == "assistant").unwrap();
        let content = asst["content"].as_str().unwrap();
        assert!(content.contains("…(500 chars total)"), "should show truncation marker, got: {}", content);
        assert_eq!(content.chars().take(100).collect::<String>().len(), 100);
    }

    #[tokio::test]
    async fn history_truncates_tool_arguments() {
        let d = tmp();
        let id = {
            let mut sm = SessionManager::new(&d);
            sm.create_session("task", None).unwrap();
            let long_args = format!(r#"{{"code":"{}"}}"#, "x".repeat(500));
            let assistant = ChatMessage::assistant_with_tools(
                "run",
                vec![ToolCall {
                    id: "c1".into(),
                    name: "execute_python".into(),
                    arguments: long_args.clone(),
                }],
            );
            sm.add_message(crate::session::SessionMessage::from_chat(&assistant))
                .unwrap();
            sm.flush().unwrap();
            sm.current_session_id.clone().unwrap()
        };

        let cancel = AtomicBool::new(false);
        let out = history_tool(d.clone())
            .call(&json!({"session_id": id, "max_content_chars": 100}), &cancel)
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        let history = v["history"].as_array().unwrap();
        let asst = history
            .iter()
            .find(|m| m["role"] == "assistant" && m["tool_calls"].is_array())
            .expect("assistant with tool_calls should be in history");
        let tcs = asst["tool_calls"].as_array().unwrap();
        let args = tcs[0]["arguments"].as_str().unwrap();
        assert!(args.contains("…("), "tool_calls arguments should be truncated, got: {}", args);
    }

    #[tokio::test]
    async fn history_range_from_and_to() {
        let d = tmp();
        let id = {
            let mut sm = SessionManager::new(&d);
            sm.create_session("task", None).unwrap();
            for i in 0..10 {
                sm.add_message(crate::session::SessionMessage::from_chat(
                    &ChatMessage::user(format!("msg {}", i)),
                ))
                .unwrap();
            }
            sm.flush().unwrap();
            sm.current_session_id.clone().unwrap()
        };

        let cancel = AtomicBool::new(false);
        // range_from=2, range_to=5 should return messages 2,3,4 (3 messages)
        let out = history_tool(d.clone())
            .call(&json!({"session_id": id, "range_from": 2, "range_to": 5}), &cancel)
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["success"], true);
        assert_eq!(v["total_messages"], 11); // 1 user + 10 additions
        let history = v["history"].as_array().unwrap();
        assert_eq!(history.len(), 3, "range_from=2, range_to=5 should return 3 messages");
        assert_eq!(history[0]["content"], "msg 1");
        assert_eq!(history[2]["content"], "msg 3");
    }

    #[tokio::test]
    async fn history_max_content_chars_zero_means_no_truncation() {
        let d = tmp();
        let id = {
            let mut sm = SessionManager::new(&d);
            sm.create_session("task", None).unwrap();
            let content = "A".repeat(500);
            sm.add_message(crate::session::SessionMessage::from_chat(
                &ChatMessage::assistant(content.clone()),
            ))
            .unwrap();
            sm.flush().unwrap();
            sm.current_session_id.clone().unwrap()
        };

        let cancel = AtomicBool::new(false);
        let out = history_tool(d.clone())
            .call(&json!({"session_id": id, "max_content_chars": 0}), &cancel)
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        let history = v["history"].as_array().unwrap();
        let asst = history.iter().find(|m| m["role"] == "assistant").unwrap();
        let content = asst["content"].as_str().unwrap();
        assert_eq!(content.len(), 500, "max_content_chars=0 should mean no truncation, got len {}", content.len());
    }

    #[tokio::test]
    async fn new_and_continue_ack() {
        let d = tmp();
        let cancel = AtomicBool::new(false);
        let nt = NewSessionTool { project_path: d.clone() };
        let out = nt.call(&json!({"task": "test"}), &cancel).await.unwrap();
        assert!(out.contains("New session created"));
        let ct = ContinueSessionTool { project_path: d };
        let out = ct.call(&json!({"message": "followup"}), &cancel).await.unwrap();
        assert!(out.contains("true"));
    }

    #[tokio::test]
    async fn history_uses_active_session_when_no_session_id() {
        let d = tmp();
        let active_id = {
            let mut sm = SessionManager::new(&d);
            sm.create_session("active session task", None).unwrap();
            sm.current_session_id.clone().unwrap()
        };

        let cancel = AtomicBool::new(false);
        let holder = Arc::new(Mutex::new(Some(active_id.clone())));
        let tool = GetConversationHistoryTool {
            project_path: d.clone(),
            active_session_id: holder,
        };
        let out = tool.call(&json!({}), &cancel).await.unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["success"], true);
        assert_eq!(v["session_id"], active_id);
        assert_eq!(v["total_messages"], 1);
    }
}
