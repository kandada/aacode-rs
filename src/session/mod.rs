// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Session persistence: `.aacode/sessions/<id>.json` + `sessions_index.json`.
//!
//! Ported from Python `utils/session_manager.py`. Uses atomic writes
//! (temp file + rename). Messages carry structured tool_calls / tool_call_id /
//! reasoning_content, matching the LLM `ChatMessage` shape.

use crate::error::{AacodeError, Result};
use crate::llm::types::{ChatMessage, ToolCall};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A persisted message (superset of ChatMessage with a timestamp + tokens).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub timestamp: String,
    #[serde(default)]
    pub tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub reasoning_content: Option<String>,
}

impl SessionMessage {
    pub fn from_chat(m: &ChatMessage) -> Self {
        SessionMessage {
            role: m.role.clone(),
            content: m.content.clone(),
            timestamp: now_iso(),
            tokens: estimate_tokens(&m.content),
            tool_calls: m.tool_calls.clone(),
            tool_call_id: m.tool_call_id.clone(),
            reasoning_content: m.reasoning_content.clone(),
        }
    }

    pub fn to_chat(&self) -> ChatMessage {
        ChatMessage {
            role: self.role.clone(),
            content: self.content.clone(),
            tool_calls: self.tool_calls.clone(),
            tool_call_id: self.tool_call_id.clone(),
            reasoning_content: self.reasoning_content.clone(),
        }
    }
}

/// One entry in the sessions index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub session_id: String,
    pub created_at: String,
    pub last_activity: String,
    #[serde(default)]
    pub total_messages: usize,
    #[serde(default)]
    pub total_tokens: usize,
    #[serde(default)]
    pub title: String,
    #[serde(default = "default_status")]
    pub status: String,
}

fn default_status() -> String {
    "active".to_string()
}

/// The on-disk session file shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SessionFile {
    session_id: String,
    #[serde(default)]
    created_at: String,
    messages: Vec<SessionMessage>,
}

/// Rough token estimate (4 chars ≈ 1 token). tiktoken is not ported.
pub fn estimate_tokens(text: &str) -> usize {
    text.chars().count() / 4
}

/// Current time as a compact ISO-ish string (no external chrono dependency).
pub fn now_iso() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Store epoch seconds as string; readable enough and monotonic for sorting.
    secs.to_string()
}

/// Atomic write: write to a temp file then rename over the target.
fn atomic_write(path: &Path, contents: &str) -> Result<()> {
    let parent = path.parent().ok_or_else(|| AacodeError::Io("no parent".into()))?;
    std::fs::create_dir_all(parent)?;
    let tmp = parent.join(format!(".tmp_{}", uuid::Uuid::new_v4().simple()));
    std::fs::write(&tmp, contents)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

pub struct SessionManager {
    sessions_dir: PathBuf,
    pub current_session_id: Option<String>,
    pub messages: Vec<SessionMessage>,
}

impl SessionManager {
    pub fn new(project_path: &Path) -> Self {
        let sessions_dir = project_path.join(".aacode").join("sessions");
        // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0
        let _ = std::fs::create_dir_all(&sessions_dir);
        SessionManager {
            sessions_dir,
            current_session_id: None,
            messages: Vec::new(),
        }
    }

    fn index_path(&self) -> PathBuf {
        self.sessions_dir.join("sessions_index.json")
    }

    fn session_path(&self, id: &str) -> PathBuf {
        self.sessions_dir.join(format!("{id}.json"))
    }

    fn load_index(&self) -> std::collections::BTreeMap<String, SessionSummary> {
        let path = self.index_path();
        if let Ok(s) = std::fs::read_to_string(&path) {
            if let Ok(m) = serde_json::from_str(&s) {
                return m;
            }
        }
        std::collections::BTreeMap::new()
    }

    fn save_index(&self, index: &std::collections::BTreeMap<String, SessionSummary>) -> Result<()> {
        // Merge with disk to avoid clobbering concurrent writers.
        let mut merged = self.load_index();
        for (k, v) in index {
            merged.insert(k.clone(), v.clone());
        }
        atomic_write(&self.index_path(), &serde_json::to_string_pretty(&merged)?)
    }

    /// Create a new session and make it current. Returns the session id.
    pub fn create_session(&mut self, task: &str, title: Option<&str>) -> Result<String> {
        let id = format!("session_{}_{}", now_iso(), &uuid::Uuid::new_v4().simple().to_string()[..8]);
        let title = title
            .map(|t| t.to_string())
            .unwrap_or_else(|| truncate_title(task));
        let now = now_iso();
        let summary = SessionSummary {
            session_id: id.clone(),
            created_at: now.clone(),
            last_activity: now,
            total_messages: 0,
            total_tokens: 0,
            title,
            status: "active".to_string(),
        };
        self.current_session_id = Some(id.clone());
        self.messages.clear();
        if !task.trim().is_empty() {
            self.messages.push(SessionMessage {
                role: "user".to_string(),
                content: task.to_string(),
                timestamp: now_iso(),
                tokens: estimate_tokens(task),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            });
        }
        let mut index = self.load_index();
        let mut s = summary;
        s.total_messages = self.messages.len();
        s.total_tokens = self.total_tokens();
        index.insert(id.clone(), s);
        self.save_index(&index)?;
        self.save_current()?;
        Ok(id)
    }

    /// Append a message to the current session and persist.
    pub fn add_message(&mut self, msg: SessionMessage) -> Result<()> {
        self.messages.push(msg);
        self.save_current()?;
        self.touch_index()?;
        Ok(())
    }

    /// Append several messages (batch) then persist once.
    pub fn add_messages(&mut self, msgs: impl IntoIterator<Item = SessionMessage>) -> Result<()> {
        for m in msgs {
            self.messages.push(m);
        }
        self.save_current()?;
        self.touch_index()?;
        Ok(())
    }

    fn total_tokens(&self) -> usize {
        self.messages.iter().map(|m| m.tokens).sum()
    }

    fn touch_index(&self) -> Result<()> {
        if let Some(id) = &self.current_session_id {
            let mut index = self.load_index();
            if let Some(s) = index.get_mut(id) {
                s.last_activity = now_iso();
                s.total_messages = self.messages.len();
                s.total_tokens = self.total_tokens();
                if s.title.trim().is_empty() {
                    if let Some(first_user) = self.messages.iter().find(|m| m.role == "user") {
                        s.title = truncate_title(&first_user.content);
                    }
                }
            }
            self.save_index(&index)?;
        }
        Ok(())
    }

    fn save_current(&self) -> Result<()> {
        if let Some(id) = &self.current_session_id {
            let created_at = self
                .load_index()
                .get(id)
                .map(|s| s.created_at.clone())
                .unwrap_or_else(now_iso);
            let file = SessionFile {
                session_id: id.clone(),
                created_at,
                messages: self.messages.clone(),
            };
            atomic_write(&self.session_path(id), &serde_json::to_string_pretty(&file)?)?;
        }
        Ok(())
    }

    /// Load an existing session as current.
    pub fn switch_session(&mut self, id: &str) -> Result<bool> {
        let path = self.session_path(id);
        if !path.exists() {
            return Ok(false);
        }
        let s = std::fs::read_to_string(&path)?;
        let file: SessionFile = serde_json::from_str(&s)?;
        self.current_session_id = Some(id.to_string());
        self.messages = file.messages;
        Ok(true)
    }

    /// List all sessions, newest activity first.
    pub fn list_sessions(&self) -> Vec<SessionSummary> {
        let index = self.load_index();
        let mut v: Vec<SessionSummary> = index.into_values().collect();
        v.sort_by(|a, b| b.last_activity.cmp(&a.last_activity));
        v
    }

    /// Delete a session file + index entry.
    pub fn delete_session(&mut self, id: &str) -> Result<bool> {
        let mut index = self.load_index();
        if index.remove(id).is_none() {
            return Ok(false);
        }
        let _ = std::fs::remove_file(self.session_path(id));
        // rewrite index without the removed key (save_index merges, so write raw)
        atomic_write(&self.index_path(), &serde_json::to_string_pretty(&index)?)?;
        if self.current_session_id.as_deref() == Some(id) {
            self.current_session_id = None;
            self.messages.clear();
        }
        Ok(true)
    }

    /// Read messages of a session file directly (without switching).
    pub fn read_session_messages(&self, id: &str) -> Vec<SessionMessage> {
        let path = self.session_path(id);
        if let Ok(s) = std::fs::read_to_string(&path) {
            if let Ok(file) = serde_json::from_str::<SessionFile>(&s) {
                return file.messages;
            }
        }
        Vec::new()
    }

    /// Get the current session's messages as ChatMessages (for LLM history).
    pub fn history_chat(&self) -> Vec<ChatMessage> {
        self.messages
            .iter()
            .filter(|m| m.role != "system")
            .map(|m| m.to_chat())
            .collect()
    }

    /// Whether the last message is a tool result (interrupted mid-execution).
    pub fn ended_mid_tool(&self) -> bool {
        self.messages
            .last()
            .map(|m| m.role == "tool")
            .unwrap_or(false)
    }
}

fn truncate_title(task: &str) -> String {
    let t: String = task.chars().take(50).collect();
    if task.chars().count() > 50 {
        format!("{t}...")
    } else {
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_project() -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "aacode_sess_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn create_and_reload() {
        let proj = tmp_project();
        let mut sm = SessionManager::new(&proj);
        let id = sm.create_session("do something big", None).unwrap();
        assert!(id.starts_with("session_"));
        assert_eq!(sm.messages.len(), 1);

        // new manager, switch to it
        let mut sm2 = SessionManager::new(&proj);
        assert!(sm2.switch_session(&id).unwrap());
        assert_eq!(sm2.messages[0].content, "do something big");
    }

    #[test]
    fn append_and_persist() {
        let proj = tmp_project();
        let mut sm = SessionManager::new(&proj);
        let id = sm.create_session("task", None).unwrap();
        sm.add_message(SessionMessage::from_chat(&ChatMessage::assistant("hi")))
            .unwrap();
        let read = sm.read_session_messages(&id);
        assert_eq!(read.len(), 2);
        assert_eq!(read[1].role, "assistant");
    }

    #[test]
    fn tool_calls_persist() {
        let proj = tmp_project();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        let msg = ChatMessage::assistant_with_tools(
            "",
            vec![ToolCall {
                id: "c1".into(),
                name: "run_shell".into(),
                arguments: "{}".into(),
            }],
        );
        sm.add_message(SessionMessage::from_chat(&msg)).unwrap();
        sm.add_message(SessionMessage::from_chat(&ChatMessage::tool_result("c1", "ok")))
            .unwrap();
        assert!(sm.ended_mid_tool());
        let hist = sm.history_chat();
        assert!(hist.iter().any(|m| m.tool_calls.is_some()));
    }

    #[test]
    fn list_and_delete() {
        let proj = tmp_project();
        let mut sm = SessionManager::new(&proj);
        let id1 = sm.create_session("one", None).unwrap();
        let _id2 = sm.create_session("two", None).unwrap();
        assert!(sm.list_sessions().len() >= 2);
        assert!(sm.delete_session(&id1).unwrap());
        assert!(!sm.list_sessions().iter().any(|s| s.session_id == id1));
    }

    #[test]
    fn history_excludes_system() {
        let proj = tmp_project();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        sm.add_message(SessionMessage::from_chat(&ChatMessage::system("sys")))
            .unwrap();
        assert!(sm.history_chat().iter().all(|m| m.role != "system"));
    }

    #[test]
    fn scheduled_task_saves_to_existing_session() {
        let proj = tmp_project();
        let mut sm = SessionManager::new(&proj);

        let id = sm.create_session("previous task", None).unwrap();
        sm.add_message(SessionMessage::from_chat(&ChatMessage::assistant("done")))
            .unwrap();

        let mut sm2 = SessionManager::new(&proj);
        assert!(sm2.switch_session(&id).unwrap());
        assert!(sm2.current_session_id.is_some());

        let scheduled =
            "[Scheduled: daily-check]\nThis is an automated scheduled run.\n\ncheck git status";

        let already = sm2
            .messages
            .last()
            .map(|m| m.role == "user" && m.content == scheduled)
            .unwrap_or(false);
        assert!(!already);

        sm2.add_message(SessionMessage::from_chat(&ChatMessage::user(scheduled)))
            .unwrap();

        let msgs = sm2.read_session_messages(&id);
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].content, "previous task");
        assert_eq!(msgs[1].role, "assistant");
        assert_eq!(msgs[2].role, "user");
        assert!(msgs[2].content.starts_with("[Scheduled: daily-check]"));
    }

    #[test]
    fn dedup_skips_duplicate_user_message() {
        let proj = tmp_project();
        let mut sm = SessionManager::new(&proj);

        let task = "[Scheduled: daily]\ncheck";
        let id = sm.create_session(task, None).unwrap();

        let mut sm2 = SessionManager::new(&proj);
        sm2.switch_session(&id).unwrap();
        let already = sm2
            .messages
            .last()
            .map(|m| m.role == "user" && m.content == task)
            .unwrap_or(false);
        assert!(already);

        let msgs = sm2.read_session_messages(&id);
        assert_eq!(msgs.len(), 1);
    }
}
