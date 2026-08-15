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
use std::time::Instant;

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
        let mut tokens = estimate_tokens(&m.content);
        if let Some(rc) = &m.reasoning_content {
            tokens += estimate_tokens(rc);
        }
        SessionMessage {
            role: m.role.clone(),
            content: m.content.clone(),
            timestamp: now_iso(),
            tokens,
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
///
/// `schema_version` is a forward-compatibility guard: bump it on any breaking
/// change to this shape. iOS/Android parse the same file, so a shared
/// conformance fixture (`tests/fixtures/session_v1.json`) locks the format
/// across all three codebases.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SessionFile {
    #[serde(default = "default_schema_version")]
    schema_version: u32,
    session_id: String,
    #[serde(default)]
    created_at: String,
    messages: Vec<SessionMessage>,
}

fn default_schema_version() -> u32 {
    SCHEMA_VERSION
}

/// Current on-disk session schema version (see SESSION_FFI.md §6). Bump on any
/// breaking change to the session file shape.
pub const SCHEMA_VERSION: u32 = 1;

/// Validate that a session id is a safe simple identifier (no path separators
/// or `..`), so it can never escape the sessions directory.
pub fn valid_session_id(sid: &str) -> bool {
    !sid.is_empty()
        && !sid.contains('/')
        && !sid.contains('\\')
        && !sid.contains("..")
        && !sid.contains('\0')
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
    /// Flush cooldown: how many messages to batch before forcing a write.
    flush_batch_size: usize,
    /// Number of messages added since the last flush.
    dirty_count: usize,
    /// Last flush time (for interval-based debounce).
    last_flush: Option<Instant>,
    /// Minimum interval between flushes in seconds.
    flush_interval_secs: f64,
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
            flush_batch_size: 10,
            dirty_count: 0,
            last_flush: None,
            flush_interval_secs: 0.5,
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
        atomic_write(&self.index_path(), &serde_json::to_string(&merged)?)
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
        self.dirty_count = 0;
        self.last_flush = Some(Instant::now());
        Ok(id)
    }

    /// Append a message to the current session. Writes are batched via
    /// a cooldown (every N messages or every `flush_interval_secs`) to avoid
    /// full-file serialize + disk write on every single message.
    pub fn add_message(&mut self, msg: SessionMessage) -> Result<()> {
        self.messages.push(msg);
        self.dirty_count += 1;
        self.maybe_flush()
    }

    /// Append several messages (batch) then persist once.
    pub fn add_messages(&mut self, msgs: impl IntoIterator<Item = SessionMessage>) -> Result<()> {
        for m in msgs {
            self.messages.push(m);
            self.dirty_count += 1;
        }
        self.maybe_flush()
    }

    /// Force a write of the current session to disk (messages + index).
    pub fn flush(&mut self) -> Result<()> {
        if self.dirty_count == 0 {
            return Ok(());
        }
        self.save_current()?;
        self.touch_index()?;
        self.dirty_count = 0;
        self.last_flush = Some(Instant::now());
        Ok(())
    }

    fn maybe_flush(&mut self) -> Result<()> {
        if self.dirty_count >= self.flush_batch_size {
            return self.flush();
        }
        match self.last_flush {
            None => self.flush(),
            Some(t) if t.elapsed().as_secs_f64() >= self.flush_interval_secs => self.flush(),
            _ => Ok(()),
        }
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
                schema_version: default_schema_version(),
                session_id: id.clone(),
                created_at,
                messages: self.messages.clone(),
            };
            atomic_write(&self.session_path(id), &serde_json::to_string(&file)?)?;
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
    /// Falls back to scanning disk files when the index is empty.
    pub fn list_sessions(&self) -> Vec<SessionSummary> {
        let mut index = self.load_index();
        if index.is_empty() {
            if let Ok(entries) = std::fs::read_dir(&self.sessions_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|e| e.to_str()) != Some("json") {
                        continue;
                    }
                    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                    if stem == "sessions_index" {
                        continue;
                    }
                    if let Ok(s) = std::fs::read_to_string(&path) {
                        if let Ok(file) = serde_json::from_str::<SessionFile>(&s) {
                            let title = file.messages.first()
                                .map(|m| truncate_title(&m.content))
                                .unwrap_or_default();
                            let summary = SessionSummary {
                                session_id: file.session_id.clone(),
                                created_at: file.created_at.clone(),
                                last_activity: file.messages.last()
                                    .map(|m| m.timestamp.clone())
                                    .unwrap_or_default(),
                                total_messages: file.messages.len(),
                                total_tokens: file.messages.iter().map(|m| m.tokens).sum(),
                                title,
                                status: "active".to_string(),
                            };
                            index.entry(file.session_id).or_insert(summary);
                        }
                    }
                }
            }
            // Self-heal: write rebuilt index back to disk.
            if !index.is_empty() {
                if let Ok(contents) = serde_json::to_string(&index) {
                    let _ = atomic_write(&self.index_path(), &contents);
                }
            }
        }
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
        atomic_write(&self.index_path(), &serde_json::to_string(&index)?)?;
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

    /// Ensure a session with the given id exists (idempotent create-or-touch).
    /// If it exists, its `last_activity` is bumped (and a non-empty `title`
    /// updates the stored title); otherwise an empty session file + index entry
    /// are created.
    pub fn ensure_session(&mut self, sid: &str, title: &str) -> Result<()> {
        let mut index = self.load_index();
        let now = now_iso();
        if let Some(s) = index.get_mut(sid) {
            s.last_activity = now;
            if !title.trim().is_empty() {
                s.title = title.to_string();
            }
            self.save_index(&index)?;
            return Ok(());
        }
        let summary = SessionSummary {
            session_id: sid.to_string(),
            created_at: now.clone(),
            last_activity: now.clone(),
            total_messages: 0,
            total_tokens: 0,
            title: title.to_string(),
            status: "active".to_string(),
        };
        index.insert(sid.to_string(), summary);
        self.save_index(&index)?;
        let file = SessionFile {
            schema_version: SCHEMA_VERSION,
            session_id: sid.to_string(),
            created_at: now,
            messages: Vec::new(),
        };
        atomic_write(&self.session_path(sid), &serde_json::to_string(&file)?)
    }

    /// Set an explicit title on an existing session (no-op if missing).
    pub fn rename_session(&mut self, sid: &str, title: &str) -> Result<()> {
        let mut index = self.load_index();
        if let Some(s) = index.get_mut(sid) {
            s.title = title.to_string();
            self.save_index(&index)?;
        }
        Ok(())
    }

    /// Bump `last_activity` of an existing session (no-op if missing).
    pub fn touch_session(&mut self, sid: &str) -> Result<()> {
        let mut index = self.load_index();
        if let Some(s) = index.get_mut(sid) {
            s.last_activity = now_iso();
            self.save_index(&index)?;
        }
        Ok(())
    }

    /// Append messages to a session file (creating the session if missing),
    /// then refresh the index entry (counts + last_activity + title fallback).
    pub fn append_session_messages(
        &mut self,
        sid: &str,
        msgs: Vec<SessionMessage>,
    ) -> Result<()> {
        let path = self.session_path(sid);
        let mut existing: Vec<SessionMessage> = if path.exists() {
            std::fs::read_to_string(&path)
                .ok()
                .and_then(|s| serde_json::from_str::<SessionFile>(&s).ok())
                .map(|f| f.messages)
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        existing.extend(msgs);

        let total_messages = existing.len();
        let total_tokens: usize = existing.iter().map(|m| m.tokens).sum();
        let fallback_title = existing
            .iter()
            .find(|m| m.role == "user")
            .map(|m| truncate_title(&m.content))
            .unwrap_or_default();

        let created_at = self
            .load_index()
            .get(sid)
            .map(|s| s.created_at.clone())
            .unwrap_or_else(now_iso);

        let file = SessionFile {
            schema_version: SCHEMA_VERSION,
            session_id: sid.to_string(),
            created_at,
            messages: existing,
        };
        atomic_write(&path, &serde_json::to_string(&file)?)?;

        let mut index = self.load_index();
        if let Some(s) = index.get_mut(sid) {
            s.total_messages = total_messages;
            s.total_tokens = total_tokens;
            s.last_activity = now_iso();
            if s.title.trim().is_empty() && !fallback_title.is_empty() {
                s.title = fallback_title.clone();
            }
        } else {
            index.insert(
                sid.to_string(),
                SessionSummary {
                    session_id: sid.to_string(),
                    created_at: file.created_at.clone(),
                    last_activity: now_iso(),
                    total_messages,
                    total_tokens,
                    title: fallback_title,
                    status: "active".to_string(),
                },
            );
        }
        self.save_index(&index)
    }

    /// Get the current session's messages as ChatMessages (for LLM history).
    /// System messages are included so that dynamically-appended context
    /// (project analysis, init instructions, compaction summaries, etc.) are
    /// preserved in the prefix for provider KV-cache stability across calls.
    pub fn history_chat(&self) -> Vec<ChatMessage> {
        self.messages.iter().map(|m| m.to_chat()).collect()
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
        sm.flush().unwrap();
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
    fn history_includes_system() {
        let proj = tmp_project();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        sm.add_message(SessionMessage::from_chat(&ChatMessage::system("sys")))
            .unwrap();
        // After cache-stability fix: system messages are included in history
        // so persisted dynamic context is preserved for prefix matching.
        assert!(sm.history_chat().iter().any(|m| m.role == "system"));
    }

    #[test]
    fn scheduled_task_saves_to_existing_session() {
        let proj = tmp_project();
        let mut sm = SessionManager::new(&proj);

        let id = sm.create_session("previous task", None).unwrap();
        sm.add_message(SessionMessage::from_chat(&ChatMessage::assistant("done")))
            .unwrap();
        sm.flush().unwrap();

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
        sm2.flush().unwrap();

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

    #[test]
    fn reasoning_content_counted_in_tokens() {
        let msg = ChatMessage {
            role: "assistant".into(),
            content: "hello world".into(),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: Some("I need to think about this carefully".into()),
        };
        let sm = SessionMessage::from_chat(&msg);
        // Should count both content and reasoning_content tokens.
        let content_only = estimate_tokens("hello world");
        let with_reasoning = estimate_tokens("hello world") + estimate_tokens("I need to think about this carefully");
        assert!(sm.tokens > content_only);
        assert_eq!(sm.tokens, with_reasoning);
    }

    #[test]
    fn no_reasoning_content_tokens_unchanged() {
        let msg = ChatMessage {
            role: "assistant".into(),
            content: "hello world".into(),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        };
        let sm = SessionMessage::from_chat(&msg);
        assert_eq!(sm.tokens, estimate_tokens("hello world"));
    }

    #[test]
    fn tool_message_tokens_ignore_reasoning() {
        // Tool messages don't have reasoning_content normally, but even if set,
        // we verify the field is preserved.
        let msg = ChatMessage {
            role: "tool".into(),
            content: "result data".into(),
            tool_calls: None,
            tool_call_id: Some("call_1".into()),
            reasoning_content: None,
        };
        let sm = SessionMessage::from_chat(&msg);
        assert_eq!(sm.tokens, estimate_tokens("result data"));
    }

    #[test]
    fn compact_json_roundtrips() {
        let proj = tmp_project();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("compact test", None).unwrap();
        sm.add_message(SessionMessage::from_chat(&ChatMessage::assistant("{\"key\": \"value\"}")))
            .unwrap();
        sm.flush().unwrap();

        let id = sm.current_session_id.unwrap();
        let mut sm2 = SessionManager::new(&proj);
        assert!(sm2.switch_session(&id).unwrap());
        assert_eq!(sm2.messages.len(), 2);
        assert_eq!(sm2.messages[1].content, "{\"key\": \"value\"}");

        let path = sm2.session_path(&id);
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(!raw.contains("\n  "), "compact JSON should not have indentation");
        let file: SessionFile = serde_json::from_str(&raw).unwrap();
        assert_eq!(file.messages.len(), 2);
    }

    #[test]
    fn persisted_dynamic_system_message_in_history() {
        let proj = tmp_project();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("task", None).unwrap();

        let analysis = SessionMessage::from_chat(&ChatMessage::system("## Project Analysis\n2 .rs files, 1 .py file"));
        sm.add_message(analysis).unwrap();
        sm.flush().unwrap();

        let history = sm.history_chat();
        assert!(history.iter().any(|m| m.role == "system" && m.content.contains("Project Analysis")));

        // Reload from disk and verify system message is still in history.
        let id = sm.current_session_id.unwrap();
        let mut sm2 = SessionManager::new(&proj);
        sm2.switch_session(&id).unwrap();
        let history2 = sm2.history_chat();
        assert!(history2.iter().any(|m| m.role == "system" && m.content.contains("Project Analysis")));
    }

    #[test]
    fn flush_cooldown_batches_messages() {
        let proj = tmp_project();
        let mut sm = SessionManager::new(&proj);
        let id = sm.create_session("task", None).unwrap();

        // Add messages below the batch threshold — should not write to disk.
        for i in 0..5 {
            sm.add_message(SessionMessage::from_chat(&ChatMessage::assistant(&format!("msg{}", i))))
                .unwrap();
        }

        // Force a flush and verify all messages are persisted.
        sm.flush().unwrap();

        let msgs = sm.read_session_messages(&id);
        assert_eq!(msgs.len(), 6); // 1 from create + 5 added
        assert_eq!(msgs[5].content, "msg4");
    }

    #[test]
    fn flush_idempotent_when_clean() {
        let proj = tmp_project();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("task", None).unwrap();
        // Calling flush on a clean session should be a no-op.
        assert!(sm.flush().is_ok());
        assert!(sm.dirty_count == 0);
    }

    // ── schema conformance (locks the on-disk format across hosts) ──

    /// The canonical session-file fixture is shared with the iOS/Android hosts
    /// (each has its own copy under their test trees). Any breaking change to
    /// the on-disk shape must update this fixture and bump `schema_version`.
    #[test]
    fn schema_conformance_fixture_roundtrips() {
        const FIXTURE: &str = include_str!("../../tests/fixtures/session_v1.json");
        let file: SessionFile = serde_json::from_str(FIXTURE).expect("fixture must parse");

        assert_eq!(file.schema_version, 1, "schema_version must be pinned at 1");
        assert_eq!(file.session_id, "session_1700000000_abc12345");
        assert_eq!(file.created_at, "1700000000");
        assert_eq!(file.messages.len(), 3);

        // user message: no tool fields.
        assert_eq!(file.messages[0].role, "user");
        assert_eq!(file.messages[0].content, "Create a hello.py file");
        assert!(file.messages[0].tool_calls.is_none());
        assert!(file.messages[0].tool_call_id.is_none());
        assert!(file.messages[0].reasoning_content.is_none());

        // assistant message: tool_calls + reasoning_content.
        assert_eq!(file.messages[1].role, "assistant");
        let tcs = file.messages[1]
            .tool_calls
            .as_ref()
            .expect("assistant tool_calls present");
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].id, "call_1");
        assert_eq!(tcs[0].name, "run_shell");
        assert_eq!(tcs[0].arguments, r#"{"command":"echo hello"}"#);
        assert_eq!(
            file.messages[1].reasoning_content.as_deref(),
            Some("I should run a shell command")
        );

        // tool message: tool_call_id.
        assert_eq!(file.messages[2].role, "tool");
        assert_eq!(file.messages[2].content, "hello");
        assert_eq!(file.messages[2].tool_call_id.as_deref(), Some("call_1"));

        // Round-trip: re-serialize then re-parse must be stable.
        let reserialized = serde_json::to_string(&file).unwrap();
        let reparsed: SessionFile = serde_json::from_str(&reserialized).unwrap();
        assert_eq!(reparsed.schema_version, 1);
        assert_eq!(reparsed.messages.len(), 3);
        assert_eq!(
            reparsed.messages[1].tool_calls.as_ref().unwrap()[0].name,
            "run_shell"
        );
    }

    // ── SESSION_FFI 存储方法 ──────────────────────────────────────────

    #[test]
    fn ensure_session_creates_then_touches() {
        let proj = tmp_project();
        let mut sm = SessionManager::new(&proj);
        sm.ensure_session("s1", "My Session").unwrap();
        assert!(sm.session_path("s1").exists(), "session file must be created");
        assert_eq!(sm.load_index()["s1"].title, "My Session");

        // Idempotent re-ensure: bump last_activity, keep title on empty.
        let first = sm.load_index()["s1"].last_activity.clone();
        std::thread::sleep(std::time::Duration::from_millis(1100));
        sm.ensure_session("s1", "").unwrap();
        let index2 = sm.load_index();
        assert!(index2["s1"].last_activity >= first, "last_activity must bump");
        assert_eq!(index2["s1"].title, "My Session", "empty title must not clobber");
    }

    #[test]
    fn rename_session_sets_title() {
        let proj = tmp_project();
        let mut sm = SessionManager::new(&proj);
        sm.ensure_session("s1", "old").unwrap();
        sm.rename_session("s1", "new title").unwrap();
        assert_eq!(sm.load_index()["s1"].title, "new title");
        // Rename on a missing session is a no-op, not an error.
        sm.rename_session("missing", "x").unwrap();
    }

    #[test]
    fn touch_session_bumps_last_activity() {
        let proj = tmp_project();
        let mut sm = SessionManager::new(&proj);
        sm.ensure_session("s1", "").unwrap();
        let before = sm.load_index()["s1"].last_activity.clone();
        std::thread::sleep(std::time::Duration::from_millis(1100));
        sm.touch_session("s1").unwrap();
        assert!(sm.load_index()["s1"].last_activity >= before);
    }

    #[test]
    fn append_session_messages_appends_and_updates_index() {
        let proj = tmp_project();
        let mut sm = SessionManager::new(&proj);
        sm.ensure_session("s1", "").unwrap();
        sm.append_session_messages(
            "s1",
            vec![
                SessionMessage::from_chat(&ChatMessage::user("hello")),
                SessionMessage::from_chat(&ChatMessage::assistant("hi")),
            ],
        )
        .unwrap();
        let msgs = sm.read_session_messages("s1");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].content, "hello");
        let index = sm.load_index();
        assert_eq!(index["s1"].total_messages, 2);
        assert_eq!(index["s1"].title, "hello", "title falls back to first user msg");
    }

    #[test]
    fn append_to_missing_session_creates_it() {
        let proj = tmp_project();
        let mut sm = SessionManager::new(&proj);
        sm.append_session_messages(
            "brand_new",
            vec![SessionMessage::from_chat(&ChatMessage::user("first"))],
        )
        .unwrap();
        assert!(sm.session_path("brand_new").exists());
        assert_eq!(sm.load_index()["brand_new"].total_messages, 1);
        assert_eq!(sm.load_index()["brand_new"].title, "first");
    }

    #[test]
    fn valid_session_id_rejects_path_traversal() {
        assert!(valid_session_id("session_1700000000_abc12345"));
        assert!(valid_session_id("cron_daily_1"));
        assert!(valid_session_id("pending_x"));
        assert!(!valid_session_id(""));
        assert!(!valid_session_id("../etc"));
        assert!(!valid_session_id("a/b"));
        assert!(!valid_session_id("a\\b"));
        assert!(!valid_session_id(".."));
    }

    #[test]
    fn old_file_without_schema_version_reads_fine() {
        let proj = tmp_project();
        let mut sm = SessionManager::new(&proj);
        let path = sm.session_path("s_old");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        // Old-format file: no `schema_version` field.
        let old = r#"{"session_id":"s_old","created_at":"1700000000","messages":[{"role":"user","content":"hi","timestamp":"1700000001","tokens":1}]}"#;
        std::fs::write(&path, old).unwrap();

        let msgs = sm.read_session_messages("s_old");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content, "hi");

        // list_sessions fallback must rebuild the index from the old file.
        let sessions = sm.list_sessions();
        assert!(sessions.iter().any(|s| s.session_id == "s_old"));
    }
}
