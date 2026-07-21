// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Todo tools — markdown-based todo lists under `.aacode/todos/`.
//!
//! Ported from Python `tools/todo_tools.py` + `utils/todo_manager.py`.
//! Each session gets its own todo file (`todo_<session>.md`); a shared
//! fallback file (`todo.md`) is used when no session is active.

use super::registry::Tool;
use super::schema::{ParamType, ToolParameter, ToolSchema};
use crate::error::Result;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Shared todo store. Uses `get_session_id` to pick the right file.
pub struct TodoStore {
    dir: PathBuf,
    next_id: std::sync::atomic::AtomicUsize,
    /// Returns the current session id, or None if no session active.
    get_session_id: Arc<dyn Fn() -> Option<String> + Send + Sync>,
}

impl TodoStore {
    pub fn new(project_path: &std::path::Path) -> Self {
        Self::with_session_resolver(project_path, Arc::new(|| None))
    }

    pub fn with_session_resolver(
        project_path: &std::path::Path,
        resolver: Arc<dyn Fn() -> Option<String> + Send + Sync>,
    ) -> Self {
        let dir = project_path.join(".aacode").join("todos");
        let _ = std::fs::create_dir_all(&dir);
        // Ensure a default file exists.
        let default = dir.join("todo.md");
        if !default.exists() {
            let _ = std::fs::write(&default, "# Todo List\n\n");
        }
        // Find max id across all todo files.
        let mut max = 0;
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().map(|x| x == "md").unwrap_or(false) {
                    if let Ok(c) = std::fs::read_to_string(&p) {
                        let m = max_existing_id(&c);
                        max = max.max(m);
                    }
                }
            }
        }
        TodoStore {
            dir,
            next_id: std::sync::atomic::AtomicUsize::new(max + 1),
            get_session_id: resolver,
        }
    }

    /// Resolve the todo file path: prefer session-specific, fallback to default.
    fn file_path(&self) -> PathBuf {
        let sid = (self.get_session_id)();
        if let Some(id) = sid {
            let p = self.dir.join(format!("todo_{id}.md"));
            if !p.exists() {
                // Create a fresh file for the new session.
                let _ = std::fs::write(&p, "# Todo List\n\n");
            }
            p
        } else {
            self.dir.join("todo.md")
        }
    }

    fn read(&self) -> String {
        std::fs::read_to_string(self.file_path()).unwrap_or_default()
    }

    fn write(&self, content: &str) {
        let _ = std::fs::write(self.file_path(), content);
    }

    fn add(&self, description: &str, priority: &str, category: &str) -> String {
        let id = format!("t{}", self.next_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst));
        let mut content = self.read();
        content.push_str(&format!(
            "- [ ] [#{id}] ({priority}/{category}): {description}\n"
        ));
        self.write(&content);
        id
    }

    fn mark_done(&self, todo_id: Option<&str>, pattern: Option<&str>) -> bool {
        let content = self.read();
        let (new_content, changed) = mark_in_content(&content, todo_id, pattern);
        if changed {
            self.write(&new_content);
        }
        changed
    }

    fn update(&self, old_pattern: &str, new_desc: &str) -> bool {
        let content = self.read();
        let mut out = String::new();
        let mut changed = false;
        for line in content.lines() {
            if !changed
                && line.trim_start().starts_with("- [")
                && line.to_lowercase().contains(&old_pattern.to_lowercase())
            {
                if let Some(pos) = line.rfind(": ") {
                    out.push_str(&line[..pos + 2]);
                    out.push_str(new_desc);
                    out.push('\n');
                    changed = true;
                    continue;
                }
            }
            out.push_str(line);
            out.push('\n');
        }
        if changed {
            self.write(&out);
        }
        changed
    }

    fn summary(&self) -> (usize, usize, usize) {
        let content = self.read();
        let mut total = 0;
        let mut done = 0;
        for line in content.lines() {
            let t = line.trim_start();
            if t.starts_with("- [ ]") {
                total += 1;
            } else if t.starts_with("- [x]") {
                total += 1;
                done += 1;
            }
        }
        (total, done, total - done)
    }

    /// List pending todo items for hint display (when mark fails).
    fn pending_hint(&self) -> String {
        let content = self.read();
        let mut pending = Vec::new();
        for line in content.lines() {
            let t = line.trim_start();
            if t.starts_with("- [ ]") {
                let re = regex::Regex::new(r"\[#(\w+)\]\s*:\s*(.*)").unwrap();
                if let Some(caps) = re.captures(t) {
                    pending.push((caps[1].to_string(), caps[2].to_string()));
                } else {
                    let desc = t
                        .replacen("- [ ]", "", 1)
                        .trim()
                        .to_string();
                    let short: String = desc.chars().take(80).collect();
                    pending.push((String::new(), short));
                }
            }
        }
        if pending.is_empty() {
            return String::new();
        }
        let mut hint = String::from("📋 Available pending todos:\n");
        for (tid, desc) in pending.iter().take(15) {
            if tid.is_empty() {
                hint.push_str(&format!("  - {desc}\n"));
            } else {
                hint.push_str(&format!("  #{tid}: {desc}\n"));
            }
        }
        if pending.len() > 15 {
            hint.push_str(&format!("  ... and {} more\n", pending.len() - 15));
        }
        hint.trim().to_string()
    }

    /// List all todo files in the directory.
    pub fn list_files(&self) -> Vec<String> {
        let mut files = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&self.dir) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if name.ends_with(".md") {
                    files.push(name);
                }
            }
        }
        files.sort();
        files
    }
}

fn max_existing_id(content: &str) -> usize {
    let mut max = 0;
    for line in content.lines() {
        if let Some(start) = line.find("[#t") {
            let rest = &line[start + 3..];
            let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(n) = num.parse::<usize>() {
                max = max.max(n);
            }
        }
    }
    max
}

fn mark_in_content(content: &str, todo_id: Option<&str>, pattern: Option<&str>) -> (String, bool) {
    let mut out = String::new();
    let mut changed = false;
    for line in content.lines() {
        let mut newline = line.to_string();
        if !changed && line.trim_start().starts_with("- [ ]") {
            let matches = match (todo_id, pattern) {
                (Some(id), _) => line.contains(&format!("[#{id}]")),
                (None, Some(p)) => line.to_lowercase().contains(&p.to_lowercase()),
                _ => false,
            };
            if matches {
                newline = line.replacen("- [ ]", "- [x]", 1);
                changed = true;
            }
        }
        out.push_str(&newline);
        out.push('\n');
    }
    (out, changed)
}

// ── Tools ──

pub struct AddTodoTool {
    pub store: Arc<TodoStore>,
}
impl Tool for AddTodoTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "add_todo_item",
            "Add a todo item. Returns a todo_id (e.g. t1) for later mark_todo_completed(todo_id=...).",
            vec![
                ToolParameter::new("description", ParamType::String, true, "Todo description", &["item", "task", "todo", "title"]),
                ToolParameter::new("priority", ParamType::String, false, "low|medium|high", &[]),
                ToolParameter::new("category", ParamType::String, false, "Category label", &[]),
            ],
        )
    }
    fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let desc = args.get("description").and_then(|v| v.as_str()).unwrap_or("");
        if desc.is_empty() {
            return Ok(json!({"success": false, "error": "missing description"}).to_string());
        }
        let priority = args.get("priority").and_then(|v| v.as_str()).unwrap_or("medium");
        let category = args.get("category").and_then(|v| v.as_str()).unwrap_or("Task");
        let id = self.store.add(desc, priority, category);
        Ok(json!({
            "success": true,
            "todo_id": id,
            "message": format!("Added [#{id}]: {desc} (use mark_todo_completed(todo_id=\"{id}\") when done)"),
        })
        .to_string())
    }
}

pub struct MarkTodoTool {
    pub store: Arc<TodoStore>,
}
impl Tool for MarkTodoTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "mark_todo_completed",
            "Mark a todo item complete. Prefer todo_id (from add_todo_item); item_pattern is a fallback.",
            vec![
                ToolParameter::new("todo_id", ParamType::String, false, "Todo id e.g. t1", &["id"]),
                ToolParameter::new("item_pattern", ParamType::String, false, "Fuzzy match text", &["title", "item", "task", "description", "text", "pattern", "todo"]),
            ],
        )
    }
    fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let id = args.get("todo_id").and_then(|v| v.as_str());
        let pat = args.get("item_pattern").and_then(|v| v.as_str());
        if id.is_none() && pat.is_none() {
            let hint = self.store.pending_hint();
            let err = if hint.is_empty() {
                "provide todo_id or item_pattern".to_string()
            } else {
                format!("provide todo_id or item_pattern\n\n{hint}")
            };
            return Ok(json!({"success": false, "error": err, "todo_id": id, "item_pattern": pat}).to_string());
        }
        let ok = self.store.mark_done(id, pat);
        Ok(json!({"success": ok, "todo_id": id, "item_pattern": pat}).to_string())
    }
}

pub struct UpdateTodoTool {
    pub store: Arc<TodoStore>,
}
impl Tool for UpdateTodoTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "update_todo_item",
            "Update a todo item's description.",
            vec![
                ToolParameter::new("old_pattern", ParamType::String, true, "Match text", &[]),
                ToolParameter::new("new_item", ParamType::String, true, "New description", &[]),
            ],
        )
    }
    fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let old = args.get("old_pattern").and_then(|v| v.as_str()).unwrap_or("");
        let new = args.get("new_item").and_then(|v| v.as_str()).unwrap_or("");
        let ok = self.store.update(old, new);
        Ok(json!({"success": ok}).to_string())
    }
}

pub struct TodoSummaryTool {
    pub store: Arc<TodoStore>,
}
impl Tool for TodoSummaryTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("get_todo_summary", "Get a summary of the todo list.", vec![])
    }
    fn call(&self, _args: &Value, _c: &AtomicBool) -> Result<String> {
        let (total, done, pending) = self.store.summary();
        let rate = if total > 0 { (done as f64 / total as f64) * 100.0 } else { 0.0 };
        Ok(json!({
            "success": true,
            "total_todos": total,
            "completed_todos": done,
            "pending_todos": pending,
            "completion_rate": rate,
        })
        .to_string())
    }
}

pub struct ListTodoFilesTool {
    pub store: Arc<TodoStore>,
}
impl Tool for ListTodoFilesTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("list_todo_files", "List all todo list files.", vec![])
    }
    fn call(&self, _args: &Value, _c: &AtomicBool) -> Result<String> {
        let files = self.store.list_files();
        Ok(json!({"success": true, "files": files, "count": files.len()}).to_string())
    }
}

pub struct AddExecutionRecordTool;
impl Tool for AddExecutionRecordTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "add_execution_record",
            "(Deprecated) Execution records merged into logging; silently succeeds.",
            vec![ToolParameter::new("record", ParamType::String, false, "record text", &["description", "details", "message", "summary", "content", "text", "note"])],
        )
    }
    fn call(&self, _args: &Value, _c: &AtomicBool) -> Result<String> {
        Ok(json!({"success": true, "message": "recorded"}).to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn store() -> (Arc<TodoStore>, PathBuf) {
        let d = std::env::temp_dir().join(format!(
            "aacode_todo_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&d).unwrap();
        (Arc::new(TodoStore::new(&d)), d)
    }

    #[test]
    fn add_returns_id_and_marks_done() {
        let (s, _) = store();
        let cancel = AtomicBool::new(false);
        let add = AddTodoTool { store: s.clone() };
        let out = add.call(&json!({"description": "write tests"}), &cancel).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        let id = v["todo_id"].as_str().unwrap().to_string();
        assert_eq!(id, "t1");

        let mark = MarkTodoTool { store: s.clone() };
        let out = mark.call(&json!({"todo_id": id}), &cancel).unwrap();
        assert_eq!(serde_json::from_str::<Value>(&out).unwrap()["success"], true);

        let sum = TodoSummaryTool { store: s.clone() };
        let out = sum.call(&json!({}), &cancel).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["total_todos"], 1);
        assert_eq!(v["completed_todos"], 1);
    }

    #[test]
    fn mark_by_pattern() {
        let (s, _) = store();
        let cancel = AtomicBool::new(false);
        AddTodoTool { store: s.clone() }.call(&json!({"description": "implement login"}), &cancel).unwrap();
        assert!(s.mark_done(None, Some("login")));
    }

    #[test]
    fn missing_id_gives_hint() {
        let (s, _) = store();
        let cancel = AtomicBool::new(false);
        AddTodoTool { store: s.clone() }.call(&json!({"description": "test"}), &cancel).unwrap();
        let mark = MarkTodoTool { store: s.clone() };
        let out = mark.call(&json!({}), &cancel).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["success"], false);
        // Should list pending todos.
        assert!(v["error"].as_str().unwrap().contains("Available"));
    }

    #[test]
    fn session_isolation() {
        let d = std::env::temp_dir().join(format!("aacode_sess_todo_{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&d).unwrap();

        // Simulate session switching via resolver.
        let session_id: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(Some("sess_a".to_string())));
        let rid = session_id.clone();
        let store = Arc::new(TodoStore::with_session_resolver(
            &d,
            Arc::new(move || rid.lock().unwrap().clone()),
        ));

        let cancel = AtomicBool::new(false);
        AddTodoTool { store: store.clone() }.call(&json!({"description": "task a"}), &cancel).unwrap();

        // Switch session.
        *session_id.lock().unwrap() = Some("sess_b".to_string());

        let out = AddTodoTool { store: store.clone() }.call(&json!({"description": "task b"}), &cancel).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["todo_id"], "t2"); // next_id is global (cross-session)

        // Back to sess_a, should see the t1 from earlier.
        *session_id.lock().unwrap() = Some("sess_a".to_string());
        let sum = TodoSummaryTool { store: store.clone() };
        let out = sum.call(&json!({}), &cancel).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["total_todos"], 1);
    }

    #[test]
    fn update_item() {
        let (s, _) = store();
        let cancel = AtomicBool::new(false);
        AddTodoTool { store: s.clone() }.call(&json!({"description": "old desc"}), &cancel).unwrap();
        let upd = UpdateTodoTool { store: s.clone() };
        assert!(serde_json::from_str::<Value>(
            &upd.call(&json!({"old_pattern": "old desc", "new_item": "new desc"}), &cancel).unwrap()
        ).unwrap()["success"].as_bool().unwrap());
        assert!(s.read().contains("new desc"));
    }
}
