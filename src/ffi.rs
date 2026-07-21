// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! C ABI for embedding aacode-rs into the Android app (via `jni_glue.c`) and
//! other hosts. Mirrors the design in `design.md` §11.
//!
//! Contract:
//!   * `aacode_register_stream_callback(cb)` — cb receives each JSONL event line.
//!   * `aacode_run_task(task_json)` — blocking; streams events via the callback;
//!     returns a JSON result string (caller frees with `aacode_free_string`).
//!   * `aacode_run_task_with_cb(task_json, cb)` — per-task callback; safe for
//!     concurrent tasks. `task_json` may include `"client_task_id": "<id>"` to
//!     enable per-task cancellation via `aacode_cancel_task`.
//!   * `aacode_cancel_task(task_id)` — cancels only the task registered with
//!     that `client_task_id`.
//!   * `aacode_cancel()` — cancels ALL in-flight tasks (legacy behavior).
//!   * `aacode_validate_api_key(config_json)` — returns `{"valid":bool,...}`.
//!   * `aacode_list_sessions(project_path)` / `aacode_get_session_messages(...)`.
//!   * `aacode_free_string(ptr)` — free any returned string.

// C-ABI surface: every extern fn validates pointers (null-checked, then
// CStr::from_ptr) per the documented host contract; `unsafe fn` would not
// change the C caller's obligations.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::config::AgentConfig;
use crate::runtime::AgentRuntime;
use crate::session::SessionManager;
use crate::stream::CallbackSink;
#[cfg(test)]
use crate::stream::EventSink;
use serde_json::json;
use std::collections::HashMap;
use std::ffi::{c_char, CStr, CString};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// The C stream callback type: receives one NUL-terminated UTF-8 line.
pub type StreamCallback = extern "C" fn(*const c_char);

// Global state: the registered (legacy) callback + per-task cancel flags.
static CALLBACK: OnceLock<Mutex<Option<StreamCallback>>> = OnceLock::new();
/// Per-task cancel flags, keyed by client_task_id (or an auto-generated id).
/// Each task owns its own flag — cancelling one task never affects others,
/// and starting a new task never clears a pending cancel for another.
static CANCEL_MAP: OnceLock<Mutex<HashMap<String, Arc<AtomicBool>>>> = OnceLock::new();
/// Monotonic counter for auto-generated task ids.
static TASK_SEQ: AtomicU64 = AtomicU64::new(1);

fn callback_slot() -> &'static Mutex<Option<StreamCallback>> {
    CALLBACK.get_or_init(|| Mutex::new(None))
}

fn cancel_map() -> &'static Mutex<HashMap<String, Arc<AtomicBool>>> {
    CANCEL_MAP.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Register a fresh cancel flag for `task_id`. If a flag with the same id is
/// already registered (duplicate client id), the same flag is shared so a
/// cancel reaches both.
fn register_cancel_flag(task_id: &str) -> Arc<AtomicBool> {
    let mut map = cancel_map().lock().unwrap_or_else(|e| e.into_inner());
    map.entry(task_id.to_string())
        .or_insert_with(|| Arc::new(AtomicBool::new(false)))
        .clone()
}

/// Remove the flag when the task finishes (avoids unbounded growth).
fn deregister_cancel_flag(task_id: &str) {
    let mut map = cancel_map().lock().unwrap_or_else(|e| e.into_inner());
    map.remove(task_id);
}

/// RAII guard so the flag is deregistered even on early return.
struct CancelRegistration {
    task_id: String,
    flag: Arc<AtomicBool>,
}

impl CancelRegistration {
    fn new(task_id: String) -> Self {
        let flag = register_cancel_flag(&task_id);
        CancelRegistration { task_id, flag }
    }
}

impl Drop for CancelRegistration {
    fn drop(&mut self) {
        deregister_cancel_flag(&self.task_id);
    }
}

/// In-flight session ids (project_path + session_id). Two concurrent tasks on
/// the SAME session would corrupt the session file (last-writer-wins), so the
/// second submission is rejected up front.
static RUNNING_SESSIONS: OnceLock<Mutex<std::collections::HashSet<String>>> = OnceLock::new();

fn running_sessions() -> &'static Mutex<std::collections::HashSet<String>> {
    RUNNING_SESSIONS.get_or_init(|| Mutex::new(std::collections::HashSet::new()))
}

/// RAII guard marking a session as busy. `try_acquire` returns None when the
/// session already has an in-flight task.
struct SessionGuard {
    key: String,
}

impl SessionGuard {
    fn try_acquire(key: String) -> Option<Self> {
        let mut set = running_sessions().lock().unwrap_or_else(|e| e.into_inner());
        if set.contains(&key) {
            return None;
        }
        set.insert(key.clone());
        Some(SessionGuard { key })
    }
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        let mut set = running_sessions().lock().unwrap_or_else(|e| e.into_inner());
        set.remove(&self.key);
    }
}

/// An EventSink that forwards each line to the registered C callback.
/// Production code streams through per-task closures (`run_task_with_cb`);
/// this sink only backs the legacy-callback unit test.
#[cfg(test)]
struct FfiSink;

#[cfg(test)]
impl EventSink for FfiSink {
    fn emit_line(&self, line: &str) {
        if let Ok(guard) = callback_slot().lock() {
            if let Some(cb) = *guard {
                if let Ok(cs) = CString::new(line) {
                    cb(cs.as_ptr());
                }
            }
        }
    }
    fn is_tty(&self) -> bool {
        false
    }
}

/// Convert a C string pointer to a Rust &str safely.
unsafe fn cstr<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    CStr::from_ptr(ptr).to_str().ok()
}

/// Allocate a C string from a Rust String (caller frees via aacode_free_string).
fn to_c_string(s: String) -> *mut c_char {
    CString::new(s)
        .unwrap_or_else(|_| CString::new("").unwrap())
        .into_raw()
}

/// Register (or clear with null) the stream callback.
#[no_mangle]
pub extern "C" fn aacode_register_stream_callback(cb: Option<StreamCallback>) {
    // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0
    if let Ok(mut guard) = callback_slot().lock() {
        *guard = cb;
    }
}

/// Run a task described by `task_json`. Blocks until completion. Streams events
/// through the registered callback. Returns a JSON result string.
/// **Deprecated**: use `aacode_run_task_with_cb` instead to support concurrency.
#[no_mangle]
pub extern "C" fn aacode_run_task(task_json: *const c_char) -> *mut c_char {
    // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0
    let input = match unsafe { cstr(task_json) } {
        Some(s) => s.to_string(),
        None => return to_c_string(json!({"status": "error", "error": "null task_json"}).to_string()),
    };
    run_task_with_cb(&input, |line| {
        if let Ok(cs) = CString::new(line) {
            if let Ok(guard) = callback_slot().lock() {
                if let Some(cb) = *guard { cb(cs.as_ptr()); }
            }
        }
    })
}

/// Run a task described by `task_json` with a **per-task** streaming callback.
/// Supports concurrent tasks — each caller provides its own callback, no shared
/// global state is used. Blocks until completion. Returns a JSON result string.
#[no_mangle]
pub extern "C" fn aacode_run_task_with_cb(
    task_json: *const c_char,
    cb: StreamCallback,
) -> *mut c_char {
    // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0
    let input = match unsafe { cstr(task_json) } {
        Some(s) => s.to_string(),
        None => return to_c_string(json!({"status": "error", "error": "null task_json"}).to_string()),
    };
    run_task_with_cb(&input, move |line| {
        if let Ok(cs) = CString::new(line) {
            cb(cs.as_ptr());
        }
    })
}

/// Decide the shell sandbox root for the SDK (mobile/embedded host) path.
/// Returns `(shell_root, cd_rel)`: the VFS root to init fastshell with, and
/// an optional sandbox-relative path to `cd` into right after init.
///
/// Default: the shell is confined to the project. When the host configured a
/// user skills directory (`config.skills.user_dir`) the agent must be able to
/// manage skills via run_shell — those live OUTSIDE the project — so the
/// shell root is widened to the whole host sandbox and the session cwd is
/// positioned at the project instead. Hosts that don't pass skills.user_dir
/// keep the old strictly-project-confined behavior.
///
/// Paths are canonicalized and compared **component-wise** (`Path::strip_prefix`),
/// not by string prefix:
///  * Android hands out `/data/user/0/<pkg>/…` while the canonical form is
///    `/data/data/<pkg>/…` (symlink); iOS/macOS have `/var` → `/private/var`.
///  * A sibling directory sharing a name prefix (`/sb` vs `/sb2/proj`) must
///    NOT be treated as inside the sandbox — a string check would wrongly
///    widen the root and then fail the `cd`.
fn resolve_shell_root(
    sandbox: &str,
    project: &str,
    skills_dir_configured: bool,
) -> (String, Option<String>) {
    if !skills_dir_configured {
        return (project.to_string(), None);
    }
    let canon = |s: &str| {
        std::fs::canonicalize(s).unwrap_or_else(|_| std::path::PathBuf::from(s))
    };
    let sandbox_c = canon(sandbox);
    let project_c = canon(project);
    match project_c.strip_prefix(&sandbox_c) {
        Ok(rel) if !rel.as_os_str().is_empty() => (
            sandbox.to_string(),
            Some(rel.to_string_lossy().to_string()),
        ),
        // Equal to the sandbox (already widest) or outside it (own jail).
        _ => (project.to_string(), None),
    }
}

/// Shared implementation: parse JSON, build runtime, run, return JSON result.
/// The `emit` closure handles streaming output — it is invoked once per line.
fn run_task_with_cb(input: &str, emit: impl Fn(&str) + Send + Sync + 'static) -> *mut c_char {
    let v: serde_json::Value = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(e) => return to_c_string(json!({"status": "error", "error": format!("bad task_json: {e}")}).to_string()),
    };
    let task = v.get("task").and_then(|x| x.as_str()).unwrap_or("");
    if task.is_empty() {
        return to_c_string(json!({"status": "error", "error": "missing task"}).to_string());
    }
    let project_path = v.get("project_path").and_then(|x| x.as_str()).unwrap_or(".").to_string();
    let session_id = v.get("session_id").and_then(|x| x.as_str()).map(|s| s.to_string());

    // Per-task cancel flag. The host may pass "client_task_id" to enable
    // targeted cancellation (aacode_cancel_task); otherwise an internal id is
    // generated. The flag is removed when this function returns.
    let task_id = v
        .get("client_task_id")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("__auto_{}", TASK_SEQ.fetch_add(1, Ordering::SeqCst)));
    let registration = CancelRegistration::new(task_id);
    let cancel = registration.flag.clone();

    // Reject a second concurrent task on the same session (would corrupt the
    // session file). New sessions (no session_id) are always allowed.
    let _session_guard = match &session_id {
        Some(sid) => {
            let key = format!("{project_path}::{sid}");
            match SessionGuard::try_acquire(key) {
                Some(g) => Some(g),
                None => {
                    return to_c_string(
                        json!({"status": "error", "error": format!("session '{sid}' already has a task running")}).to_string(),
                    );
                }
            }
        }
        None => None,
    };

    let mut config: AgentConfig = serde_json::from_value(v.clone()).unwrap_or_default();
    config.apply_env();

    let rt = if let Some(sdk) = fastshell::sdk::try_get_sdk_instance() {
        let sandbox = { let guard = sdk.lock().unwrap_or_else(|e| e.into_inner()); guard.vfs_root() };
        let pp = if project_path == "." { sandbox.clone() } else { project_path.clone() };
        // Shell sandbox root + optional cwd reposition (see resolve_shell_root).
        let (shell_root, cd_rel) =
            resolve_shell_root(&sandbox, &pp, config.skills.user_dir.is_some());
        let fs_arc: std::sync::Arc<std::sync::Mutex<fastshell::Fastshell>> =
            std::sync::Arc::new(std::sync::Mutex::new(fastshell::Fastshell::new()));
        {
            let mut ours = fs_arc.lock().unwrap();
            let mut fscfg = fastshell::Config::default();
            fscfg.sandbox_path = shell_root.clone();
            fscfg.python_enabled = true;
            if let Err(e) = ours.init(fscfg) {
                return to_c_string(json!({"status": "error", "error": e}).to_string());
            }
            if let Some(rel) = cd_rel {
                // Position the persistent shell session inside the project so
                // relative paths behave exactly as before the root widening.
                let r = ours.execute(&format!("cd '{}'", rel.replace('\'', "'\\''")));
                if r.exit_code != 0 {
                    return to_c_string(
                        json!({"status": "error", "error": format!("cd to project failed: {}", r.stderr)}).to_string(),
                    );
                }
            }
        }
        AgentRuntime::with_fastshell(config, std::path::PathBuf::from(pp), fs_arc)
    } else {
        match AgentRuntime::init(config, std::path::PathBuf::from(project_path)) {
            Ok(r) => r,
            Err(e) => return to_c_string(json!({"status": "error", "error": e.to_string()}).to_string()),
        }
    };

    let result = match rt.run_task(task, session_id.as_deref(), &CallbackSink::new(Box::new(emit)), &cancel) {
        Ok(res) => {
            let status = match res.status {
                crate::agent::RunStatus::Completed => "completed",
                crate::agent::RunStatus::MaxIterations => "max_iterations",
                crate::agent::RunStatus::Cancelled => "cancelled",
                crate::agent::RunStatus::Error(_) => "error",
            };
            to_c_string(json!({"status": status, "iterations": res.iterations, "final_text": res.final_text}).to_string())
        }
        Err(e) => to_c_string(json!({"status": "error", "error": e.to_string()}).to_string()),
    };
    drop(registration); // deregister the cancel flag
    result
}

/// Cancel ALL in-flight tasks (legacy behavior, kept for compatibility).
#[no_mangle]
pub extern "C" fn aacode_cancel() {
    // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0
    let map = cancel_map().lock().unwrap_or_else(|e| e.into_inner());
    for flag in map.values() {
        flag.store(true, Ordering::SeqCst);
    }
}

/// Cancel only the task registered with `client_task_id`. No-op if the task
/// already finished or the id is unknown.
#[no_mangle]
pub extern "C" fn aacode_cancel_task(task_id: *const c_char) {
    // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0
    let id = match unsafe { cstr(task_id) } {
        Some(s) => s,
        None => return,
    };
    let map = cancel_map().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(flag) = map.get(id) {
        flag.store(true, Ordering::SeqCst);
    }
}

/// Validate an API key. `config_json` should contain a `model` object (or the
/// flat model fields). Returns `{"valid":bool,"error":...}`.
#[no_mangle]
pub extern "C" fn aacode_validate_api_key(config_json: *const c_char) -> *mut c_char {
    // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0
    let input = match unsafe { cstr(config_json) } {
        Some(s) => s.to_string(),
        None => return to_c_string(json!({"valid": false, "error": "null config"}).to_string()),
    };
    let mut config: AgentConfig = serde_json::from_str(&input).unwrap_or_default();
    config.apply_env();
    let client = crate::llm::build_client(&config.model);
    let out = match client.validate() {
        Ok(()) => json!({"valid": true}),
        Err(e) => json!({"valid": false, "error": e.to_string()}),
    };
    to_c_string(out.to_string())
}

/// List sessions for a project path. Returns a JSON array.
#[no_mangle]
pub extern "C" fn aacode_list_sessions(project_path: *const c_char) -> *mut c_char {
    // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0
    let pp = match unsafe { cstr(project_path) } {
        Some(s) => s.to_string(),
        None => return to_c_string(json!({"success": false, "error": "null path"}).to_string()),
    };
    let sm = SessionManager::new(std::path::Path::new(&pp));
    let sessions: Vec<serde_json::Value> = sm
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
    to_c_string(json!({"success": true, "sessions": sessions}).to_string())
}

/// Get the messages of a session. Returns a JSON array of {role, content, ...}.
#[no_mangle]
pub extern "C" fn aacode_get_session_messages(
    project_path: *const c_char,
    session_id: *const c_char,
) -> *mut c_char {
    // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0
    let pp = unsafe { cstr(project_path) }.unwrap_or("").to_string();
    let sid = unsafe { cstr(session_id) }.unwrap_or("").to_string();
    let sm = SessionManager::new(std::path::Path::new(&pp));
    let msgs = sm.read_session_messages(&sid);
    let arr: Vec<serde_json::Value> = msgs
        .iter()
        .map(|m| {
            json!({
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
                "tool_calls": m.tool_calls,
                "tool_call_id": m.tool_call_id,
                "reasoning_content": m.reasoning_content,
            })
        })
        .collect();
    to_c_string(json!({"success": true, "messages": arr}).to_string())
}

/// Free a string previously returned by this library.
#[no_mangle]
pub extern "C" fn aacode_free_string(ptr: *mut c_char) {
    // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = CString::from_raw(ptr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern "C" fn noop_cb(_line: *const c_char) {}
    fn rt(json: &str) -> String {
        // NUL-terminate properly — passing str::as_ptr() directly is UB.
        let c = CString::new(json).unwrap();
        unsafe {
            CStr::from_ptr(aacode_run_task_with_cb(c.as_ptr(), noop_cb))
                .to_str()
                .unwrap()
                .to_string()
        }
    }

    #[test]
    fn run_task_bad_json() { let out = rt("not json"); assert!(out.contains("bad task_json")); }
    #[test]
    fn run_task_missing_task() { let out = rt(r#"{"project_path":"/tmp"}"#); assert!(out.contains("error"), "got: {out}"); }
    #[test]
    fn run_task_missing_key_reports_error() {
        let d = std::env::temp_dir().join(format!("ffi_{}", uuid::Uuid::new_v4().simple()));
        let input = json!({
            "task": "do x",
            "project_path": d.to_string_lossy(),
            "model": {"name": "deepseek-chat"} // no api key
        })
        .to_string();
        let out = rt(&input);
        assert!(out.contains("error"));
    }

    #[test]
    fn cancel_task_targets_only_its_flag() {
        let flag_a = register_cancel_flag("task_a");
        let flag_b = register_cancel_flag("task_b");
        let id = CString::new("task_a").unwrap();
        aacode_cancel_task(id.as_ptr());
        assert!(flag_a.load(Ordering::SeqCst), "task_a must be cancelled");
        assert!(!flag_b.load(Ordering::SeqCst), "task_b must NOT be cancelled");
        deregister_cancel_flag("task_a");
        deregister_cancel_flag("task_b");
    }

    #[test]
    fn cancel_all_sets_every_flag() {
        let flag_a = register_cancel_flag("all_a");
        let flag_b = register_cancel_flag("all_b");
        aacode_cancel();
        assert!(flag_a.load(Ordering::SeqCst));
        assert!(flag_b.load(Ordering::SeqCst));
        deregister_cancel_flag("all_a");
        deregister_cancel_flag("all_b");
    }

    #[test]
    fn cancel_unknown_task_is_noop() {
        let id = CString::new("nonexistent").unwrap();
        aacode_cancel_task(id.as_ptr()); // must not panic
        aacode_cancel_task(std::ptr::null());
    }

    #[test]
    fn new_task_does_not_clear_other_cancels() {
        // Regression: starting task B must not reset a pending cancel on A.
        let flag_a = register_cancel_flag("pending_a");
        let id = CString::new("pending_a").unwrap();
        aacode_cancel_task(id.as_ptr());
        assert!(flag_a.load(Ordering::SeqCst));
        // Register another task — A's flag must stay cancelled.
        let _flag_b = register_cancel_flag("pending_b");
        assert!(flag_a.load(Ordering::SeqCst), "cancel on A must survive B's start");
        deregister_cancel_flag("pending_a");
        deregister_cancel_flag("pending_b");
    }

    #[test]
    fn registration_guard_deregisters_on_drop() {
        {
            let _reg = CancelRegistration::new("guard_x".to_string());
            assert!(cancel_map().lock().unwrap().contains_key("guard_x"));
        }
        assert!(!cancel_map().lock().unwrap().contains_key("guard_x"));
    }

    #[test]
    fn session_guard_rejects_duplicate() {
        let g1 = SessionGuard::try_acquire("p::s1".to_string());
        assert!(g1.is_some());
        let g2 = SessionGuard::try_acquire("p::s1".to_string());
        assert!(g2.is_none(), "duplicate session must be rejected");
        drop(g1);
        let g3 = SessionGuard::try_acquire("p::s1".to_string());
        assert!(g3.is_some(), "released session must be reacquirable");
    }

    #[test]
    fn concurrent_tasks_same_session_rejected_via_ffi() {
        // Two tasks with the same session_id: one must be rejected while the
        // first is still registered. We simulate the first by holding a guard.
        let _hold = SessionGuard::try_acquire("/tmp/proj::sess_dup".to_string()).unwrap();
        let input = json!({
            "task": "do x",
            "project_path": "/tmp/proj",
            "session_id": "sess_dup",
            "model": {"name": "m", "api_key": "k"}
        })
        .to_string();
        let out = rt(&input);
        assert!(out.contains("already has a task running"), "got: {out}");
    }

    #[test]
    fn free_null_is_safe() {
        aacode_free_string(std::ptr::null_mut());
    }

    // ── resolve_shell_root (skills user_dir root widening) ─────────

    fn mkdirs(p: &std::path::Path) -> String {
        std::fs::create_dir_all(p).unwrap();
        p.to_string_lossy().to_string()
    }

    fn tmp_root(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("ffi_root_{tag}_{}", uuid::Uuid::new_v4().simple()))
    }

    #[test]
    fn no_skills_config_keeps_project_jail() {
        let base = tmp_root("nocfg");
        let sb = mkdirs(&base.join("sb"));
        let pp = mkdirs(&base.join("sb").join("proj"));
        let (root, rel) = resolve_shell_root(&sb, &pp, false);
        assert_eq!(root, pp);
        assert!(rel.is_none());
    }

    #[test]
    fn project_inside_sandbox_widens_and_positions_cwd() {
        let base = tmp_root("inside");
        let sb = mkdirs(&base.join("sb"));
        let pp = mkdirs(&base.join("sb").join("a").join("b"));
        let (root, rel) = resolve_shell_root(&sb, &pp, true);
        assert_eq!(root, sb);
        let rel = rel.expect("must reposition cwd");
        assert_eq!(
            std::path::PathBuf::from(rel),
            std::path::PathBuf::from("a").join("b")
        );
    }

    #[test]
    fn project_equal_to_sandbox_not_widened() {
        let base = tmp_root("equal");
        let sb = mkdirs(&base.join("sb"));
        let (root, rel) = resolve_shell_root(&sb, &sb, true);
        assert_eq!(root, sb);
        assert!(rel.is_none());
    }

    #[test]
    fn sibling_prefix_dir_is_not_inside_sandbox() {
        // Regression: /…/sb vs /…/sb2/proj — a string prefix check would
        // wrongly widen the root and then fail the cd with a hard error.
        let base = tmp_root("sibling");
        let sb = mkdirs(&base.join("sb"));
        let pp = mkdirs(&base.join("sb2").join("proj"));
        let (root, rel) = resolve_shell_root(&sb, &pp, true);
        assert_eq!(root, pp, "sibling project must keep its own jail");
        assert!(rel.is_none());
    }

    #[test]
    fn project_outside_sandbox_keeps_own_jail() {
        let base = tmp_root("outside");
        let sb = mkdirs(&base.join("sb"));
        let pp = mkdirs(&base.join("elsewhere").join("proj"));
        let (root, rel) = resolve_shell_root(&sb, &pp, true);
        assert_eq!(root, pp);
        assert!(rel.is_none());
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_project_path_still_recognized() {
        // Android: /data/user/0/<pkg> is a symlink to /data/data/<pkg>;
        // macOS/iOS: /var → /private/var. The host may pass the symlink form
        // for the project while the sandbox root is canonical (or vice
        // versa) — canonicalization must reconcile them.
        let base = tmp_root("symlink");
        let real_sb = mkdirs(&base.join("real_sb"));
        mkdirs(&base.join("real_sb").join("proj"));
        let link_sb = base.join("link_sb");
        std::os::unix::fs::symlink(&real_sb, &link_sb).unwrap();
        let pp_via_link = link_sb.join("proj").to_string_lossy().to_string();
        let (root, rel) = resolve_shell_root(&real_sb, &pp_via_link, true);
        assert_eq!(root, real_sb, "symlinked project must widen to sandbox");
        assert_eq!(rel.as_deref(), Some("proj"));
    }

    #[test]
    fn list_sessions_ffi() {
        let d = std::env::temp_dir().join(format!("ffi_ls_{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&d).unwrap();
        {
            let mut sm = SessionManager::new(&d);
            sm.create_session("task", None).unwrap();
        }
        let c = CString::new(d.to_string_lossy().to_string()).unwrap();
        let ptr = aacode_list_sessions(c.as_ptr());
        let s = unsafe { CStr::from_ptr(ptr).to_str().unwrap().to_string() };
        aacode_free_string(ptr);
        assert!(s.contains("\"success\":true"));
    }

    #[test]
    fn callback_roundtrip() {
        use std::sync::atomic::AtomicUsize;
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        extern "C" fn cb(_line: *const c_char) {
            COUNT.fetch_add(1, Ordering::SeqCst);
        }
        aacode_register_stream_callback(Some(cb));
        let sink = FfiSink;
        sink.done("s1");
        assert!(COUNT.load(Ordering::SeqCst) >= 1);
        aacode_register_stream_callback(None);
    }
}
