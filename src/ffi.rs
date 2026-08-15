// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! C ABI for embedding aacode-rs into mobile hosts (Android via `jni_glue.c`,
//! iOS via the Swift bridge) and other hosts.
//!
//! Contract (handle-based, userdata-carrying, single event stream):
//!   * `aacode_task_start(task_json, cb, userdata)` — non-blocking. Returns an
//!     opaque handle. Each JSONL event line is delivered via `cb(line, userdata)`
//!     from the tokio worker threads. Early failures (bad JSON, missing task,
//!     session already running, init/cd failure) emit an `error` event and a
//!     handle whose `wait` immediately returns the error.
//!   * `aacode_task_wait(handle)` — blocks until the task finishes; returns a
//!     JSON result string (caller frees with `aacode_free_string`).
//!   * `aacode_task_cancel(handle)` — non-blocking; the running task observes
//!     it asynchronously.
//!   * `aacode_task_free(handle)` — free a finished handle.
//!   * `aacode_validate_api_key(config_json)` — returns `{"valid":bool,...}`.
//!   * `aacode_list_sessions(project_path)` / `aacode_get_session_messages(...)`.
//!   * `aacode_free_string(ptr)` — free any returned string.
//!
//! The terminal event is an enriched `done` (`{"type":"done","session_id":...,
//! "status":...,"iterations":...,"final_text":...}`); `wait` returns the same
//! outcome for hosts that prefer a return value. Events are authoritative.

// C-ABI surface: every extern fn validates pointers (null-checked, then
// CStr::from_ptr) per the documented host contract; `unsafe fn` would not
// change the C caller's obligations.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::config::AgentConfig;
use crate::runtime::{AgentRuntime, TOKIO_RT};
use crate::session::{valid_session_id, SessionManager, SessionMessage, SCHEMA_VERSION};
use crate::stream::CallbackSink;
use serde_json::json;
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::sync::mpsc;

/// The C stream callback type: receives one NUL-terminated UTF-8 line plus the
/// opaque `userdata` context supplied at `aacode_task_start`. The userdata lets
/// each host bind per-task state without thread-local or global hacks.
pub type StreamCallback = extern "C" fn(*const c_char, *mut c_void);

/// Opaque task handle returned by `aacode_task_start`.
pub struct AacodeTask {
    cancel: Arc<AtomicBool>,
    result_rx: mpsc::Receiver<String>,
}

/// Wrapper making the raw `userdata` pointer safe to move into the tokio task
/// (the host guarantees the pointed-to context outlives the task).
struct SendUserData(*mut c_void);
// SAFETY: the host contract requires `userdata` to remain valid until the task
// completes and `aacode_task_free` is called. It is only passed back verbatim
// to the host callback.
unsafe impl Send for SendUserData {}
unsafe impl Sync for SendUserData {}

impl SendUserData {
    /// Raw pointer access via a method so closures capture the whole wrapper
    /// (and its `Send`/`Sync` impls) rather than the bare `*mut c_void` field.
    #[inline]
    fn ptr(&self) -> *mut c_void {
        self.0
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

/// Decide the shell sandbox root for the SDK (mobile/embedded host) path.
/// Returns `(shell_root, cd_rel)`: the VFS root to init fastshell with, and
/// an optional sandbox-relative path to `cd` into right after init.
///
/// Default: the shell is confined to the project. When the host configured a
/// user skills directory (`config.skills.user_dir`) the agent must be able to
/// manage skills via run_shell — those may live OUTSIDE the project — so the
/// shell root is widened to the minimal common ancestor of project and
/// user_dir. Hosts that don't pass skills.user_dir keep the old
/// strictly-project-confined behavior.
///
/// Widening uses a three-step cascade (first match wins):
/// 1. Project is a proper subdirectory of sandbox → widen to sandbox.
/// 2. User dir is already inside project → no widening needed.
/// 3. Walk project ancestors to find the closest one containing user dir.
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
    user_dir: Option<&str>,
) -> (String, Option<String>) {
    let Some(user_dir) = user_dir else {
        return (project.to_string(), None);
    };
    let canon = |s: &str| {
        std::fs::canonicalize(s).unwrap_or_else(|_| std::path::PathBuf::from(s))
    };
    let sandbox_c = canon(sandbox);
    let project_c = canon(project);
    let user_dir_c = canon(user_dir);

    // Step 1: project is a proper subdirectory of sandbox → widen to sandbox.
    match project_c.strip_prefix(&sandbox_c) {
        Ok(rel) if !rel.as_os_str().is_empty() => {
            return (sandbox.to_string(), Some(rel.to_string_lossy().to_string()));
        }
        _ => {}
    }

    // Step 2: user_dir is already inside project → no widening needed.
    if user_dir_c.strip_prefix(&project_c).is_ok() {
        return (project.to_string(), None);
    }

    // Step 3: walk project ancestors to find the closest common ancestor
    //         that also contains user_dir. Stop before the filesystem root.
    //
    // Example (iOS): project = /a/b/projects/Test3, user_dir = /a/b/skills
    //   ancestors of project:  /a/b/projects  → user_dir not inside ✗
    //                          /a/b            → user_dir inside      ✓  ← widen here
    //   Result: shell_root = /a/b, cd_rel = projects/Test3
    for ancestor in project_c.ancestors().skip(1) {
        if ancestor.parent().is_none() {
            break;
        }
        if user_dir_c.strip_prefix(ancestor).is_ok() {
            if let Ok(rel) = project_c.strip_prefix(ancestor) {
                if !rel.as_os_str().is_empty() {
                    return (ancestor.to_string_lossy().to_string(), Some(rel.to_string_lossy().to_string()));
                }
            }
        }
    }

    (project.to_string(), None)
}

/// Build the agent runtime for the parsed config + project path. Returns an
/// error string suitable for an `error` event / terminal JSON.
fn build_runtime(
    config: &AgentConfig,
    project_path: &str,
) -> Result<AgentRuntime, String> {
    if let Some(sdk) = fastshell::sdk::try_get_sdk_instance() {
        let sandbox = {
            let guard = sdk.lock().unwrap_or_else(|e| e.into_inner());
            guard.vfs_root()
        };
        let pp = if project_path == "." { sandbox.clone() } else { project_path.to_string() };
        // Shell sandbox root + optional cwd reposition (see resolve_shell_root).
        let (shell_root, cd_rel) =
            resolve_shell_root(&sandbox, &pp, config.skills.user_dir.as_deref());
        // Compute a VFS-internal skills path (relative to shell root) so
        // skill prompts can use short, platform-independent paths like
        // `/skills` instead of long absolute physical paths.
        let mut cfg = config.clone();
        if let Some(ref user_dir) = cfg.skills.user_dir {
            if let Ok(rel) = std::path::Path::new(user_dir).strip_prefix(&shell_root) {
                if !rel.as_os_str().is_empty() {
                    cfg.skills.vfs_skills_dir = Some(format!("/{}", rel.to_string_lossy()));
                }
            }
        }
        let fs_arc: Arc<Mutex<fastshell::Fastshell>> =
            Arc::new(Mutex::new(fastshell::Fastshell::new()));
        {
            let mut ours = fs_arc.lock().unwrap();
            let mut fscfg = fastshell::Config::default();
            fscfg.sandbox_path = shell_root.clone();
            fscfg.python_enabled = true;
            fscfg.allow_subprocess = false; // mobile sandbox: no subprocess spawn
            fscfg.network_ask_permission = false;
            if let Err(e) = ours.init(fscfg) {
                return Err(e.to_string());
            }
            if let Some(rel) = cd_rel {
                // Position the persistent shell session inside the project so
                // relative paths behave exactly as before the root widening.
                let r = ours.execute(&format!("cd '{}'", rel.replace('\'', "'\\''")));
                if r.exit_code != 0 {
                    return Err(format!("cd to project failed: {}", r.stderr));
                }
            }
        }
        Ok(AgentRuntime::with_fastshell(cfg, std::path::PathBuf::from(pp), fs_arc))
    } else {
        AgentRuntime::init(config.clone(), std::path::PathBuf::from(project_path))
            .map_err(|e| e.to_string())
    }
}

/// Start a task described by `task_json`. Non-blocking; events stream through
/// `cb(line, userdata)`. Returns an opaque handle (NULL on catastrophic OOM).
#[no_mangle]
pub extern "C" fn aacode_task_start(
    task_json: *const c_char,
    cb: Option<StreamCallback>,
    userdata: *mut c_void,
) -> *mut AacodeTask {
    // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0
    let (tx, rx) = mpsc::channel::<String>();
    let cancel = Arc::new(AtomicBool::new(false));

    let ud = SendUserData(userdata);
    let emit = move |line: &str| {
        if let Some(cb) = cb {
            if let Ok(cs) = CString::new(line) {
                cb(cs.as_ptr(), ud.ptr());
            }
        }
    };

    let input = match unsafe { cstr(task_json) } {
        Some(s) => s.to_string(),
        None => {
            let msg = "null task_json";
            emit(&json!({"type": "error", "message": msg}).to_string());
            let _ = tx.send(json!({"status": "error", "error": msg}).to_string());
            return Box::into_raw(Box::new(AacodeTask { cancel, result_rx: rx }));
        }
    };

    let v: serde_json::Value = match serde_json::from_str(&input) {
        Ok(v) => v,
        Err(e) => {
            let msg = format!("bad task_json: {e}");
            emit(&json!({"type": "error", "message": msg}).to_string());
            let _ = tx.send(json!({"status": "error", "error": msg}).to_string());
            return Box::into_raw(Box::new(AacodeTask { cancel, result_rx: rx }));
        }
    };
    let task = v.get("task").and_then(|x| x.as_str()).unwrap_or("").to_string();
    if task.is_empty() {
        let msg = "missing task";
        emit(&json!({"type": "error", "message": msg}).to_string());
        let _ = tx.send(json!({"status": "error", "error": msg}).to_string());
        return Box::into_raw(Box::new(AacodeTask { cancel, result_rx: rx }));
    }
    let project_path = v.get("project_path").and_then(|x| x.as_str()).unwrap_or(".").to_string();
    let session_id = v.get("session_id").and_then(|x| x.as_str()).map(|s| s.to_string());

    // Reject a second concurrent task on the same session (would corrupt the
    // session file). New sessions (no session_id) are always allowed.
    let session_guard = match &session_id {
        Some(sid) => {
            let key = format!("{project_path}::{sid}");
            match SessionGuard::try_acquire(key) {
                Some(g) => Some(g),
                None => {
                    let msg = format!("session '{sid}' already has a task running");
                    emit(&json!({"type": "error", "message": msg}).to_string());
                    let _ = tx.send(json!({"status": "error", "error": msg}).to_string());
                    return Box::into_raw(Box::new(AacodeTask { cancel, result_rx: rx }));
                }
            }
        }
        None => None,
    };

    let mut config: AgentConfig = serde_json::from_value(v).unwrap_or_default();
    config.apply_env();

    let rt = match build_runtime(&config, &project_path) {
        Ok(rt) => rt,
        Err(e) => {
            emit(&json!({"type": "error", "message": e}).to_string());
            let _ = tx.send(json!({"status": "error", "error": e}).to_string());
            return Box::into_raw(Box::new(AacodeTask { cancel, result_rx: rx }));
        }
    };

    let cancel2 = cancel.clone();
    let _join = TOKIO_RT.spawn(async move {
        // Hold the session guard for the duration of the task.
        let _guard = session_guard;
        let result = rt
            .run_task(&task, session_id.as_deref(), &CallbackSink::new(Box::new(emit)), &cancel2)
            .await;
        let json = match result {
            Ok(res) => res.to_result_json().to_string(),
            Err(e) => json!({"status": "error", "error": e.to_string()}).to_string(),
        };
        let _ = tx.send(json);
    });

    Box::into_raw(Box::new(AacodeTask { cancel, result_rx: rx }))
}

/// Block until the task finishes; return the terminal JSON result string
/// (caller frees with `aacode_free_string`).
#[no_mangle]
pub extern "C" fn aacode_task_wait(handle: *mut AacodeTask) -> *mut c_char {
    // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0
    if handle.is_null() {
        return to_c_string(json!({"status": "error", "error": "null handle"}).to_string());
    }
    let h = unsafe { &*handle };
    let result = h.result_rx.recv().unwrap_or_else(|_| {
        json!({"status": "error", "error": "task was dropped before completion"}).to_string()
    });
    to_c_string(result)
}

/// Signal cancellation. Non-blocking; the running task observes it async.
#[no_mangle]
pub extern "C" fn aacode_task_cancel(handle: *mut AacodeTask) {
    // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0
    if handle.is_null() {
        return;
    }
    let h = unsafe { &*handle };
    h.cancel.store(true, Ordering::SeqCst);
}

/// Free a finished handle.
#[no_mangle]
pub extern "C" fn aacode_task_free(handle: *mut AacodeTask) {
    // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0
    if handle.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(handle));
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
    let handle = TOKIO_RT.handle().clone();
    let out = std::thread::scope(|s| {
        s.spawn(move || match handle.block_on(async { client.validate().await }) {
            Ok(()) => json!({"valid": true}),
            Err(e) => json!({"valid": false, "error": e.to_string()}),
        })
        .join()
        .unwrap()
    });
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

// ── Session store FFI (see SESSION_FFI.md) ─────────────────────────

/// Serializes all session-store write FFI calls so concurrent callers can't
/// interleave read-modify-write on the shared index/session files (in-process
/// serialization; the agent's own writes are not under this lock).
static SESSION_WRITE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Serialize one persisted message into the SESSION_FFI wire shape.
fn message_to_json(m: &SessionMessage) -> serde_json::Value {
    json!({
        "role": m.role,
        "content": m.content,
        "timestamp": m.timestamp,
        "tokens": m.tokens,
        "tool_calls": m.tool_calls,
        "tool_call_id": m.tool_call_id,
        "reasoning_content": m.reasoning_content,
    })
}

/// Build the `{"success": false, "error": ...}` failure envelope.
fn session_err(msg: &str) -> *mut c_char {
    to_c_string(json!({"success": false, "error": msg}).to_string())
}

/// Parse `project` + `session_id` C strings into owned Strings, validating the
/// session id so it cannot escape the sessions directory.
fn session_id_args(
    project_path: *const c_char,
    session_id: *const c_char,
) -> Result<(String, String), String> {
    let pp = unsafe { cstr(project_path) }
        .ok_or_else(|| "null project_path".to_string())?
        .to_string();
    let sid = unsafe { cstr(session_id) }
        .ok_or_else(|| "null session_id".to_string())?
        .to_string();
    if !valid_session_id(&sid) {
        return Err("invalid session_id".to_string());
    }
    Ok((pp, sid))
}

/// Current on-disk session schema version (see SESSION_FFI.md §6).
#[no_mangle]
pub extern "C" fn aacode_session_version() -> u32 {
    SCHEMA_VERSION
}

/// List sessions for a project (SESSION_FFI contract). Returns a JSON array.
#[no_mangle]
pub extern "C" fn aacode_session_list(project_path: *const c_char) -> *mut c_char {
    let pp = match unsafe { cstr(project_path) } {
        Some(s) => s.to_string(),
        None => return session_err("null project_path"),
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
                "total_tokens": s.total_tokens,
                "status": s.status,
            })
        })
        .collect();
    to_c_string(json!({"success": true, "sessions": sessions}).to_string())
}

/// Paginated session messages (SESSION_FFI contract).
#[no_mangle]
pub extern "C" fn aacode_session_messages(
    project_path: *const c_char,
    session_id: *const c_char,
    offset: u32,
    limit: u32,
) -> *mut c_char {
    let (pp, sid) = match session_id_args(project_path, session_id) {
        Ok(v) => v,
        Err(e) => return session_err(&e),
    };
    let sm = SessionManager::new(std::path::Path::new(&pp));
    let all = sm.read_session_messages(&sid);
    let total = all.len();
    let offset = offset as usize;
    let limit = limit as usize;
    let end = total.saturating_sub(offset);
    let start = end.saturating_sub(limit);
    let slice: Vec<serde_json::Value> = if start < end {
        all[start..end].iter().map(message_to_json).collect()
    } else {
        Vec::new()
    };
    to_c_string(
        json!({
            "success": true,
            "total": total,
            "offset": offset,
            "limit": limit,
            "messages": slice,
        })
        .to_string(),
    )
}

/// Idempotent create-or-touch a session (SESSION_FFI contract).
#[no_mangle]
pub extern "C" fn aacode_session_ensure(
    project_path: *const c_char,
    session_id: *const c_char,
    title: *const c_char,
) -> *mut c_char {
    let (pp, sid) = match session_id_args(project_path, session_id) {
        Ok(v) => v,
        Err(e) => return session_err(&e),
    };
    let title = unsafe { cstr(title) }.unwrap_or("").to_string();
    let _guard = SESSION_WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut sm = SessionManager::new(std::path::Path::new(&pp));
    match sm.ensure_session(&sid, &title) {
        Ok(()) => to_c_string(json!({"success": true}).to_string()),
        Err(e) => session_err(&e.to_string()),
    }
}

/// Set an explicit title on a session (SESSION_FFI contract).
#[no_mangle]
pub extern "C" fn aacode_session_rename(
    project_path: *const c_char,
    session_id: *const c_char,
    title: *const c_char,
) -> *mut c_char {
    let (pp, sid) = match session_id_args(project_path, session_id) {
        Ok(v) => v,
        Err(e) => return session_err(&e),
    };
    let title = unsafe { cstr(title) }.unwrap_or("").to_string();
    let _guard = SESSION_WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut sm = SessionManager::new(std::path::Path::new(&pp));
    match sm.rename_session(&sid, &title) {
        Ok(()) => to_c_string(json!({"success": true}).to_string()),
        Err(e) => session_err(&e.to_string()),
    }
}

/// Bump `last_activity` of a session (SESSION_FFI contract).
#[no_mangle]
pub extern "C" fn aacode_session_touch(
    project_path: *const c_char,
    session_id: *const c_char,
) -> *mut c_char {
    let (pp, sid) = match session_id_args(project_path, session_id) {
        Ok(v) => v,
        Err(e) => return session_err(&e),
    };
    let _guard = SESSION_WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut sm = SessionManager::new(std::path::Path::new(&pp));
    match sm.touch_session(&sid) {
        Ok(()) => to_c_string(json!({"success": true}).to_string()),
        Err(e) => session_err(&e.to_string()),
    }
}

/// Delete a session (idempotent; SESSION_FFI contract).
#[no_mangle]
pub extern "C" fn aacode_session_delete(
    project_path: *const c_char,
    session_id: *const c_char,
) -> *mut c_char {
    let (pp, sid) = match session_id_args(project_path, session_id) {
        Ok(v) => v,
        Err(e) => return session_err(&e),
    };
    let _guard = SESSION_WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut sm = SessionManager::new(std::path::Path::new(&pp));
    match sm.delete_session(&sid) {
        Ok(_) => to_c_string(json!({"success": true}).to_string()),
        Err(e) => session_err(&e.to_string()),
    }
}

/// Append messages to a session (SESSION_FFI contract).
#[no_mangle]
pub extern "C" fn aacode_session_append(
    project_path: *const c_char,
    session_id: *const c_char,
    msgs_json: *const c_char,
) -> *mut c_char {
    let (pp, sid) = match session_id_args(project_path, session_id) {
        Ok(v) => v,
        Err(e) => return session_err(&e),
    };
    let msgs_str = match unsafe { cstr(msgs_json) } {
        Some(s) => s,
        None => return session_err("null msgs_json"),
    };
    let msgs: Vec<SessionMessage> = match serde_json::from_str(msgs_str) {
        Ok(v) => v,
        Err(e) => return session_err(&format!("bad msgs_json: {e}")),
    };
    let _guard = SESSION_WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut sm = SessionManager::new(std::path::Path::new(&pp));
    match sm.append_session_messages(&sid, msgs) {
        Ok(()) => to_c_string(json!({"success": true}).to_string()),
        Err(e) => session_err(&e.to_string()),
    }
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

    extern "C" fn capture_cb(line: *const c_char, userdata: *mut c_void) {
        if line.is_null() {
            return;
        }
        let l = unsafe { CStr::from_ptr(line).to_str().unwrap_or("").to_string() };
        let sink = userdata as *mut Mutex<Vec<String>>;
        if !sink.is_null() {
            unsafe {
                if let Ok(mut v) = (*sink).lock() {
                    v.push(l);
                }
            }
        }
    }

    fn run_task(json: &str) -> (String, Vec<String>) {
        let c = CString::new(json).unwrap();
        let lines: Mutex<Vec<String>> = Mutex::new(Vec::new());
        let ud = &lines as *const _ as *mut c_void;
        let h = aacode_task_start(c.as_ptr(), Some(capture_cb), ud);
        let result = unsafe {
            let p = aacode_task_wait(h);
            let s = CStr::from_ptr(p).to_str().unwrap().to_string();
            aacode_free_string(p);
            aacode_task_free(h);
            s
        };
        let captured = lines.lock().unwrap().clone();
        (result, captured)
    }

    #[test]
    fn run_task_bad_json() {
        let (out, events) = run_task("not json");
        assert!(out.contains("bad task_json"), "got: {out}");
        assert!(
            events.iter().any(|l| l.contains(r#""type":"error""#) && l.contains("bad task_json")),
            "error event must be emitted: {events:?}"
        );
    }

    #[test]
    fn run_task_missing_task() {
        let (out, events) = run_task(r#"{"project_path":"/tmp"}"#);
        assert!(out.contains("error"), "got: {out}");
        assert!(events.iter().any(|l| l.contains(r#""type":"error""#)));
    }

    #[test]
    fn callback_receives_userdata_on_early_error() {
        use std::sync::atomic::AtomicUsize;
        static RECEIVED_UD: AtomicUsize = AtomicUsize::new(0);
        extern "C" fn ud_cb(_line: *const c_char, ud: *mut c_void) {
            RECEIVED_UD.store(ud as usize, Ordering::SeqCst);
        }
        let c = CString::new(r#"{"project_path":"/tmp"}"#).unwrap();
        let marker = 0xABCD1234usize as *mut c_void;
        let h = aacode_task_start(c.as_ptr(), Some(ud_cb), marker);
        let p = aacode_task_wait(h);
        let s = unsafe { CStr::from_ptr(p).to_str().unwrap().to_string() };
        aacode_free_string(p);
        aacode_task_free(h);
        assert!(s.contains("missing task"), "got: {s}");
        // The host's opaque userdata must be passed back verbatim to the callback.
        assert_eq!(RECEIVED_UD.load(Ordering::SeqCst), 0xABCD1234usize);
    }

    #[test]
    fn start_with_null_callback_still_returns_terminal_result() {
        let c = CString::new("not json").unwrap();
        let h = aacode_task_start(c.as_ptr(), None, std::ptr::null_mut());
        let p = aacode_task_wait(h);
        let s = unsafe { CStr::from_ptr(p).to_str().unwrap().to_string() };
        aacode_free_string(p);
        aacode_task_free(h);
        assert!(s.contains("bad task_json"), "got: {s}");
    }

    #[test]
    fn null_task_json() {
        let lines: Mutex<Vec<String>> = Mutex::new(Vec::new());
        let ud = &lines as *const _ as *mut c_void;
        let h = aacode_task_start(std::ptr::null(), Some(capture_cb), ud);
        let result = unsafe {
            let p = aacode_task_wait(h);
            let s = CStr::from_ptr(p).to_str().unwrap().to_string();
            aacode_free_string(p);
            aacode_task_free(h);
            s
        };
        assert!(result.contains("null task_json"));
    }

    #[test]
    fn wait_null_handle_is_safe() {
        let p = aacode_task_wait(std::ptr::null_mut());
        let s = unsafe { CStr::from_ptr(p).to_str().unwrap().to_string() };
        aacode_free_string(p);
        assert!(s.contains("error"));
    }

    #[test]
    fn cancel_null_handle_is_safe() {
        aacode_task_cancel(std::ptr::null_mut());
    }

    #[test]
    fn free_null_is_safe() {
        aacode_task_free(std::ptr::null_mut());
        aacode_free_string(std::ptr::null_mut());
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
    fn concurrent_tasks_same_session_rejected() {
        let _hold = SessionGuard::try_acquire("/tmp/proj::sess_dup".to_string()).unwrap();
        let (out, _events) = run_task(
            r#"{"task":"do x","project_path":"/tmp/proj","session_id":"sess_dup","model":{"name":"m","api_key":"k"}}"#,
        );
        assert!(out.contains("already has a task running"), "got: {out}");
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
        let (root, rel) = resolve_shell_root(&sb, &pp, None);
        assert_eq!(root, pp);
        assert!(rel.is_none());
    }

    #[test]
    fn project_inside_sandbox_widens_and_positions_cwd() {
        let base = tmp_root("inside");
        let sb = mkdirs(&base.join("sb"));
        let sk = mkdirs(&base.join("sb").join("skills"));
        let pp = mkdirs(&base.join("sb").join("a").join("b"));
        let (root, rel) = resolve_shell_root(&sb, &pp, Some(&sk));
        assert_eq!(root, sb);
        let rel = rel.expect("must reposition cwd");
        assert_eq!(
            std::path::PathBuf::from(rel),
            std::path::PathBuf::from("a").join("b")
        );
    }

    #[test]
    fn project_equal_to_sandbox_user_dir_inside_no_widening() {
        let base = tmp_root("equal_in");
        let sb = mkdirs(&base.join("sb"));
        let sk = mkdirs(&base.join("sb").join("skills"));
        let (root, rel) = resolve_shell_root(&sb, &sb, Some(&sk));
        assert_eq!(root, sb);
        assert!(rel.is_none());
    }

    #[test]
    fn project_equal_to_sandbox_user_dir_outside_widens() {
        let base = tmp_root("equal_out");
        let sb = mkdirs(&base.join("sb"));
        let sk = mkdirs(&base.join("skills"));
        let (root, rel) = resolve_shell_root(&sb, &sb, Some(&sk));
        let base_c = std::fs::canonicalize(&base).unwrap_or_else(|_| base.clone());
        assert_eq!(root, base_c.to_string_lossy());
        assert_eq!(rel.as_deref(), Some("sb"));
    }

    #[test]
    fn sibling_prefix_dir_is_not_inside_sandbox() {
        let base = tmp_root("sibling");
        let sb = mkdirs(&base.join("sb"));
        let pp = mkdirs(&base.join("sb2").join("proj"));
        let sk = mkdirs(&base.join("sb2").join("proj").join("skills"));
        let (root, rel) = resolve_shell_root(&sb, &pp, Some(&sk));
        assert_eq!(root, pp, "sibling project must keep its own jail");
        assert!(rel.is_none());
    }

    #[test]
    fn project_outside_sandbox_keeps_own_jail() {
        let base = tmp_root("outside");
        let sb = mkdirs(&base.join("sb"));
        let pp = mkdirs(&base.join("elsewhere").join("proj"));
        let sk = mkdirs(&base.join("elsewhere").join("proj").join("skills"));
        let (root, rel) = resolve_shell_root(&sb, &pp, Some(&sk));
        assert_eq!(root, pp);
        assert!(rel.is_none());
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_project_path_still_recognized() {
        let base = tmp_root("symlink");
        let real_sb = mkdirs(&base.join("real_sb"));
        mkdirs(&base.join("real_sb").join("proj"));
        let sk = mkdirs(&base.join("real_sb").join("skills"));
        let link_sb = base.join("link_sb");
        std::os::unix::fs::symlink(&real_sb, &link_sb).unwrap();
        let pp_via_link = link_sb.join("proj").to_string_lossy().to_string();
        let (root, rel) = resolve_shell_root(&real_sb, &pp_via_link, Some(&sk));
        assert_eq!(root, real_sb, "symlinked project must widen to sandbox");
        assert_eq!(rel.as_deref(), Some("proj"));
    }

    // ── SESSION_FFI store functions ──────────────────────────────────

    fn call_str(f: impl FnOnce() -> *mut c_char) -> String {
        let p = f();
        let s = unsafe { CStr::from_ptr(p).to_str().unwrap().to_string() };
        aacode_free_string(p);
        s
    }

    #[test]
    fn session_version_is_one() {
        assert_eq!(aacode_session_version(), SCHEMA_VERSION);
        assert_eq!(aacode_session_version(), 1);
    }

    #[test]
    fn session_lifecycle_roundtrip() {
        let proj = mkdirs(&tmp_root("sess_life"));
        let pp = CString::new(proj).unwrap();
        let sid = CString::new("sess_a").unwrap();

        // ensure (create)
        let title = CString::new("My Session").unwrap();
        let r = call_str(|| aacode_session_ensure(pp.as_ptr(), sid.as_ptr(), title.as_ptr()));
        assert!(r.contains("\"success\":true"), "ensure: {r}");

        // append two messages
        let msgs = r#"[{"role":"user","content":"hello","timestamp":"1700000001"},{"role":"assistant","content":"hi","timestamp":"1700000002"}]"#;
        let mj = CString::new(msgs).unwrap();
        let r = call_str(|| aacode_session_append(pp.as_ptr(), sid.as_ptr(), mj.as_ptr()));
        assert!(r.contains("\"success\":true"), "append: {r}");

        // rename
        let new_title = CString::new("Renamed").unwrap();
        let r = call_str(|| aacode_session_rename(pp.as_ptr(), sid.as_ptr(), new_title.as_ptr()));
        assert!(r.contains("\"success\":true"), "rename: {r}");

        // list
        let r = call_str(|| aacode_session_list(pp.as_ptr()));
        assert!(r.contains("\"success\":true"), "list: {r}");
        assert!(r.contains("sess_a"));
        assert!(r.contains("Renamed"));

        // paginated messages
        let r = call_str(|| aacode_session_messages(pp.as_ptr(), sid.as_ptr(), 0, 50));
        assert!(r.contains("\"success\":true"), "messages: {r}");
        assert!(r.contains("\"total\":2"));
        assert!(r.contains("hello"));

        // touch
        let r = call_str(|| aacode_session_touch(pp.as_ptr(), sid.as_ptr()));
        assert!(r.contains("\"success\":true"), "touch: {r}");

        // delete (idempotent)
        let r = call_str(|| aacode_session_delete(pp.as_ptr(), sid.as_ptr()));
        assert!(r.contains("\"success\":true"), "delete: {r}");
        let r = call_str(|| aacode_session_delete(pp.as_ptr(), sid.as_ptr()));
        assert!(r.contains("\"success\":true"), "delete again (idempotent): {r}");
    }

    #[test]
    fn session_append_bad_json_errors() {
        let proj = mkdirs(&tmp_root("sess_bad"));
        let pp = CString::new(proj).unwrap();
        let sid = CString::new("s1").unwrap();
        let bad = CString::new("not json").unwrap();
        let r = call_str(|| aacode_session_append(pp.as_ptr(), sid.as_ptr(), bad.as_ptr()));
        assert!(r.contains("\"success\":false"), "bad msgs_json must error: {r}");
        assert!(r.contains("bad msgs_json"));
    }

    #[test]
    fn session_messages_pagination() {
        let proj = mkdirs(&tmp_root("sess_page"));
        let pp = CString::new(proj.clone()).unwrap();
        let sid = CString::new("s1").unwrap();
        {
            let mut sm = SessionManager::new(std::path::Path::new(&proj));
            sm.ensure_session("s1", "").unwrap();
            for i in 0..5 {
                sm.append_session_messages(
                    "s1",
                    vec![SessionMessage {
                        role: "user".to_string(),
                        content: format!("m{i}"),
                        timestamp: format!("17000000{i:02}"),
                        tokens: 1,
                        tool_calls: None,
                        tool_call_id: None,
                        reasoning_content: None,
                    }],
                )
                .unwrap();
            }
        }
        // offset=0 limit=2 → newest two (m3, m4)
        let r = call_str(|| aacode_session_messages(pp.as_ptr(), sid.as_ptr(), 0, 2));
        assert!(r.contains("\"total\":5"), "total: {r}");
        assert!(r.contains("m4") && r.contains("m3") && !r.contains("m0"), "page0: {r}");
        // offset=2 limit=2 → m1, m2
        let r = call_str(|| aacode_session_messages(pp.as_ptr(), sid.as_ptr(), 2, 2));
        assert!(r.contains("m2") && r.contains("m1") && !r.contains("m3"), "page1: {r}");
    }

    #[test]
    fn session_messages_null_args_error() {
        let r = call_str(|| aacode_session_messages(std::ptr::null(), std::ptr::null(), 0, 50));
        assert!(r.contains("\"success\":false"), "null args: {r}");
    }

    #[test]
    fn session_ffi_rejects_path_traversal_sid() {
        let proj = mkdirs(&tmp_root("sess_trav"));
        let pp = CString::new(proj).unwrap();
        let evil = CString::new("../escape").unwrap();
        let r = call_str(|| aacode_session_ensure(pp.as_ptr(), evil.as_ptr(), std::ptr::null()));
        assert!(r.contains("\"success\":false"), "path traversal must be rejected: {r}");
        assert!(r.contains("invalid session_id"));
        let r = call_str(|| aacode_session_messages(pp.as_ptr(), evil.as_ptr(), 0, 50));
        assert!(r.contains("\"success\":false"));
        let r = call_str(|| aacode_session_delete(pp.as_ptr(), evil.as_ptr()));
        assert!(r.contains("\"success\":false"));
    }

    #[test]
    fn session_messages_limit_zero_and_beyond_total() {
        let proj = mkdirs(&tmp_root("sess_l0"));
        let pp = CString::new(proj.clone()).unwrap();
        let sid = CString::new("s1").unwrap();
        {
            let mut sm = SessionManager::new(std::path::Path::new(&proj));
            sm.ensure_session("s1", "").unwrap();
            sm.append_session_messages(
                "s1",
                vec![SessionMessage {
                    role: "user".to_string(),
                    content: "hi".to_string(),
                    timestamp: "1".to_string(),
                    tokens: 1,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                }],
            )
            .unwrap();
        }
        // limit=0 → no messages, total still reported.
        let r = call_str(|| aacode_session_messages(pp.as_ptr(), sid.as_ptr(), 0, 0));
        assert!(r.contains("\"total\":1"), "limit=0 total: {r}");
        assert!(!r.contains("\"hi\""), "limit=0 must return no messages: {r}");
        // offset beyond total → empty.
        let r = call_str(|| aacode_session_messages(pp.as_ptr(), sid.as_ptr(), 5, 10));
        assert!(r.contains("\"total\":1"));
        assert!(!r.contains("\"hi\""), "offset beyond total must be empty: {r}");
    }

    #[test]
    fn session_unicode_content_roundtrip() {
        let proj = mkdirs(&tmp_root("sess_uni"));
        let pp = CString::new(proj).unwrap();
        let sid = CString::new("s1").unwrap();
        let msgs = r#"[{"role":"user","content":"你好，世界 🌍","timestamp":"1700000001"}]"#;
        let mj = CString::new(msgs).unwrap();
        let r = call_str(|| aacode_session_append(pp.as_ptr(), sid.as_ptr(), mj.as_ptr()));
        assert!(r.contains("\"success\":true"), "append unicode: {r}");
        let r = call_str(|| aacode_session_messages(pp.as_ptr(), sid.as_ptr(), 0, 50));
        assert!(r.contains("你好，世界"), "unicode content preserved: {r}");
    }

    #[test]
    fn session_list_empty_project() {
        let proj = mkdirs(&tmp_root("sess_empty"));
        let pp = CString::new(proj).unwrap();
        let r = call_str(|| aacode_session_list(pp.as_ptr()));
        assert!(r.contains("\"success\":true"));
        assert!(r.contains("\"sessions\":[]"), "empty project: {r}");
    }
}
