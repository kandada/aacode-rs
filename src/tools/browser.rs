// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Browser tools backed by the **fastbrowser** kernel (`engine = "webview"`).
//!
//! The host app provides an **offscreen** native WebView (Android WebView /
//! iOS WKWebView) through `aacode_browser_register_webview_ops`; the user never
//! sees a browser UI — only the normal tool-call cards in the chat.
//!
//! Tools exposed (kept intentionally few; capabilities are reached through
//! `browser_tools` + `browser_call`, so the prompt stays small):
//!   * `fetch_rendered` — open → render JS → return text (stateless).
//!   * `browser_tools`  — list the fastbrowser tools available for this engine.
//!   * `browser_call`   — call any of them by name.
//!
//! Enabled by the `browser` cargo feature; if no host WebView is registered the
//! tools are simply not registered (the agent behaves exactly as before).

use super::registry::{Tool, ToolRegistry};
use super::schema::{ParamType, ToolParameter, ToolSchema};
use super::web;
use crate::error::{AacodeError, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, OnceLock};

use fastbrowser::engine::Viewport as FbViewport;
use fastbrowser::sdk::Fastbrowser;
use fastbrowser::Config as FbConfig;

static BROWSER: OnceLock<Mutex<Option<Arc<Fastbrowser>>>> = OnceLock::new();

/// Serializes actual engine execution. The agent can issue several browser
/// tools in one turn; without this they race on the single shared engine
/// (navigate/type/click/execute_js interleave, so reads see stale/blank pages
/// and writes are lost).
static ENGINE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn lock_engine() -> std::sync::MutexGuard<'static, ()> {
    ENGINE_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn browser_slot() -> &'static Mutex<Option<Arc<Fastbrowser>>> {
    BROWSER.get_or_init(|| Mutex::new(None))
}

/// Shared, lazily-initialized fastbrowser (`engine = "webview"`).
///
/// Returns `None` when the host has not registered a WebView backend (e.g. on
/// desktop, or before `aacode_browser_register_webview_ops` was called).
pub fn browser() -> Option<Arc<Fastbrowser>> {
    let mut guard = browser_slot().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(b) = guard.as_ref() {
        return Some(b.clone());
    }
    let b = Arc::new(Fastbrowser::new());
    let cfg = FbConfig {
        engine: "webview".into(),
        // Mobile-ish viewport; the offscreen WebView is sized to this.
        viewport: Some(FbViewport::new(390, 844)),
        ..FbConfig::default()
    };
    match b.init(cfg) {
        Ok(()) => {
            *guard = Some(b.clone());
            Some(b)
        }
        Err(_) => None,
    }
}

/// Whether a browser backend is available (host WebView registered).
pub fn is_available() -> bool {
    browser().is_some()
}

/// Register the browser tools. No-op when no browser backend is available.
pub fn register(reg: &mut ToolRegistry, project_path: PathBuf) {
    if browser().is_none() {
        return;
    }
    reg.register(Box::new(FetchRenderedTool));
    reg.register(Box::new(BrowserToolsTool));
    reg.register(Box::new(BrowserCallTool { project_path }));
}

/// Run a blocking fastbrowser call off the async executor.
async fn blocking<T, F>(f: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| AacodeError::Other(format!("browser task join error: {e}")))?
}

fn engine_unavailable() -> String {
    json!({
        "success": false,
        "error": "browser backend unavailable (host WebView not registered)"
    })
    .to_string()
}

/// Map a fastbrowser engine error into an aacode error.
fn fb_err(e: fastbrowser::EngineError) -> AacodeError {
    AacodeError::Other(format!("browser: {e}"))
}

/// Per-tool parameter types, parsed once from fastbrowser's `tool_list()`
/// (`tool_name -> { param_name -> json_type }`). Used to coerce string args the
/// model sends for numeric/boolean params (e.g. `"timeout_ms": "10000"`).
fn tool_param_types() -> &'static HashMap<String, HashMap<String, String>> {
    static TYPES: OnceLock<HashMap<String, HashMap<String, String>>> = OnceLock::new();
    TYPES.get_or_init(|| {
        let mut map: HashMap<String, HashMap<String, String>> = HashMap::new();
        if let Some(b) = browser() {
            let list = b.tool_list();
            {
                let arr = list
                    .as_array()
                    .cloned()
                    .or_else(|| list.get("tools").and_then(|v| v.as_array()).cloned())
                    .unwrap_or_default();
                for t in arr {
                    let Some(name) = t.get("name").and_then(|v| v.as_str()) else {
                        continue;
                    };
                    let mut pm = HashMap::new();
                    if let Some(params) = t.get("params").and_then(|v| v.as_object()) {
                        for (pk, pv) in params {
                            if let Some(ty) = pv.get("type").and_then(|v| v.as_str()) {
                                pm.insert(pk.clone(), ty.to_string());
                            }
                        }
                    }
                    map.insert(name.to_string(), pm);
                }
            }
        }
        map
    })
}

/// Coerce string args to the declared numeric/boolean types (LLMs commonly send
/// `"10000"` for an integer param). Unknown params/types pass through unchanged.
fn coerce_with(
    types: &HashMap<String, HashMap<String, String>>,
    tool: &str,
    params: Value,
) -> Value {
    let Some(pm) = types.get(tool) else {
        return params;
    };
    let Some(obj) = params.as_object() else {
        return params;
    };
    let mut out = serde_json::Map::new();
    for (k, v) in obj {
        let coerced = match pm.get(k).map(String::as_str) {
            Some("integer") | Some("number") => match v.as_str() {
                Some(s) => match s.parse::<i64>() {
                    Ok(i) => json!(i),
                    Err(_) => match s.parse::<f64>() {
                        Ok(f) => json!(f),
                        Err(_) => v.clone(),
                    },
                },
                None => v.clone(),
            },
            Some("boolean") => match v.as_str() {
                Some("true") => json!(true),
                Some("false") => json!(false),
                _ => v.clone(),
            },
            _ => v.clone(),
        };
        out.insert(k.clone(), coerced);
    }
    Value::Object(out)
}

/// Preview length (chars) returned by `browser_call` in `compact` mode when the
/// caller does not pass `max_chars`.
const COMPACT_PREVIEW_CHARS: usize = 6_000;
/// Raw results larger than this are auto-compacted (aligned with the unified
/// observation cap of `limits.tool_output_chars`, 24k by default).
const AUTO_COMPACT_CHARS: usize = 24_000;
/// Keep at most this many `browser_*` archive files to bound disk usage.
const ARCHIVE_KEEP: usize = 20;

/// Normalize fastbrowser's tool list (`[...]` or `{tools:[...]}`) to a Vec.
fn tool_array(list: &Value) -> Vec<Value> {
    list.as_array()
        .cloned()
        .or_else(|| list.get("tools").and_then(|v| v.as_array()).cloned())
        .unwrap_or_default()
}

/// Max chars for a listing description (full text lives in `help` mode).
const MAX_LISTING_DESC: usize = 80;

/// First line of a description, capped to `MAX_LISTING_DESC` chars.
fn short_desc(s: &str) -> String {
    let line = s.lines().next().unwrap_or("").trim();
    if line.chars().count() <= MAX_LISTING_DESC {
        line.to_string()
    } else {
        let mut t: String = line.chars().take(MAX_LISTING_DESC - 1).collect();
        t.push('…');
        t
    }
}

/// Listing form: `{name, description(≤80), params}` (drops `schema` and
/// `example`; the full schema is available via `mode:"help"`).
fn slim_tools(list: &Value) -> Value {
    let out: Vec<Value> = tool_array(list)
        .into_iter()
        .map(|t| {
            json!({
                "name": t.get("name").cloned().unwrap_or(Value::Null),
                "description": short_desc(t.get("description").and_then(|v| v.as_str()).unwrap_or("")),
                "params": t.get("params").cloned().unwrap_or_else(|| json!({})),
            })
        })
        .collect();
    json!(out)
}

/// Full schema for a single tool (progressive disclosure). `None` if unknown.
fn help_tool(list: &Value, name: &str) -> Option<Value> {
    tool_array(list)
        .into_iter()
        .find(|t| t.get("name").and_then(|v| v.as_str()) == Some(name))
        .map(|t| {
            json!({
                "name": t.get("name").cloned().unwrap_or(Value::Null),
                "description": t.get("description").cloned().unwrap_or(Value::Null),
                "params": t.get("params").cloned().unwrap_or_else(|| json!({})),
                "example": t.get("example").cloned().unwrap_or(Value::Null),
            })
        })
}

/// Smaller listing: `{name, description, params, required}` — parameter *names*
/// only (no types/examples), plus which are required, so the model doesn't have
/// to guess param names.
fn compact_tools(list: &Value) -> Value {
    let out: Vec<Value> = tool_array(list)
        .into_iter()
        .map(|t| {
            let desc = t.get("description").and_then(|v| v.as_str()).unwrap_or("");
            let mut params: Vec<String> = Vec::new();
            let mut required: Vec<String> = Vec::new();
            if let Some(obj) = t.get("params").and_then(|v| v.as_object()) {
                for (k, v) in obj {
                    params.push(k.clone());
                    if v.get("required").and_then(|r| r.as_bool()).unwrap_or(false) {
                        required.push(k.clone());
                    }
                }
            }
            json!({
                "name": t.get("name").cloned().unwrap_or(Value::Null),
                "description": short_desc(desc),
                "params": params,
                "required": required,
            })
        })
        .collect();
    json!(out)
}

/// Names only.
fn tool_names(list: &Value) -> Value {
    Value::Array(
        tool_array(list)
            .into_iter()
            .filter_map(|t| t.get("name").cloned())
            .collect(),
    )
}

/// Collapse whitespace, drop blank lines and consecutive duplicate lines.
/// Deterministic (no model call) — cuts tokens from noisy page dumps.
fn clean_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len().min(64 * 1024));
    let mut prev: Option<String> = None;
    for raw in s.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if prev.as_deref() == Some(line) {
            continue;
        }
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(line);
        prev = Some(line.to_string());
    }
    out
}

/// First `limit` chars, plus a trailing marker when truncated.
fn truncate_chars(s: &str, limit: usize) -> (String, bool) {
    let total = s.chars().count();
    if limit == 0 || total <= limit {
        return (s.to_string(), false);
    }
    let head: String = s.chars().take(limit).collect();
    (format!("{head}\n…[truncated, {total} chars total]"), true)
}

/// The tool result's dominant text payload (for cleaning + archiving), if any.
fn dominant_text_field(result: &Value) -> Option<(&'static str, &str)> {
    for key in ["text", "content", "markdown", "html", "value"] {
        if let Some(s) = result.get(key).and_then(|v| v.as_str()) {
            if !s.trim().is_empty() {
                return Some((key, s));
            }
        }
    }
    None
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Archive a full result under `.aacode/extracts/` and return its path.
fn archive_extracts(project_path: &Path, tool: &str, ext: &str, content: &str) -> Option<String> {
    let dir = project_path.join(".aacode").join("extracts");
    std::fs::create_dir_all(&dir).ok()?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let name = format!(
        "browser_{}_{}_{}.{}",
        sanitize(tool),
        stamp,
        uuid::Uuid::new_v4().simple(),
        ext
    );
    let path = dir.join(name);
    std::fs::write(&path, content).ok()?;
    prune_browser_extracts(&dir);
    Some(path.to_string_lossy().to_string())
}

/// Keep only the most recent [`ARCHIVE_KEEP`] `browser_*` extracts.
fn prune_browser_extracts(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut files: Vec<_> = entries
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with("browser_"))
        .collect();
    if files.len() > ARCHIVE_KEEP {
        files.sort_by_key(|e| e.metadata().and_then(|m| m.modified()).ok());
        for old in files.iter().take(files.len() - ARCHIVE_KEEP) {
            let _ = std::fs::remove_file(old.path());
        }
    }
}

/// Accept booleans the model may send as strings/numbers (`"true"`, `1`).
fn arg_bool(v: Option<&Value>) -> bool {
    match v {
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => s == "true" || s == "1",
        Some(Value::Number(n)) => n.as_i64().unwrap_or(0) != 0,
        _ => false,
    }
}

/// Accept integers the model may send as strings.
fn arg_usize(v: Option<&Value>) -> usize {
    match v {
        Some(Value::Number(n)) => n.as_u64().unwrap_or(0) as usize,
        Some(Value::String(s)) => s.trim().parse::<usize>().unwrap_or(0),
        _ => 0,
    }
}

/// Resolve a `download` tool's `path` into `project_path`.
///
/// - Absolute paths already inside the project are used **as-is**.
/// - Absolute paths **outside** the project (e.g. `/sdcard/Download/x.jpg` on
///   Android) are remapped into the project by file name — the app sandbox is
///   the only reliably writable location on mobile.
/// - Relative paths are joined to `project_path`, rejecting `../` escapes.
/// fastbrowser writes the resulting absolute path.
fn resolve_download_path(project_path: &Path, params: &mut Value) {
    let Some(obj) = params.as_object_mut() else {
        return;
    };
    let Some(rel) = obj.get("path").and_then(|v| v.as_str()).map(str::to_string) else {
        return;
    };
    if rel.is_empty() {
        return;
    }
    if rel.starts_with('/') {
        let abs = normalize_lexical(Path::new(&rel));
        let root = normalize_lexical(project_path);
        // Already inside the project → use as-is.
        if abs == root || abs.starts_with(&root) {
            obj.insert("path".to_string(), json!(abs.to_string_lossy()));
            return;
        }
        // Absolute path outside the project (e.g. `/sdcard/Download/x.jpg` on
        // Android, `/var/mobile/...` on iOS, or the *logical* sandbox path
        // `/projects/<name>/sub/f.jpg`) is not reliably writable inside the app
        // sandbox → remap it into the project, **preserving the sub-path** so the
        // agent finds it where it asked (`<project>/sub/f.jpg`).
        let proj_name = root
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let comps: Vec<std::path::Component> = abs.components().collect();
        // If the absolute path contains the project directory name (the logical
        // sandbox path does), start *after* its last occurrence so we don't
        // re-insert the project prefix.
        let mut start = 0;
        if !proj_name.is_empty() {
            for (i, c) in comps.iter().enumerate() {
                if let std::path::Component::Normal(s) = c {
                    if s.to_string_lossy() == proj_name {
                        start = i + 1;
                    }
                }
            }
        }
        let mut rel = PathBuf::new();
        for c in comps[start..].iter() {
            if let std::path::Component::Normal(s) = c {
                rel.push(s);
            }
        }
        let mapped = if rel.as_os_str().is_empty() {
            root.join("download.bin")
        } else {
            root.join(rel)
        };
        obj.insert("path".to_string(), json!(mapped.to_string_lossy()));
        return;
    }
    let root = normalize_lexical(project_path);
    let candidate = normalize_lexical(&root.join(&rel));
    if candidate != root && !candidate.starts_with(&root) {
        return;
    }
    obj.insert("path".to_string(), json!(candidate.to_string_lossy()));
}

fn normalize_lexical(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            c => out.push(c.as_os_str()),
        }
    }
    out
}

/// Structure-aware preview body for a fastbrowser result (before truncation):
/// HTML → text (tags/script/style stripped), text → whitespace/dedupe cleaned,
/// links/images → de-duplicated, otherwise pretty JSON.
fn compact_body(result: &Value) -> String {
    if let Some(h) = result.get("html").and_then(|v| v.as_str()) {
        return web::html_to_text(h);
    }
    if let Some(arr) = result.get("links").and_then(|v| v.as_array()) {
        return serde_json::to_string_pretty(&dedupe_by_url(arr, "url", 200)).unwrap_or_default();
    }
    if let Some(arr) = result.get("images").and_then(|v| v.as_array()) {
        return serde_json::to_string_pretty(&dedupe_by_url(arr, "src", 200)).unwrap_or_default();
    }
    if let Some((_, s)) = dominant_text_field(result) {
        return clean_text(s);
    }
    serde_json::to_string_pretty(result).unwrap_or_default()
}

/// Lossless raw payload + extension for archiving (HTML / text / JSON).
fn raw_payload(result: &Value) -> (&'static str, String) {
    if let Some(h) = result.get("html").and_then(|v| v.as_str()) {
        return ("html", h.to_string());
    }
    if let Some((_, s)) = dominant_text_field(result) {
        return ("txt", s.to_string());
    }
    (
        "json",
        serde_json::to_string_pretty(result).unwrap_or_default(),
    )
}

/// De-duplicate an array of objects by `key`, dropping empty / `javascript:` /
/// `#` values, capped at `cap` entries.
fn dedupe_by_url(arr: &[Value], key: &str, cap: usize) -> Vec<Value> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for it in arr {
        let u = it.get(key).and_then(|v| v.as_str()).unwrap_or("").trim();
        if u.is_empty() || u.starts_with("javascript:") || u == "#" {
            continue;
        }
        if !seen.insert(u.to_string()) {
            continue;
        }
        out.push(it.clone());
        if out.len() >= cap {
            break;
        }
    }
    out
}

/// Build the `compact` observation: a structure-aware, cleaned preview plus the
/// lossless raw payload archived to `.aacode/extracts/` when it is large.
fn compact_result(project_path: &Path, tool: &str, result: Value, max_chars: usize) -> String {
    let limit = if max_chars > 0 {
        max_chars
    } else {
        COMPACT_PREVIEW_CHARS
    };
    // Archive any embedded binary (base64 / data URI) BEFORE truncating the
    // preview — otherwise a truncated base64 blob is undecodable and useless.
    let store = crate::artifacts::ArtifactStore::new(project_path);
    let (result, artifacts) = crate::observation::redact_binary(&result, &store);
    let body = compact_body(&result);
    let (raw_ext, raw_full) = raw_payload(&result);
    let raw_chars = raw_full.chars().count();
    let (preview, truncated) = truncate_chars(&body, limit);

    let mut out = json!({
        "success": true,
        "result": preview,
        "truncated": truncated,
        "chars": body.chars().count(),
    });
    if artifacts > 0 {
        out["artifacts"] = json!(artifacts);
        out["hint_binary"] = json!(
            "Binary payload(s) were saved under .aacode/artifacts/ (see the `artifact` fields above). Inspect a bounded sample with run_shell (e.g. `xxd -l 64 <path>`), or use understand_image with the artifact path to view an image."
        );
    }
    if raw_chars > limit {
        if let Some(path) = archive_extracts(project_path, tool, raw_ext, &raw_full) {
            out["archive"] = json!({"path": path, "chars": raw_chars});
            out["hint"] = json!(format!(
                "Full raw result ({raw_chars} chars) saved to {path}. Explore it with run_shell (grep/head/tail) or execute_python; avoid cat-ing the whole file into context."
            ));
        }
    }
    out.to_string()
}

// ─────────────────────────── fetch_rendered ─────────────────────────────

/// Stateless "open → render JS → extract text → close".
pub struct FetchRenderedTool;

#[async_trait::async_trait]
impl Tool for FetchRenderedTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "fetch_rendered",
            "Fetch a URL in a real (offscreen) browser that executes JS, then return the rendered text. Use for SPAs / JS-heavy pages that fetch_url cannot read.",
            vec![
                ToolParameter::new(
                    "url",
                    ParamType::String,
                    true,
                    "URL to render",
                    &["link", "uri", "address"],
                ),
                ToolParameter::new(
                    "max_chars",
                    ParamType::Integer,
                    false,
                    "Max returned characters (default 5000)",
                    &["max_length", "max_content_length"],
                ),
            ],
        )
    }

    async fn call(&self, args: &Value, _cancel: &AtomicBool) -> Result<String> {
        let url = args
            .get("url")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if url.is_empty() {
            return Ok(json!({"success": false, "error": "empty url"}).to_string());
        }
        let max_chars = args
            .get("max_chars")
            .and_then(|v| v.as_u64())
            .unwrap_or(5000) as usize;
        let Some(b) = browser() else {
            return Ok(engine_unavailable());
        };

        let url_for_open = url.clone();
        let out = blocking(move || -> Result<Value> {
            let _guard = lock_engine();
            let opened = b.open(&url_for_open).map_err(fb_err)?;
            let tab = opened.get("tab").and_then(|v| v.as_u64()).unwrap_or(0);
            // The host WebView `load()` is asynchronous, so a fixed sleep can read
            // the blank/previous page on slow sites. Wait for the real load to
            // finish first, then a short settle for SPA hydration / lazy content.
            // Wait for the real load; prefer `networkidle` (SPAs keep
            // hydrating after `load`) and fall back to `load`.
            let _ = b.tool_call(
                "wait_for_load_state",
                json!({"tab": tab, "state": "networkidle", "timeout_ms": 8000}),
            );
            let _ = b.tool_call(
                "wait_for_load_state",
                json!({"tab": tab, "state": "load", "timeout_ms": 8000}),
            );
            // Lazy/async content: poll extract_text until non-empty (bounded),
            // instead of a single fixed sleep that can read a blank page.
            let mut text = String::new();
            for _ in 0..12 {
                std::thread::sleep(std::time::Duration::from_millis(300));
                let tv = b
                    .tool_call("extract_text", json!({"tab": tab, "max_chars": max_chars}))
                    .unwrap_or(Value::Null);
                text = tv
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if !text.trim().is_empty() {
                    break;
                }
            }
            let title_val = b
                .tool_call("get_page_title", json!({"tab": tab}))
                .unwrap_or(Value::Null);
            // Close the tab so repeated fetches stay stateless.
            let _ = b.tool_call("close_tab", json!({"tab": tab}));
            Ok(json!({
                "text": text,
                "title": title_val.get("title").and_then(|v| v.as_str()).unwrap_or(""),
                "final_url": opened.get("url").and_then(|v| v.as_str()).unwrap_or(&url_for_open),
            }))
        })
        .await?;

        let text = out.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let total = text.chars().count();
        if total == 0 {
            return Ok(json!({
                "success": false,
                "url": url,
                "title": out.get("title").and_then(|v| v.as_str()).unwrap_or(""),
                "content": "",
                "content_length": 0,
                "error": "no readable content (page may be JS-only, empty, or blocked)",
                "hint": "No text rendered — the page may need interaction or is behind an anti-bot/JS challenge. \
                         Use `browser_tools` + `browser_call`: `navigate` to the URL, then `get_page_text` (or `extract_text`); \
                         plain `fetch_url` will not pass the challenge.",
            })
            .to_string());
        }
        let content: String = text.chars().take(max_chars).collect();
        let mut v = json!({
            "success": true,
            "url": url,
            "title": out.get("title").and_then(|v| v.as_str()).unwrap_or(""),
            "content": content,
            "content_length": total,
        });
        // The page rendered, but it may be an anti-bot challenge (e.g. a slider
        // captcha). Flag it so the model doesn't treat the challenge text as
        // real content — same semantics as `fetch_url`.
        if let Some(reason) = crate::tools::web::detect_block(200, text, total) {
            v["blocked"] = json!(true);
            v["block_reason"] = json!(reason);
            v["hint"] = json!(
                "Rendered page looks like an anti-bot / JS challenge. Try the site's mobile/AMP version or a public API, or interact via browser_call."
            );
        }
        Ok(v.to_string())
    }
}

// ─────────────────────────── browser_tools ──────────────────────────────

/// List the fastbrowser tools available for the current engine (capability
/// filtered). Lets the model discover capabilities without bloating the prompt.
pub struct BrowserToolsTool;

#[async_trait::async_trait]
impl Tool for BrowserToolsTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "browser_tools",
            "Discover browser tools (only needed for the long tail — call `browser_call` directly for common actions). `mode`: 'names' (default, names only), 'compact' (name + short description + param names), 'full' (adds param types), or 'help' (full schema for ONE tool — also pass `name`).",
            vec![
                ToolParameter::new(
                    "mode",
                    ParamType::String,
                    false,
                    "Detail level: 'names' (default), 'compact', 'full', or 'help'.",
                    &["detail", "level"],
                ),
                ToolParameter::new(
                    "name",
                    ParamType::String,
                    false,
                    "With mode='help': the tool to describe.",
                    &["tool"],
                ),
            ],
        )
    }

    async fn call(&self, args: &Value, _cancel: &AtomicBool) -> Result<String> {
        let Some(b) = browser() else {
            return Ok(engine_unavailable());
        };
        let mode = args
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("names")
            .to_string();
        let want = args
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let list = blocking(move || Ok::<Value, AacodeError>(b.tool_list())).await?;
        if mode == "help" {
            if want.is_empty() {
                return Ok(
                    json!({"success": false, "error": "mode='help' requires `name`"}).to_string(),
                );
            }
            return Ok(match help_tool(&list, &want) {
                Some(t) => json!({"success": true, "mode": "help", "tool": t}),
                None => {
                    json!({"success": false, "error": format!("unknown browser tool '{want}'")})
                }
            }
            .to_string());
        }
        let tools = match mode.as_str() {
            "compact" => compact_tools(&list),
            "full" => slim_tools(&list),
            _ => tool_names(&list),
        };
        let count = tool_array(&list).len();
        Ok(json!({"success": true, "mode": mode, "count": count, "tools": tools}).to_string())
    }
}

// ─────────────────────────── browser_call ───────────────────────────────

/// Call any fastbrowser tool by name. `args` is the tool's parameter object.
pub struct BrowserCallTool {
    /// Project root; `compact` archives the full result under `.aacode/extracts/`.
    pub project_path: PathBuf,
}

#[async_trait::async_trait]
impl Tool for BrowserCallTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "browser_call",
            "Call a browser tool by name (see browser_tools). 'args' is that tool's parameters as an object. Returns the raw result by default; if it exceeds ~24k chars it is AUTO-compacted (structure-aware: HTML→text, link/image de-dupe; binaries saved under .aacode/artifacts/, full text under .aacode/extracts/) — the result then carries `compacted:true`. Pass compact=true to always clean, or max_chars to size the preview.",
            vec![
                ToolParameter::new("name", ParamType::String, true, "Browser tool name", &["tool"]),
                ToolParameter::new(
                    "args",
                    ParamType::Object,
                    false,
                    "Tool parameters (default {})",
                    &["params", "arguments"],
                ),
                ToolParameter::new(
                    "compact",
                    ParamType::Boolean,
                    false,
                    "Return cleaned+truncated output and archive the full result (default false = full).",
                    &["clean"],
                ),
                ToolParameter::new(
                    "max_chars",
                    ParamType::Integer,
                    false,
                    "Preview length for compact mode (default 6000).",
                    &["max_length"],
                ),
            ],
        )
    }

    async fn call(&self, args: &Value, _cancel: &AtomicBool) -> Result<String> {
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if name.is_empty() {
            return Ok(json!({"success": false, "error": "empty tool name"}).to_string());
        }
        let mut params = args.get("args").cloned().unwrap_or_else(|| json!({}));
        // fastbrowser's `download` writes `path` verbatim; resolve it into the
        // project workdir (rejecting escapes) so a relative path just works.
        if name == "download" {
            resolve_download_path(&self.project_path, &mut params);
        }
        let compact = arg_bool(args.get("compact"));
        let max_chars = arg_usize(args.get("max_chars"));
        let Some(b) = browser() else {
            return Ok(engine_unavailable());
        };
        // Tool-name/param aliases (Playwright / browser-use) are resolved inside
        // fastbrowser; here we only coerce string args to declared types.
        let params = coerce_with(tool_param_types(), &name, params);
        let name_for_call = name.clone();
        let result = blocking(move || {
            let _guard = lock_engine();
            b.tool_call(&name_for_call, params).map_err(fb_err)
        })
        .await?;
        if !compact {
            // Auto-compact an oversized raw result (aligned with the unified
            // observation cap) so huge payloads are filtered structurally rather
            // than dumped. The caller is told compact was applied.
            let raw_len = serde_json::to_string(&result).map(|s| s.len()).unwrap_or(0);
            if raw_len > AUTO_COMPACT_CHARS {
                let mut v: Value =
                    serde_json::from_str(&compact_result(&self.project_path, &name, result, 0))
                        .unwrap_or_else(|_| json!({"success": true}));
                v["compacted"] = json!(true);
                v["hint"] = json!(format!(
                    "Result was {raw_len} chars (> {AUTO_COMPACT_CHARS}); auto-compacted to a cleaned preview. Full text archived under .aacode/extracts/, binaries under .aacode/artifacts/."
                ));
                return Ok(v.to_string());
            }
            return Ok(json!({"success": true, "result": result}).to_string());
        }
        Ok(compact_result(&self.project_path, &name, result, max_chars))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call_tool() -> BrowserCallTool {
        BrowserCallTool {
            project_path: PathBuf::from("."),
        }
    }

    #[test]
    fn schemas_are_well_formed() {
        let s = FetchRenderedTool.schema();
        assert_eq!(s.name, "fetch_rendered");
        assert!(s.parameters.iter().any(|p| p.name == "url" && p.required));
        assert_eq!(call_tool().schema().name, "browser_call");
        assert_eq!(BrowserToolsTool.schema().name, "browser_tools");
        // compact knobs are exposed on browser_call
        let bc = call_tool().schema();
        assert!(bc.parameters.iter().any(|p| p.name == "compact"));
        assert!(bc.parameters.iter().any(|p| p.name == "max_chars"));
    }

    #[tokio::test]
    async fn fetch_rendered_empty_url() {
        let out = FetchRenderedTool
            .call(&json!({"url": ""}), &AtomicBool::new(false))
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&out).unwrap()["success"],
            false
        );
    }

    #[tokio::test]
    async fn browser_call_empty_name() {
        let out = call_tool()
            .call(&json!({}), &AtomicBool::new(false))
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&out).unwrap()["success"],
            false
        );
    }

    #[test]
    fn is_unavailable_without_host_webview() {
        // Tests never register a host WebView → the backend must be unavailable.
        assert!(!is_available());
    }

    #[test]
    fn register_is_noop_without_host() {
        let mut reg = ToolRegistry::new();
        register(&mut reg, PathBuf::from("."));
        assert!(!reg.contains("fetch_rendered"));
        assert!(!reg.contains("browser_tools"));
        assert!(!reg.contains("browser_call"));
    }

    #[test]
    fn coerces_string_numbers_and_booleans() {
        let mut types: HashMap<String, HashMap<String, String>> = HashMap::new();
        let mut pm = HashMap::new();
        pm.insert("timeout_ms".to_string(), "number".to_string());
        pm.insert("limit".to_string(), "integer".to_string());
        pm.insert("checked".to_string(), "boolean".to_string());
        pm.insert("text".to_string(), "string".to_string());
        types.insert("wait_for_text".to_string(), pm);

        let out = coerce_with(
            &types,
            "wait_for_text",
            json!({"timeout_ms": "10000", "limit": "10", "checked": "true", "text": "123"}),
        );
        assert_eq!(out["timeout_ms"], json!(10000));
        assert_eq!(out["limit"], json!(10));
        assert_eq!(out["checked"], json!(true));
        assert_eq!(out["text"], json!("123")); // string param stays a string
    }

    #[test]
    fn download_path_resolves_inside_project() {
        let root = PathBuf::from("/proj");
        let mut p = json!({"url": "u", "path": "a/b.jpg"});
        resolve_download_path(&root, &mut p);
        assert_eq!(p["path"], json!("/proj/a/b.jpg"));
        // An absolute path outside the project is remapped by file name into
        // the project (the only reliably writable place on mobile).
        let mut p = json!({"path": "/a.png"});
        resolve_download_path(&root, &mut p);
        assert_eq!(p["path"], json!("/proj/a.png"));
        // A full mobile host path is likewise remapped (never double-prefixed).
        let mut p = json!({"path": "/var/mobile/App/Documents/proj/a.jpg"});
        resolve_download_path(&root, &mut p);
        assert_eq!(p["path"], json!("/proj/a.jpg"));
        // An absolute path already inside the project is left as-is.
        let mut p = json!({"path": "/proj/sub/a.jpg"});
        resolve_download_path(&root, &mut p);
        assert_eq!(p["path"], json!("/proj/sub/a.jpg"));
        // Escapes are left unchanged (never resolved outside the project).
        let mut p = json!({"path": "../evil"});
        resolve_download_path(&root, &mut p);
        assert_eq!(p["path"], json!("../evil"));
    }

    #[test]
    fn download_path_normalizes_dot_and_inner_dotdot() {
        let root = PathBuf::from("/proj");
        // `.` components collapse; `..` that stays inside is resolved.
        let mut p = json!({"path": "./a/./b.jpg"});
        resolve_download_path(&root, &mut p);
        assert_eq!(p["path"], json!("/proj/a/b.jpg"));
        let mut p = json!({"path": "a/../b.jpg"});
        resolve_download_path(&root, &mut p);
        assert_eq!(p["path"], json!("/proj/b.jpg"));
        // Deep escape via an inner `..` is rejected.
        let mut p = json!({"path": "a/../../evil"});
        resolve_download_path(&root, &mut p);
        assert_eq!(p["path"], json!("a/../../evil"));
        // Multiple leading slashes: absolute path, collapsed, then remapped
        // preserving the sub-path.
        let mut p = json!({"path": "///x/y.png"});
        resolve_download_path(&root, &mut p);
        assert_eq!(p["path"], json!("/proj/x/y.png"));
    }

    #[test]
    fn download_path_leaves_other_params_untouched() {
        let root = PathBuf::from("/proj");
        // No `path` key: params unchanged.
        let mut p = json!({"url": "u"});
        resolve_download_path(&root, &mut p);
        assert_eq!(p, json!({"url": "u"}));
        // Empty path: unchanged (fastbrowser keeps its own default).
        let mut p = json!({"path": ""});
        resolve_download_path(&root, &mut p);
        assert_eq!(p["path"], json!(""));
        // Non-string path: unchanged, no panic.
        let mut p = json!({"path": 42});
        resolve_download_path(&root, &mut p);
        assert_eq!(p["path"], json!(42));
        // Non-object params: no panic, no change.
        let mut p = json!("not-an-object");
        resolve_download_path(&root, &mut p);
        assert_eq!(p, json!("not-an-object"));
        // Other keys survive when path is resolved.
        let mut p = json!({"path": "f.bin", "timeout_ms": 500});
        resolve_download_path(&root, &mut p);
        assert_eq!(p["path"], json!("/proj/f.bin"));
        assert_eq!(p["timeout_ms"], json!(500));
    }

    #[test]
    fn normalize_lexical_collapses_components() {
        assert_eq!(
            normalize_lexical(Path::new("/a/./b/../c")),
            PathBuf::from("/a/c")
        );
        // Popping past the root is clamped, not an underflow panic.
        assert_eq!(
            normalize_lexical(Path::new("/a/../../b")),
            PathBuf::from("/b")
        );
    }

    #[test]
    fn slim_tools_drops_schema() {
        let list = json!([
            {"name": "navigate", "description": "line1\nline2", "params": {"url": {"type": "string"}}, "schema": {"type": "object"}, "example": "{}"}
        ]);
        let out = slim_tools(&list);
        assert_eq!(out[0]["name"], "navigate");
        assert_eq!(out[0]["description"], "line1");
        assert!(out[0].get("schema").is_none());
        assert!(out[0].get("example").is_none());
    }

    #[test]
    fn short_desc_caps_and_first_line() {
        assert_eq!(short_desc("a\nb"), "a");
        let long = "x".repeat(200);
        let d = short_desc(&long);
        assert_eq!(d.chars().count(), 80);
        assert!(d.ends_with('…'));
    }

    #[test]
    fn help_tool_returns_full_schema() {
        let list = json!([
            {"name": "navigate", "description": "go", "params": {"url": {"type": "string"}}, "example": "{\"url\":\"x\"}"}
        ]);
        let h = help_tool(&list, "navigate").unwrap();
        assert_eq!(h["name"], "navigate");
        assert_eq!(h["example"], "{\"url\":\"x\"}");
        assert!(help_tool(&list, "nope").is_none());
    }

    #[test]
    fn compact_and_names_modes_shrink_catalog() {
        let list = json!([
            {"name": "navigate", "description": "line1\nline2", "params": {"url": {"type": "string", "required": true}}, "example": "{}"},
            {"name": "click", "description": "click things", "params": {"id": {"type": "string"}, "ref": {"type": "string"}}}
        ]);
        let c = compact_tools(&list);
        assert_eq!(c[0]["name"], "navigate");
        assert_eq!(c[0]["description"], "line1");
        // compact keeps param names + which are required (no types/examples)
        assert_eq!(c[0]["params"], json!(["url"]));
        assert_eq!(c[0]["required"], json!(["url"]));
        assert!(c[0].get("example").is_none());
        let n = tool_names(&list);
        assert_eq!(n, json!(["navigate", "click"]));
    }

    #[test]
    fn clean_text_collapses_noise() {
        let raw = "  hello  \n\nhello\nworld\n   \nworld\nbye";
        assert_eq!(clean_text(raw), "hello\nworld\nbye");
    }

    #[test]
    fn truncate_marks_and_preserves_small() {
        let (s, t) = truncate_chars("short", 100);
        assert_eq!(s, "short");
        assert!(!t);
        let (s, t) = truncate_chars("abcdefghij", 4);
        assert!(t);
        assert!(s.starts_with("abcd"));
        assert!(s.contains("10 chars total"));
    }

    #[test]
    fn dominant_field_picks_text() {
        let r = json!({"text": "  body  ", "title": "t"});
        assert_eq!(dominant_text_field(&r), Some(("text", "  body  ")));
        let r = json!({"html": "<p>x</p>"});
        assert_eq!(dominant_text_field(&r).map(|(k, _)| k), Some("html"));
        assert!(dominant_text_field(&json!({"links": []})).is_none());
    }

    #[test]
    fn arg_helpers_accept_strings() {
        assert!(arg_bool(Some(&json!("true"))));
        assert!(arg_bool(Some(&json!(true))));
        assert!(!arg_bool(Some(&json!("false"))));
        assert!(!arg_bool(None));
        assert_eq!(arg_usize(Some(&json!("42"))), 42);
        assert_eq!(arg_usize(Some(&json!(7))), 7);
    }

    fn tmp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "aacode_browser_{tag}_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn compact_result_archives_and_previews() {
        let dir = tmp_dir("compact");
        let big: String = (0..2000)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let out: Value = serde_json::from_str(&compact_result(
            &dir,
            "get_page_text",
            json!({"text": big}),
            0,
        ))
        .unwrap();
        assert_eq!(out["success"], true);
        assert_eq!(out["truncated"], true);
        let path = out["archive"]["path"].as_str().unwrap();
        assert!(Path::new(path).exists());
        assert!(out["result"].as_str().unwrap().contains("truncated"));
        // small payload: no archive
        let small: Value = serde_json::from_str(&compact_result(
            &dir,
            "get_page_text",
            json!({"text": "hi"}),
            0,
        ))
        .unwrap();
        assert_eq!(small["truncated"], false);
        assert!(small.get("archive").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compact_result_html_is_stripped_and_raw_archived() {
        let dir = tmp_dir("html");
        let html = format!(
            "<html><head><style>.a{{color:red}}</style><script>var x=1;</script></head>\
             <body><h1>Title</h1><p>Hello&nbsp;world</p>{}</body></html>",
            "<div>row</div>".repeat(1000)
        );
        let out: Value = serde_json::from_str(&compact_result(
            &dir,
            "extract_html",
            json!({"html": html}),
            0,
        ))
        .unwrap();
        // Preview is cleaned text: no tags/script/style, entities decoded.
        let preview = out["result"].as_str().unwrap();
        assert!(!preview.contains('<'), "preview should be tag-free");
        assert!(!preview.contains("var x=1"), "script should be dropped");
        assert!(preview.contains("Hello world"));
        assert!(preview.contains("Title"));
        // Raw HTML archived losslessly with .html extension.
        let path = out["archive"]["path"].as_str().unwrap();
        assert!(path.ends_with(".html"), "got {path}");
        let saved = std::fs::read_to_string(path).unwrap();
        assert!(saved.contains("<h1>Title</h1>"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compact_body_strips_html_and_dedupes_links() {
        let html = "<div>a</div><script>x</script><div>a</div><div>b</div>";
        assert_eq!(compact_body(&json!({"html": html})), "a\nb");
        let links = json!({"links": [
            {"url": "https://x", "text": "X"},
            {"url": "https://x", "text": "dup"},
            {"url": "#", "text": "no"},
            {"url": "javascript:void(0)", "text": "no"},
            {"url": "https://y", "text": "Y"}
        ]});
        let parsed: Value = serde_json::from_str(&compact_body(&links)).unwrap();
        assert_eq!(parsed.as_array().unwrap().len(), 2);
    }

    #[test]
    fn dedupe_by_url_filters_and_caps() {
        let arr = vec![
            json!({"url": "a"}),
            json!({"url": "a"}),
            json!({"url": ""}),
            json!({"url": "b"}),
            json!({"url": "c"}),
        ];
        let out = dedupe_by_url(&arr, "url", 2);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0]["url"], "a");
        assert_eq!(out[1]["url"], "b");
    }

    #[test]
    fn compact_result_links_deduped_no_archive() {
        let dir = tmp_dir("json");
        let result = json!({"links": [{"url": "a", "text": "A"}, {"url": "b"}]});
        let out: Value =
            serde_json::from_str(&compact_result(&dir, "extract_links", result, 0)).unwrap();
        // No dominant text field → preview is the de-duped links array (small).
        assert_eq!(out["truncated"], false);
        let parsed: Value = serde_json::from_str(out["result"].as_str().unwrap()).unwrap();
        assert_eq!(parsed.as_array().unwrap().len(), 2);
        assert!(out.get("archive").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compact_result_respects_max_chars() {
        let dir = tmp_dir("maxchars");
        let body: String = (0..500)
            .map(|i| format!("row {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let out: Value = serde_json::from_str(&compact_result(
            &dir,
            "get_page_text",
            json!({"text": body}),
            25,
        ))
        .unwrap();
        assert_eq!(out["truncated"], true);
        // Preview is the 25-char head + a marker line.
        let preview = out["result"].as_str().unwrap();
        let marker = preview.find("\n…").unwrap();
        assert_eq!(preview[..marker].chars().count(), 25);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn archive_prunes_to_keep_limit() {
        let dir = tmp_dir("prune");
        for i in 0..(ARCHIVE_KEEP + 7) {
            let _ = archive_extracts(&dir, "get_page_text", "txt", &format!("body {i}"));
        }
        let count = std::fs::read_dir(dir.join(".aacode").join("extracts"))
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("browser_"))
            .count();
        assert!(
            count <= ARCHIVE_KEEP,
            "kept {count}, expected <= {ARCHIVE_KEEP}"
        );
        assert!(count > 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn truncate_handles_multibyte_boundary() {
        let (s, t) = truncate_chars("你好世界再见", 3);
        assert!(t);
        assert!(s.starts_with("你好世"));
        assert!(s.contains("6 chars total"));
        // byte-safe: no panic, valid UTF-8
        assert!(s.is_char_boundary(s.len()));
    }

    #[test]
    fn clean_text_keeps_nonconsecutive_duplicates() {
        assert_eq!(clean_text("a\nb\na"), "a\nb\na");
        assert_eq!(clean_text("a\na\na"), "a");
    }

    #[test]
    fn tool_array_accepts_wrapper_shape() {
        let wrapped = json!({"tools": [{"name": "navigate"}, {"name": "click"}]});
        assert_eq!(tool_array(&wrapped).len(), 2);
        assert_eq!(tool_names(&wrapped), json!(["navigate", "click"]));
        assert_eq!(tool_names(&json!([])), json!([]));
    }

    #[test]
    fn sanitize_tool_name() {
        assert_eq!(sanitize("get_page_text"), "get_page_text");
        assert_eq!(sanitize("a/b c-d"), "a_b_c_d");
    }

    #[test]
    fn arg_helpers_reject_garbage() {
        assert!(!arg_bool(Some(&json!("nope"))));
        assert!(!arg_bool(Some(&json!(null))));
        assert_eq!(arg_usize(Some(&json!("abc"))), 0);
        assert_eq!(arg_usize(Some(&json!(null))), 0);
    }

    #[test]
    fn compact_result_redacts_base64_to_artifact() {
        use base64::Engine;
        let dir = tmp_dir("b64");
        // Screenshot-shaped result: a big base64 payload must never reach the
        // preview — it becomes an artifact ref.
        let raws: Vec<u8> = (0..20_000u32).map(|i| (i % 256) as u8).collect();
        let b64 = base64::engine::general_purpose::STANDARD.encode(&raws);
        let result = json!({"width": 400, "height": 800, "format": "rgba", "base64": b64});
        let s = compact_result(&dir, "screenshot", result, 0);
        let v: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["artifacts"], 1, "s={s}");
        assert!(s.contains(".aacode/artifacts/"), "s={s}");
        assert!(!s.contains(&"A".repeat(200)), "bulk base64 leaked");
        assert!(s.len() < 4_000, "preview too big: {}", s.len());
    }
}
