// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Observation sanitisation — the single choke point between a tool's raw
//! output and the model/session.
//!
//! Root invariant: **message content never contains an unbounded binary
//! payload.** Any binary blob (inline `base64`, a `data:` URI, a high-entropy
//! long string, or a field whose key marks it as binary) is written to an
//! [`ArtifactStore`] and replaced in-place by a small [`ArtifactRef`] plus a
//! *bounded, labelled* hex preview.
//!
//! This deliberately lives in aacode-rs (tool layer) and knows nothing about
//! fastshell / fastbrowser internals — it operates purely on the tool's
//! returned string/JSON.

use crate::artifacts::{hex_preview, ArtifactStore};
use base64::Engine;
use serde_json::{json, Map, Value};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Tunables for [`sanitize`]. All have conservative defaults.
#[derive(Debug, Clone)]
pub struct SanitizeLimits {
    /// Hard cap on the final observation string handed to the model.
    pub max_observation_chars: usize,
    /// Hard cap on preview bytes per artifact (regardless of what a tool wants).
    pub preview_bytes: usize,
    /// A string at least this long that is pure base64 is treated as binary.
    pub min_base64_len: usize,
    /// Per-task cumulative budget for preview hex (chars). Stops a peek loop
    /// from exfiltrating a whole file through many small samples.
    pub budget_chars: usize,
}

impl Default for SanitizeLimits {
    fn default() -> Self {
        SanitizeLimits {
            // 0 = do NOT truncate plain text; only binary/base64 is externalized.
            // (`run_shell` length is the caller's choice via `max_output`.)
            max_observation_chars: 0,
            preview_bytes: 64,
            min_base64_len: 256,
            budget_chars: 16_384,
        }
    }
}

/// A tiny per-task budget for preview bytes (shared across tool calls).
#[derive(Debug)]
pub struct PreviewBudget(AtomicUsize);

impl PreviewBudget {
    pub fn new(chars: usize) -> Self {
        PreviewBudget(AtomicUsize::new(chars))
    }
    pub fn default_for(limits: &SanitizeLimits) -> Self {
        PreviewBudget::new(limits.budget_chars)
    }
    /// Reserve `want` chars of budget; returns how many were granted.
    fn take(&self, want: usize) -> usize {
        self.0
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                Some(cur.saturating_sub(want))
            })
            .map(|prev| want.min(prev))
            .unwrap_or(0)
    }
}

fn is_base64_alphabet(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '=' || c == '-' || c == '_'
}

/// Heuristic: is `s` a single base64 blob (ignoring whitespace/newlines)?
pub fn is_probably_base64(s: &str, min_len: usize) -> bool {
    let t = s.trim();
    if t.len() < min_len {
        return false;
    }
    if t.contains(' ') {
        return false;
    }
    let core: String = t.chars().filter(|c| !c.is_whitespace()).collect();
    if core.len() < min_len || core.len() % 4 != 0 {
        return false;
    }
    core.chars().all(is_base64_alphabet)
}

fn unique_chars(s: &str) -> usize {
    let mut set = std::collections::HashSet::new();
    for c in s.chars() {
        if !c.is_whitespace() {
            set.insert(c);
        }
    }
    set.len()
}

/// Stricter, non-keyed backstop: a long base64-looking blob that also carries
/// base64 punctuation and high character diversity. Avoids mangling ordinary
/// repeated/plain text (e.g. a long run of `x`) into an artifact.
fn looks_like_b64_blob(s: &str, min_len: usize) -> bool {
    is_probably_base64(s, min_len)
        && (s.contains('+') || s.contains('/') || s.contains('='))
        && unique_chars(s) >= 12
}

fn key_is_binary(k: &str) -> bool {
    let k = k.to_ascii_lowercase();
    k == "base64"
        || k == "b64"
        || k == "data"
        || k == "blob"
        || k == "image"
        || k == "audio"
        || k == "video"
        || k.ends_with("_base64")
        || k.ends_with("_b64")
        || k.contains("base64")
}

fn decode_data_uri(s: &str) -> Option<(String, Vec<u8>)> {
    let rest = s.strip_prefix("data:")?;
    let (meta, b64) = rest.split_once(',')?;
    let mime = meta.split(';').next().unwrap_or("").to_string();
    if !meta.contains("base64") {
        // Percent-encoded data URIs are rare here; only handle base64.
        return None;
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .or_else(|_| base64::engine::general_purpose::URL_SAFE.decode(b64.trim()))
        .ok()?;
    Some((
        if mime.is_empty() {
            "application/octet-stream".to_string()
        } else {
            mime
        },
        bytes,
    ))
}

fn looks_binary(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }
    if bytes.contains(&0) {
        return true;
    }
    let nontext = bytes
        .iter()
        .filter(|&&b| b < 0x09 || (b > 0x0d && b < 0x20) || b >= 0x7f)
        .count();
    nontext * 100 / bytes.len() >= 10
}

fn take_preview(
    store: &ArtifactStore,
    budget: &PreviewBudget,
    limits: &SanitizeLimits,
    bytes: &[u8],
) -> Option<String> {
    let (kind, _) = ArtifactStore::sniff(bytes);
    let want_bytes = kind
        .default_preview_bytes()
        .min(limits.preview_bytes)
        .max(1);
    let want_chars = want_bytes * 3; // "xx " per byte
    let granted = budget.take(want_chars);
    if granted == 0 {
        return None;
    }
    let grant_bytes = (granted / 3).max(1);
    Some(hex_preview(bytes, grant_bytes))
}

/// Turn one binary blob into `{artifact, mime, bytes, kind, preview_hex…}`.
fn blob_to_ref_json(
    store: &ArtifactStore,
    budget: &PreviewBudget,
    limits: &SanitizeLimits,
    bytes: &[u8],
    mime_hint: Option<&str>,
) -> Value {
    match store.put_bytes(bytes, mime_hint, "obs") {
        Ok(r) => {
            let preview = take_preview(store, budget, limits, bytes);
            r.to_model_json(preview.as_deref())
        }
        Err(_) => {
            // Could not persist (disk error) → degrade to a bounded sample only.
            let preview = hex_preview(bytes, limits.preview_bytes);
            json!({"binary_omitted": true, "bytes": bytes.len(), "preview_hex": preview})
        }
    }
}

/// Handle a single string value that may embed binary.
fn sanitize_string_value(
    s: &str,
    key: Option<&str>,
    store: &ArtifactStore,
    budget: &PreviewBudget,
    limits: &SanitizeLimits,
) -> Value {
    if let Some((mime, bytes)) = decode_data_uri(s) {
        if !bytes.is_empty() {
            return blob_to_ref_json(store, budget, limits, &bytes, Some(&mime));
        }
    }
    let key_binary = key.map(key_is_binary).unwrap_or(false);
    let maybe_b64 = key_binary || looks_like_b64_blob(s, limits.min_base64_len);
    if maybe_b64 {
        let core: String = s.chars().filter(|c| !c.is_whitespace()).collect();
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(core.as_bytes())
            .or_else(|_| base64::engine::general_purpose::URL_SAFE.decode(core.as_bytes()))
            .ok();
        if let Some(bytes) = decoded {
            // For a non-keyed match, only treat it as binary if it really looks
            // binary (avoid mangling ordinary long text that happens to decode).
            if key_binary || looks_binary(&bytes) {
                return blob_to_ref_json(store, budget, limits, &bytes, None);
            }
        }
    }
    Value::String(s.to_string())
}

fn sanitize_value(
    v: &Value,
    key: Option<&str>,
    store: &ArtifactStore,
    budget: &PreviewBudget,
    limits: &SanitizeLimits,
) -> Value {
    match v {
        Value::String(s) => sanitize_string_value(s, key, store, budget, limits),
        Value::Array(a) => Value::Array(
            a.iter()
                .map(|x| sanitize_value(x, None, store, budget, limits))
                .collect(),
        ),
        Value::Object(o) => {
            let mut m = Map::new();
            for (k, val) in o {
                m.insert(
                    k.clone(),
                    sanitize_value(val, Some(k), store, budget, limits),
                );
            }
            Value::Object(m)
        }
        other => other.clone(),
    }
}

fn cap_text(s: String, max: usize) -> String {
    if max == 0 || s.chars().count() <= max {
        return s;
    }
    let head: String = s.chars().take(max).collect();
    format!("{head}\n…[truncated, {} chars total]", s.chars().count())
}

/// Sanitize a raw tool observation. Idempotent and safe for any tool.
pub fn sanitize(
    raw: &str,
    store: &ArtifactStore,
    budget: &PreviewBudget,
    limits: &SanitizeLimits,
) -> String {
    // JSON path: walk and replace embedded blobs.
    if let Ok(v) = serde_json::from_str::<Value>(raw) {
        // Fast exit: a plain short JSON without any binary-ish token.
        let has_token = raw.contains("base64")
            || raw.contains("data:")
            || raw.contains("b64")
            || raw.len() > limits.min_base64_len;
        if !has_token {
            return cap_text(raw.to_string(), limits.max_observation_chars);
        }
        let out = sanitize_value(&v, None, store, budget, limits);
        // No substitution happened → return the ORIGINAL text verbatim so we
        // don't needlessly re-serialize (compact / reorder keys) every result.
        if out == v {
            return cap_text(raw.to_string(), limits.max_observation_chars);
        }
        let s = serde_json::to_string(&out).unwrap_or_else(|_| raw.to_string());
        return cap_text(s, limits.max_observation_chars);
    }

    // Plain-text path: a bare base64 blob / data URI / binary-ish text.
    if let Some((mime, bytes)) = decode_data_uri(raw.trim()) {
        if !bytes.is_empty() {
            let j = blob_to_ref_json(store, budget, limits, &bytes, Some(&mime));
            return j.to_string();
        }
    }
    let trimmed = raw.trim();
    if looks_like_b64_blob(trimmed, limits.min_base64_len) {
        let core: String = trimmed.chars().filter(|c| !c.is_whitespace()).collect();
        if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(core.as_bytes()) {
            if looks_binary(&bytes) {
                let j = blob_to_ref_json(store, budget, limits, &bytes, None);
                return j.to_string();
            }
        }
    }
    cap_text(raw.to_string(), limits.max_observation_chars)
}

fn count_artifacts(v: &Value) -> usize {
    match v {
        Value::Object(m) => {
            let here = usize::from(m.contains_key("artifact"));
            here + m.values().map(count_artifacts).sum::<usize>()
        }
        Value::Array(a) => a.iter().map(count_artifacts).sum(),
        _ => 0,
    }
}

/// Replace every binary-ish string in `value` with an artifact-ref object and
/// return `(redacted, artifact_count)`. No preview is emitted (budget 0), so
/// this is safe to run *before* a preview/truncation step (e.g. browser compact
/// mode) — the raw payload is archived, not truncated away.
pub fn redact_binary(value: &Value, store: &ArtifactStore) -> (Value, usize) {
    let budget = PreviewBudget::new(0);
    let limits = SanitizeLimits::default();
    let out = sanitize_value(value, None, store, &budget, &limits);
    let n = count_artifacts(&out);
    (out, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn store() -> (ArtifactStore, PathBuf) {
        let dir = std::env::temp_dir().join(format!("aacode_obs_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        (ArtifactStore::new(&dir), dir)
    }

    fn lim() -> SanitizeLimits {
        SanitizeLimits::default()
    }

    #[test]
    fn inline_base64_field_becomes_artifact_ref() {
        let (st, _d) = store();
        let budget = PreviewBudget::new(lim().budget_chars);
        // 400x800 raw RGBA-ish payload (binary, not PNG) encoded as base64.
        let raws: Vec<u8> = (0..100_000u32).map(|i| (i * 7 % 256) as u8).collect();
        let b64 = base64::engine::general_purpose::STANDARD.encode(&raws);
        let raw = json!({"result": {"base64": b64, "format": "rgba", "width": 400, "height": 800}, "success": true}).to_string();
        let out = sanitize(&raw, &st, &budget, &lim());
        assert!(out.len() < 2_000, "sanitized len={}", out.len());
        assert!(out.contains("artifact"), "out={out}");
        assert!(out.contains("preview_hex"), "out={out}");
        assert!(!out.contains(&"A".repeat(200)), "no bulk base64 remains");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["result"]["base64"]["bytes"], 100_000);
    }

    #[test]
    fn data_uri_image_is_detected_and_extracted() {
        let (st, _d) = store();
        let budget = PreviewBudget::new(lim().budget_chars);
        let png: Vec<u8> = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 1, 2, 3, 4];
        let uri = format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&png)
        );
        let raw = json!({"img": uri}).to_string();
        let out = sanitize(&raw, &st, &budget, &lim());
        assert!(out.contains("image/png"), "out={out}");
        assert!(out.contains("artifact"), "out={out}");
    }

    #[test]
    fn budget_limits_repeated_previews() {
        let (st, _d) = store();
        let mut l = lim();
        l.preview_bytes = 64;
        let budget = PreviewBudget::new(9); // only 3 bytes of preview total
        let blob = base64::engine::general_purpose::STANDARD.encode(vec![0u8; 4096]);
        let raw = json!({"a": {"base64": blob.clone()}, "b": {"base64": blob}}).to_string();
        let out = sanitize(&raw, &st, &budget, &l);
        // Second artifact reuses the exhausted budget → no preview.
        let count = out.matches("preview_hex").count();
        assert!(count <= 1, "preview budget not enforced: {out}");
    }

    #[test]
    fn plain_text_not_truncated_by_default() {
        let (st, _d) = store();
        let budget = PreviewBudget::new(lim().budget_chars);
        let raw = "line of text\n".repeat(20_000); // ~280k chars
        let out = sanitize(&raw, &st, &budget, &lim());
        assert_eq!(
            out.len(),
            raw.len(),
            "plain text must pass through unbounded"
        );
    }

    #[test]
    fn plain_text_is_only_capped() {
        let (st, _d) = store();
        let budget = PreviewBudget::new(lim().budget_chars);
        let raw = "hello world\n".repeat(10);
        let out = sanitize(&raw, &st, &budget, &lim());
        assert!(out.starts_with("hello world"));
    }

    #[test]
    fn long_text_is_truncated() {
        let (st, _d) = store();
        let budget = PreviewBudget::new(lim().budget_chars);
        let mut l = lim();
        l.max_observation_chars = 100;
        let raw = "x".repeat(5000);
        let out = sanitize(&raw, &st, &budget, &l);
        assert!(out.len() < 200, "len={}", out.len());
        assert!(out.contains("truncated"));
    }
}
