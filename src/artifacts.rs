// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Artifacts — the only place binary payloads live.
//!
//! Design invariant (see `observation.rs`): message content that reaches the
//! model or the session must never contain an unbounded binary payload. Binary
//! data is always written to an [`ArtifactStore`] file and represented in the
//! conversation by a small [`ArtifactRef`] (path + mime + size + optional
//! bounded preview). This module is self-contained and does **not** depend on
//! fastshell or fastbrowser.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::Cursor;
use std::path::{Path, PathBuf};

/// Keep at most this many artifact files on disk.
pub const ARTIFACT_KEEP: usize = 50;

/// Coarse type of an artifact (drives default preview policy).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ArtifactKind {
    Image,
    Pdf,
    Audio,
    Video,
    Archive,
    Binary,
    Text,
}

impl ArtifactKind {
    /// Default number of leading bytes shown as hex when this artifact is
    /// first surfaced (images only need the magic header; unknown binaries a
    /// bit more for eyeballing).
    pub fn default_preview_bytes(self) -> usize {
        match self {
            ArtifactKind::Image => 16,
            ArtifactKind::Pdf
            | ArtifactKind::Archive
            | ArtifactKind::Audio
            | ArtifactKind::Video
            | ArtifactKind::Binary => 64,
            ArtifactKind::Text => 0,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArtifactMeta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
}

impl ArtifactMeta {
    pub fn is_empty(&self) -> bool {
        self.width.is_none() && self.height.is_none()
    }
}

/// Reference to a stored artifact. This is the *only* binary representation
/// allowed inside messages/sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub id: String,
    /// Project-relative path (e.g. `.aacode/artifacts/<id>.png`).
    pub path: String,
    pub mime: String,
    pub bytes: usize,
    pub kind: ArtifactKind,
    #[serde(default, skip_serializing_if = "ArtifactMeta::is_empty")]
    pub meta: ArtifactMeta,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub sha256: String,
}

impl ArtifactRef {
    /// Small, model-facing JSON: metadata + path + an optional bounded hex
    /// preview. Never contains the full payload.
    pub fn to_model_json(&self, preview_hex: Option<&str>) -> Value {
        let mut v = json!({
            "artifact": self.path,
            "mime": self.mime,
            "bytes": self.bytes,
            "kind": self.kind,
        });
        if let Some(w) = self.meta.width {
            v["width"] = json!(w);
        }
        if let Some(h) = self.meta.height {
            v["height"] = json!(h);
        }
        if !self.sha256.is_empty() {
            v["sha256"] = json!(self.sha256);
        }
        if let Some(p) = preview_hex {
            v["preview_hex"] = json!(p);
            v["preview_truncated"] = json!(self.bytes > p.len() / 2);
        }
        v
    }
}

/// A directory-backed artifact store rooted at `<project>/.aacode/artifacts/`.
#[derive(Debug, Clone)]
pub struct ArtifactStore {
    root: PathBuf,
    rel_root: String,
}

impl ArtifactStore {
    pub fn new(project_path: &Path) -> Self {
        ArtifactStore {
            root: project_path.join(".aacode").join("artifacts"),
            rel_root: ".aacode/artifacts".to_string(),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Sniff magic bytes → (kind, mime). Falls back to `Binary`/octet-stream.
    pub fn sniff(bytes: &[u8]) -> (ArtifactKind, &'static str) {
        let b = bytes;
        let starts = |m: &[u8]| b.len() >= m.len() && &b[..m.len()] == m;
        if starts(b"\x89PNG\r\n\x1a\n") {
            return (ArtifactKind::Image, "image/png");
        }
        if starts(b"\xff\xd8\xff") {
            return (ArtifactKind::Image, "image/jpeg");
        }
        if starts(b"GIF87a") || starts(b"GIF89a") {
            return (ArtifactKind::Image, "image/gif");
        }
        if b.len() >= 12 && starts(b"RIFF") && &b[8..12] == b"WEBP" {
            return (ArtifactKind::Image, "image/webp");
        }
        if starts(b"BM") {
            return (ArtifactKind::Image, "image/bmp");
        }
        if starts(b"%PDF") {
            return (ArtifactKind::Pdf, "application/pdf");
        }
        if starts(b"PK\x03\x04") || starts(b"PK\x05\x06") {
            return (ArtifactKind::Archive, "application/zip");
        }
        if starts(b"\x1f\x8b") {
            return (ArtifactKind::Archive, "application/gzip");
        }
        if starts(b"ID3") || starts(b"\xff\xfb") {
            return (ArtifactKind::Audio, "audio/mpeg");
        }
        if b.len() >= 12 && &b[4..8] == b"ftyp" {
            return (ArtifactKind::Video, "video/mp4");
        }
        if starts(b"OggS") {
            return (ArtifactKind::Audio, "audio/ogg");
        }
        (ArtifactKind::Binary, "application/octet-stream")
    }

    fn extension_for(kind: ArtifactKind, mime: &str) -> &'static str {
        match mime {
            "image/png" => "png",
            "image/jpeg" => "jpg",
            "image/gif" => "gif",
            "image/webp" => "webp",
            "image/bmp" => "bmp",
            "application/pdf" => "pdf",
            "application/zip" => "zip",
            "application/gzip" => "gz",
            "audio/mpeg" => "mp3",
            "audio/ogg" => "ogg",
            "video/mp4" => "mp4",
            _ => match kind {
                ArtifactKind::Image => "img",
                ArtifactKind::Pdf => "pdf",
                ArtifactKind::Audio => "audio",
                ArtifactKind::Video => "video",
                ArtifactKind::Archive => "bin",
                _ => "bin",
            },
        }
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(bytes);
        h.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }

    fn image_dims(bytes: &[u8]) -> ArtifactMeta {
        let dims = image::ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .ok()
            .and_then(|r| r.into_dimensions().ok());
        match dims {
            Some((w, h)) => ArtifactMeta {
                width: Some(w),
                height: Some(h),
            },
            None => ArtifactMeta::default(),
        }
    }

    /// Write `bytes` to a new artifact file and return its reference.
    /// `mime_hint` (e.g. from a response header) is honored when it looks sane,
    /// otherwise magic sniffing wins.
    pub fn put_bytes(
        &self,
        bytes: &[u8],
        mime_hint: Option<&str>,
        _name_hint: &str,
    ) -> std::io::Result<ArtifactRef> {
        let (sniffed_kind, sniffed_mime) = Self::sniff(bytes);
        let (kind, mime) = match mime_hint {
            Some(m) if !m.is_empty() && m != "application/octet-stream" => (sniffed_kind, m),
            _ => (sniffed_kind, sniffed_mime),
        };
        let id = uuid::Uuid::new_v4().simple().to_string();
        let ext = Self::extension_for(kind, mime);
        std::fs::create_dir_all(&self.root)?;
        let file = format!("{id}.{ext}");
        std::fs::write(self.root.join(&file), bytes)?;
        let meta = match kind {
            ArtifactKind::Image => Self::image_dims(bytes),
            _ => ArtifactMeta::default(),
        };
        let _ = self.prune(ARTIFACT_KEEP);
        Ok(ArtifactRef {
            id,
            path: format!("{}/{}", self.rel_root, file),
            mime: mime.to_string(),
            bytes: bytes.len(),
            kind,
            meta,
            sha256: Self::sha256_hex(bytes),
        })
    }

    /// Resolve an id or a path (relative or absolute) to a concrete file.
    pub fn path_for(&self, id_or_path: &str) -> Option<PathBuf> {
        let p = PathBuf::from(id_or_path);
        for cand in [p.clone(), self.root.join(id_or_path)] {
            if cand.is_file() {
                return Some(cand);
            }
        }
        // Bare id (no extension): find first matching `<id>.*`.
        if let Ok(rd) = std::fs::read_dir(&self.root) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if name == id_or_path || name.starts_with(&format!("{id_or_path}.")) {
                    return Some(e.path());
                }
            }
        }
        None
    }

    pub fn read(&self, id_or_path: &str) -> Option<Vec<u8>> {
        let p = self.path_for(id_or_path)?;
        std::fs::read(p).ok()
    }

    /// Keep only the most recent `keep` artifact files.
    pub fn prune(&self, keep: usize) -> std::io::Result<()> {
        let mut files: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
        for e in std::fs::read_dir(&self.root)?.flatten() {
            if let Ok(md) = e.metadata() {
                let t = md.modified().unwrap_or(std::time::UNIX_EPOCH);
                files.push((t, e.path()));
            }
        }
        files.sort_by_key(|(t, _)| *t);
        if files.len() > keep {
            for (_, p) in files.iter().take(files.len() - keep) {
                let _ = std::fs::remove_file(p);
            }
        }
        Ok(())
    }
}

/// Render the first `n` bytes as a lowercase hex string with spaces, e.g.
/// `89 50 4e 47`. Always bounded by `n`.
pub fn hex_preview(bytes: &[u8], n: usize) -> String {
    bytes
        .iter()
        .take(n)
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> (ArtifactStore, PathBuf) {
        let dir = std::env::temp_dir().join(format!("aacode_art_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        (ArtifactStore::new(&dir), dir)
    }

    #[test]
    fn sniffs_png_and_records_dims() {
        // Complete 1x1 PNG (so the image crate can read its dimensions).
        use base64::Engine;
        let png: Vec<u8> = base64::engine::general_purpose::STANDARD
            .decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            )
            .unwrap();
        let (kind, mime) = ArtifactStore::sniff(&png);
        assert_eq!(kind, ArtifactKind::Image);
        assert_eq!(mime, "image/png");
        let (store, _d) = tmp();
        let r = store.put_bytes(&png, None, "shot").unwrap();
        assert_eq!(r.kind, ArtifactKind::Image);
        assert_eq!(r.meta.width, Some(1));
        assert!(r.path.ends_with(".png"));
        assert_eq!(store.read(&r.id).unwrap(), png);
    }

    #[test]
    fn hex_preview_is_bounded() {
        let b: Vec<u8> = (0..100).collect();
        let p = hex_preview(&b, 4);
        assert_eq!(p, "00 01 02 03");
    }
}
