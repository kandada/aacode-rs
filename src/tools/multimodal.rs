// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Multimodal tools — understand_image / understand_video / understand_ui_design
//! / analyze_image_consistency.
//!
//! Supports both OpenAI and Anthropic vision formats based on the configured
//! model gateway. Uses `reqwest` async HTTP for vision API calls.

use super::registry::Tool;
use super::schema::{ParamType, ToolParameter, ToolSchema};
use crate::config::{Gateway, ModelConfig};
use crate::error::Result;
use base64::Engine;
use image::{GenericImageView, ImageEncoder};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Connect timeout for HTTP client (seconds). Separate from the total request
/// timeout so DNS/TCP hang is detected quickly rather than waiting for the
/// full request deadline.
const CONNECT_TIMEOUT_SECS: u64 = 10;

/// Maximum short-side pixel dimension before an image is downscaled.
/// Most vision models (GPT-4V, Claude, Kimi) recommend ≤ 1024 px for
/// cost/performance; sticking to this keeps uploads fast on mobile.
const MAX_IMAGE_DIMENSION: u32 = 1024;

/// JPEG quality for resized photos (trade-off between size and visual fidelity).
const JPEG_QUALITY: u8 = 80;

/// Downsize and re-encode an image if it exceeds `max_dim` on its short side.
/// Photos (JPEG / WebP / HEIC) are re-encoded as JPEG; screenshots / UI mocks
/// (PNG / BMP) stay as PNG to preserve text sharpness.
///
/// Never panics — on any failure the original bytes and MIME are returned
/// unchanged so downstream processing is not affected.
fn prepare_image(bytes: &[u8], ext: &str, max_dim: u32) -> (Vec<u8>, &'static str) {
    let original_mime = ext_to_mime(ext);

    // GIF — don't touch; frames would be lost, and GIFs are typically small.
    if ext == "gif" {
        return (bytes.to_vec(), original_mime);
    }

    let img = match image::load_from_memory(bytes) {
        Ok(v) => v,
        Err(_) => return (bytes.to_vec(), original_mime),
    };

    let (w, h) = img.dimensions();
    let short = w.min(h);
    if short <= max_dim {
        return (bytes.to_vec(), original_mime);
    }

    // Scale so the short side equals max_dim, preserving aspect ratio.
    let ratio = max_dim as f64 / short as f64;
    let new_w = (w as f64 * ratio).round() as u32;
    let new_h = (h as f64 * ratio).round() as u32;
    let resized = img.resize_exact(new_w, new_h, image::imageops::FilterType::Lanczos3);

    let is_photo = matches!(ext, "jpg" | "jpeg" | "webp" | "heic" | "heif");
    if is_photo {
        let rgb = resized.to_rgb8();
        let mut buf = Vec::new();
        let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, JPEG_QUALITY);
        if encoder.write_image(&rgb, rgb.width(), rgb.height(), image::ExtendedColorType::Rgb8).is_ok() {
            return (buf, "image/jpeg");
        }
    } else {
        let rgba = resized.to_rgba8();
        let mut buf = Vec::new();
        let encoder = image::codecs::png::PngEncoder::new(&mut buf);
        if encoder.write_image(&rgba, rgba.width(), rgba.height(), image::ExtendedColorType::Rgba8).is_ok() {
            return (buf, "image/png");
        }
    }

    // Encoding failed — return originals.
    (bytes.to_vec(), original_mime)
}

fn ext_to_mime(ext: &str) -> &'static str {
    match ext {
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        _ => "image/png",
    }
}

/// Shared config for the multimodal tools.
#[derive(Clone)]
pub struct MultimodalCtx {
    pub model: Option<ModelConfig>,
    pub project_path: PathBuf,
    pub timeout_secs: u64,
    /// Lazily-initialised shared HTTP client.
    client: Arc<Mutex<Option<reqwest::Client>>>,
}

impl MultimodalCtx {
    pub fn new(model: Option<ModelConfig>, project_path: PathBuf, timeout_secs: u64) -> Self {
        MultimodalCtx {
            model,
            project_path,
            timeout_secs,
            client: Arc::new(Mutex::new(None)),
        }
    }
    fn resolve_path(&self, p: &str) -> PathBuf {
        let path = PathBuf::from(p);
        if path.is_absolute() { path } else { self.project_path.join(p) }
    }

    async fn image_data_uri(&self, path: &str) -> Option<String> {
        let full = self.resolve_path(path);
        let ext = full.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_else(|| "png".into());
        let bytes = tokio::task::spawn_blocking(move || std::fs::read(&full))
            .await
            .ok()?
            .ok()?;
        let (img_bytes, mime) = prepare_image(&bytes, &ext, MAX_IMAGE_DIMENSION);
        let b64 = base64::engine::general_purpose::STANDARD.encode(&img_bytes);
        Some(format!("data:{mime};base64,{b64}"))
    }

    async fn vision_chat(&self, prompt: &str, image_paths: &[&str]) -> Result<String> {
        let model = self.model.as_ref()
            .ok_or_else(|| crate::error::AacodeError::Config("no multimodal model configured".into()))?;
        let is_anthropic = model.gateway == Gateway::Anthropic;
        let timeout_secs = model.request_timeout_secs.unwrap_or(self.timeout_secs);

        let endpoint = if is_anthropic {
            let base = model.resolved_base_url();
            if base.contains("minimax") || base.contains("deepseek") || base.contains("moonshot") {
                format!("{}/anthropic/v1/messages", base.trim_end_matches('/'))
            } else {
                format!("{}/v1/messages", base.trim_end_matches('/'))
            }
        } else {
            format!("{}/chat/completions", model.resolved_base_url().trim_end_matches('/'))
        };

        let (body, auth_name, auth_val, anthropic_ver) = if is_anthropic {
            self.build_anthropic_body(model, prompt, image_paths).await?
        } else {
            self.build_openai_body(model, prompt, image_paths).await?
        };

        async fn do_vision(
                client: &reqwest::Client, endpoint: &str, body: &Value,
                auth_name: &str, auth_val: &str, anthropic_ver: &str, is_anthropic: bool,
            ) -> Result<String> {
                let mut req = client.post(endpoint)
                    .header(auth_name, auth_val)
                    .header("Content-Type", "application/json");
                if !anthropic_ver.is_empty() {
                    req = req.header("anthropic-version", anthropic_ver);
                }
                let resp = req.json(body).send().await
                    .map_err(|e| crate::error::AacodeError::Network(format!("vision: {}", error_chain_display(&e))))?;
                if !resp.status().is_success() {
                    let code = resp.status().as_u16();
                    let msg = resp.text().await.unwrap_or_default();
                    return Err(crate::error::AacodeError::Api(format!("vision HTTP {code}: {}", truncate(&msg, 300))));
                }
                let v: Value = resp.json().await
                    .map_err(|e| crate::error::AacodeError::Network(format!("vision response: {}", error_chain_display(&e))))?;
                Ok(extract_text(&v, is_anthropic))
            }

            let client = self.get_or_init_http_client(timeout_secs)?;
            do_vision(&client, &endpoint, &body, &auth_name, &auth_val, &anthropic_ver, is_anthropic).await
    }

    async fn build_openai_body(&self, model: &ModelConfig, prompt: &str, image_paths: &[&str]) -> Result<(Value, String, String, String)> {
        let api_key = model.api_key.clone().filter(|k| !k.trim().is_empty())
            .ok_or_else(|| crate::error::AacodeError::Config("multimodal api key missing".into()))?;
        let mut parts: Vec<Value> = vec![json!({"type":"text","text":prompt})];
        for p in image_paths {
            match self.image_data_uri(p).await {
                Some(uri) => parts.push(json!({"type":"image_url","image_url":{"url":uri}})),
                None => return Err(crate::error::AacodeError::Io(format!("cannot read image: {p}"))),
            }
        }
        let body = json!({"model":model.name,"messages":[{"role":"user","content":parts}],"max_tokens":model.max_tokens});
        Ok((body, "Authorization".into(), format!("Bearer {api_key}"), String::new()))
    }

    async fn build_anthropic_body(&self, model: &ModelConfig, prompt: &str, image_paths: &[&str]) -> Result<(Value, String, String, String)> {
        let api_key = model.api_key.clone().filter(|k| !k.trim().is_empty())
            .ok_or_else(|| crate::error::AacodeError::Config("multimodal api key missing".into()))?;
        let mut blocks: Vec<Value> = Vec::new();
        for p in image_paths {
            let uri = self.image_data_uri(p).await
                .ok_or_else(|| crate::error::AacodeError::Io(format!("cannot read image: {p}")))?;
            let (media_type, b64) = parse_data_uri(&uri);
            blocks.push(json!({"type":"image","source":{"type":"base64","media_type":media_type,"data":b64}}));
        }
        blocks.push(json!({"type":"text","text":prompt}));
        let body = json!({"model":model.name,"max_tokens":model.max_tokens,"messages":[{"role":"user","content":blocks}]});
        Ok((body, "x-api-key".into(), api_key, "2023-06-01".into()))
    }
    fn get_or_init_http_client(&self, timeout_secs: u64) -> Result<reqwest::Client> {
        let mut guard = self.client.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(c) = guard.as_ref() {
            return Ok(c.clone());
        }
        let c = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(CONNECT_TIMEOUT_SECS))
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .map_err(|e| crate::error::AacodeError::Network(e.to_string()))?;
        *guard = Some(c.clone());
        Ok(c)
    }
}

/// Walk an error's source chain for full diagnostics (e.g. reqwest errors
/// whose Display only says "error sending request for url" while the real
/// cause — timeout / DNS / TLS — is buried in the source).
fn error_chain_display(e: &dyn std::error::Error) -> String {
    let mut s = e.to_string();
    let mut source = e.source();
    while let Some(src) = source {
        s.push_str(": ");
        s.push_str(&src.to_string());
        source = src.source();
    }
    s
}

fn parse_data_uri(uri: &str) -> (String, String) {
    if let Some(rest) = uri.strip_prefix("data:") {
        let parts: Vec<&str> = rest.splitn(2, ';').collect();
        let media = parts.first().map(|s| s.to_string()).unwrap_or_else(|| "image/png".into());
        let b64 = parts.last().map(|s| s.strip_prefix("base64,").unwrap_or(s)).unwrap_or(uri).to_string();
        (media, b64)
    } else {
        ("image/png".into(), uri.to_string())
    }
}

fn extract_text(v: &Value, is_anthropic: bool) -> String {
    if is_anthropic {
        v["content"].as_array().and_then(|arr|
            arr.iter().find_map(|b| b["text"].as_str().map(|s| s.to_string()))
        ).unwrap_or_default()
    } else {
        v["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string()
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n { s.to_string() } else { let h: String = s.chars().take(n).collect(); format!("{h}...") }
}

fn split_paths(s: &str) -> Vec<&str> {
    s.split(',').map(|p| p.trim()).filter(|p| !p.is_empty()).collect()
}

pub struct UnderstandImageTool { pub ctx: MultimodalCtx }
#[async_trait::async_trait]
impl Tool for UnderstandImageTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("understand_image","Understand image content (one or more, comma-separated). Analyze screenshots, photos, design drafts.",
            vec![
                ToolParameter::new("image_path",ParamType::String,true,"Image path(s), comma-separated",&["image","path","file"]),
                ToolParameter::new("prompt",ParamType::String,false,"What to ask about the image",&["question","query"]),
            ],
        )
    }
    async fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let paths_s = args.get("image_path").and_then(|v| v.as_str()).unwrap_or("");
        let prompt = args.get("prompt").and_then(|v| v.as_str()).unwrap_or("Describe this image in detail.");
        let paths = split_paths(paths_s);
        if paths.is_empty() { return Ok(json!({"success":false,"error":"no image_path"}).to_string()); }
        match self.ctx.vision_chat(prompt, &paths).await {
            Ok(desc) => Ok(json!({"success":true,"description":desc,"images_count":paths.len()}).to_string()),
            Err(e) => Ok(json!({"success":false,"error":e.to_string()}).to_string()),
        }
    }
}

pub struct UnderstandVideoTool { pub ctx: MultimodalCtx }
#[async_trait::async_trait]
impl Tool for UnderstandVideoTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("understand_video","Video understanding is not supported yet.",
            vec![
                ToolParameter::new("video_path",ParamType::String,true,"Video file path",&["video","path","file"]),
                ToolParameter::new("prompt",ParamType::String,false,"What to ask about the video",&["question","query"]),
            ],
        )
    }
    async fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let path = args.get("video_path").and_then(|v| v.as_str()).unwrap_or("");
        if path.is_empty() { return Ok(json!({"success":false,"error":"no video_path"}).to_string()); }
        Ok(json!({"success":false,"error":"video understanding is not supported yet."}).to_string())
    }
}

pub struct UnderstandUiDesignTool { pub ctx: MultimodalCtx }
#[async_trait::async_trait]
impl Tool for UnderstandUiDesignTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("understand_ui_design","Analyze UI design mockups/screenshots and generate frontend code.",
            vec![
                ToolParameter::new("design_path",ParamType::String,true,"Path to the design image file",&["design","image","path"]),
                ToolParameter::new("prompt",ParamType::String,false,"What to focus on (e.g., 'generate HTML/CSS')",&["question","focus"]),
                ToolParameter::new("generate_code",ParamType::Boolean,false,"Whether to generate frontend code",&["code","html"]),
            ],
        )
    }
    async fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let path = args.get("design_path").and_then(|v| v.as_str()).unwrap_or("");
        if path.is_empty() { return Ok(json!({"success":false,"error":"no design_path"}).to_string()); }
        let gen_code = args.get("generate_code").and_then(|v| v.as_bool()).unwrap_or(false);
        let base = args.get("prompt").and_then(|v| v.as_str()).unwrap_or("Describe this UI design.");
        let prompt = if gen_code { format!("{base}. Generate HTML/CSS code for this design.") } else { base.to_string() };
        match self.ctx.vision_chat(&prompt, &[path]).await {
            Ok(desc) => Ok(json!({"success":true,"description":desc,"code_generated":gen_code}).to_string()),
            Err(e) => Ok(json!({"success":false,"error":e.to_string()}).to_string()),
        }
    }
}

pub struct ImageConsistencyTool { pub ctx: MultimodalCtx }
#[async_trait::async_trait]
impl Tool for ImageConsistencyTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("analyze_image_consistency","Check image consistency (people or objects) across multiple images.",
            vec![
                ToolParameter::new("image_paths",ParamType::String,true,"Comma-separated image paths (at least 2)",&["images","paths","files"]),
                ToolParameter::new("prompt",ParamType::String,false,"What aspect to check for consistency",&["question","aspect"]),
            ],
        )
    }
    async fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let paths_s = args.get("image_paths").and_then(|v| v.as_str()).unwrap_or("");
        let paths = split_paths(paths_s);
        if paths.len() < 2 { return Ok(json!({"success":false,"error":"need at least 2 images for consistency check"}).to_string()); }
        let prompt = args.get("prompt").and_then(|v| v.as_str()).unwrap_or("Compare these images and check for consistency.");
        match self.ctx.vision_chat(prompt, &paths).await {
            Ok(desc) => Ok(json!({"success":true,"description":desc,"images_count":paths.len()}).to_string()),
            Err(e) => Ok(json!({"success":false,"error":e.to_string()}).to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_chain_display_single() {
        let e = std::io::Error::new(std::io::ErrorKind::TimedOut, "operation timed out");
        let out = error_chain_display(&e);
        assert!(out.contains("operation timed out"));
    }

    #[test]
    fn error_chain_display_source_chain() {
        let inner = std::io::Error::new(std::io::ErrorKind::TimedOut, "operation timed out");
        let outer = std::io::Error::new(std::io::ErrorKind::Other, inner);
        let out = error_chain_display(&outer);
        assert!(out.contains("operation timed out"), "must include source: {out}");
    }

    #[test]
    fn error_chain_no_source() {
        let e = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let out = error_chain_display(&e);
        assert_eq!(out, "file not found");
    }

    #[test]
    fn truncate_short_enough() {
        assert_eq!(truncate("hi", 10), "hi");
    }

    #[test]
    fn truncate_too_long() {
        let s = "0123456789".repeat(5);
        let out = truncate(&s, 5);
        assert!(out.len() <= 10);
        assert!(out.ends_with("..."));
    }

    #[test]
    fn split_paths_empty() {
        assert!(split_paths("").is_empty());
    }

    #[test]
    fn split_paths_multiple() {
        let v = split_paths("a.jpg, b.png , c.gif");
        assert_eq!(v, vec!["a.jpg", "b.png", "c.gif"]);
    }

    // ── prepare_image tests ──────────────────────────────────────────

    fn make_test_rgba(w: u32, h: u32) -> Vec<u8> {
        let mut img = image::RgbaImage::new(w, h);
        // fill with a simple gradient so it's not blank
        for (x, _y, px) in img.enumerate_pixels_mut() {
            *px = image::Rgba([(x % 256) as u8, 128, 200, 255]);
        }
        let mut buf = Vec::new();
        let enc = image::codecs::png::PngEncoder::new(&mut buf);
        enc.write_image(&img, w, h, image::ExtendedColorType::Rgba8).unwrap();
        buf
    }

    #[test]
    fn prepare_image_resizes_large_photo() {
        let png = make_test_rgba(2048, 1536);
        let (out, mime) = prepare_image(&png, "jpg", MAX_IMAGE_DIMENSION);
        assert_eq!(mime, "image/jpeg");
        // Verify short side is ≤ max_dim after resize
        let decoded = image::load_from_memory(&out).unwrap();
        let (w, h) = decoded.dimensions();
        assert_eq!(w.min(h), MAX_IMAGE_DIMENSION);
    }

    #[test]
    fn prepare_image_resizes_large_screenshot() {
        let png = make_test_rgba(3000, 4000);
        let (out, mime) = prepare_image(&png, "png", MAX_IMAGE_DIMENSION);
        assert_eq!(mime, "image/png");
        let decoded = image::load_from_memory(&out).unwrap();
        let (w, h) = decoded.dimensions();
        assert_eq!(w.min(h), MAX_IMAGE_DIMENSION);
    }

    #[test]
    fn prepare_image_small_unchanged() {
        let png = make_test_rgba(200, 150);
        let (out, mime) = prepare_image(&png, "png", MAX_IMAGE_DIMENSION);
        assert_eq!(mime, "image/png");
        assert_eq!(out, png, "small image must be returned unchanged");
    }

    #[test]
    fn prepare_image_gif_untouched() {
        let png = make_test_rgba(2000, 2000);
        let (out, mime) = prepare_image(&png, "gif", MAX_IMAGE_DIMENSION);
        assert_eq!(mime, "image/gif");
        assert_eq!(out, png, "GIF must be returned unchanged");
    }

    #[test]
    fn prepare_image_invalid_bytes_fallback() {
        let garbage = b"not an image at all";
        let (out, mime) = prepare_image(garbage, "jpg", MAX_IMAGE_DIMENSION);
        assert_eq!(mime, "image/jpeg");
        assert_eq!(out, garbage.as_slice(), "invalid bytes must fall back to original");
    }

    #[test]
    fn prepare_image_webp_to_jpeg() {
        let png = make_test_rgba(3000, 2000);
        let (out, mime) = prepare_image(&png, "webp", MAX_IMAGE_DIMENSION);
        assert_eq!(mime, "image/jpeg");
        let decoded = image::load_from_memory(&out).unwrap();
        let (w, h) = decoded.dimensions();
        assert_eq!(w.min(h), MAX_IMAGE_DIMENSION);
    }
}
