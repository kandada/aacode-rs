// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Multimodal tools — understand_image / understand_video / understand_ui_design
//! / analyze_image_consistency.
//!
//! Ported from Python `tools/multimodal_tools.py`. Uses an OpenAI-compatible
//! vision chat request (image_url with base64 data URI). Requires a multimodal
//! model to be configured (`config.multimodal`). Reads image files from the
//! sandbox project path.

use super::registry::Tool;
use super::schema::{ParamType, ToolParameter, ToolSchema};
use crate::config::ModelConfig;
use crate::error::Result;
use base64::Engine;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

/// Shared config for the multimodal tools.
#[derive(Clone)]
pub struct MultimodalCtx {
    pub model: Option<ModelConfig>,
    pub project_path: PathBuf,
    pub timeout_secs: u64,
}

impl MultimodalCtx {
    fn resolve_path(&self, p: &str) -> PathBuf {
        let path = PathBuf::from(p);
        if path.is_absolute() {
            path
        } else {
            self.project_path.join(p)
        }
    }

    fn image_data_uri(&self, path: &str) -> Option<String> {
        let full = self.resolve_path(path);
        let bytes = std::fs::read(&full).ok()?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        let ext = full
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_else(|| "png".into());
        let mime = match ext.as_str() {
            "jpg" | "jpeg" => "image/jpeg",
            "gif" => "image/gif",
            "webp" => "image/webp",
            "bmp" => "image/bmp",
            _ => "image/png",
        };
        Some(format!("data:{mime};base64,{b64}"))
    }

    /// Perform a vision chat completion. Returns the assistant text.
    fn vision_chat(&self, prompt: &str, image_paths: &[&str]) -> Result<String> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| crate::error::AacodeError::Config("no multimodal model configured".into()))?;
        let api_key = model
            .api_key
            .clone()
            .filter(|k| !k.trim().is_empty())
            .ok_or_else(|| crate::error::AacodeError::Config("multimodal api key missing".into()))?;

        let mut content = vec![json!({"type": "text", "text": prompt})];
        for p in image_paths {
            match self.image_data_uri(p) {
                Some(uri) => content.push(json!({"type": "image_url", "image_url": {"url": uri}})),
                None => {
                    return Err(crate::error::AacodeError::Io(format!(
                        "cannot read image: {p}"
                    )))
                }
            }
        }

        let endpoint = format!(
            "{}/chat/completions",
            model.resolved_base_url().trim_end_matches('/')
        );
        let body = json!({
            "model": model.name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": model.max_tokens,
            "stream": false,
        });
        let resp = ureq::post(&endpoint)
            .timeout(std::time::Duration::from_secs(self.timeout_secs))
            .set("Authorization", &format!("Bearer {api_key}"))
            .set("Content-Type", "application/json")
            .send_json(body);
        match resp {
            Ok(r) => {
                let s = r.into_string().unwrap_or_default();
                let v: Value = serde_json::from_str(&s)?;
                let text = v["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();
                Ok(text)
            }
            Err(ureq::Error::Status(code, r)) => Err(crate::error::AacodeError::Api(format!(
                "HTTP {code}: {}",
                r.into_string().unwrap_or_default()
            ))),
            Err(e) => Err(crate::error::AacodeError::Network(e.to_string())),
        }
    }
}

fn split_paths(s: &str) -> Vec<&str> {
    s.split(',').map(|p| p.trim()).filter(|p| !p.is_empty()).collect()
}

pub struct UnderstandImageTool {
    pub ctx: MultimodalCtx,
}
impl Tool for UnderstandImageTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "understand_image",
            "Understand image content (one or more, comma-separated). Analyze screenshots, photos, design drafts.",
            vec![
                ToolParameter::new("image_path", ParamType::String, true, "Image path(s), comma-separated", &["image", "path", "file"]),
                ToolParameter::new("prompt", ParamType::String, false, "What to ask about the image", &["question", "query"]),
            ],
        )
    }
    fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let paths_s = args.get("image_path").and_then(|v| v.as_str()).unwrap_or("");
        let prompt = args
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("Describe this image in detail.");
        let paths = split_paths(paths_s);
        if paths.is_empty() {
            return Ok(json!({"success": false, "error": "no image_path"}).to_string());
        }
        match self.ctx.vision_chat(prompt, &paths) {
            Ok(desc) => Ok(json!({"success": true, "description": desc, "images_count": paths.len()}).to_string()),
            Err(e) => Ok(json!({"success": false, "error": e.to_string()}).to_string()),
        }
    }
}

pub struct UnderstandVideoTool {
    pub ctx: MultimodalCtx,
}
impl Tool for UnderstandVideoTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "understand_video",
            "Understand video content. (Requires a multimodal model that supports video; otherwise reports unsupported.)",
            vec![
                ToolParameter::new("video_path", ParamType::String, true, "Video file path", &["video", "path", "file"]),
                ToolParameter::new("prompt", ParamType::String, false, "What to ask about the video", &["question", "query"]),
            ],
        )
    }
    fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        // Video frames extraction is out of scope without ffmpeg; report clearly.
        let path = args.get("video_path").and_then(|v| v.as_str()).unwrap_or("");
        if path.is_empty() {
            return Ok(json!({"success": false, "error": "no video_path"}).to_string());
        }
        Ok(json!({
            "success": false,
            "error": "video understanding requires frame extraction (ffmpeg) not available on-device; extract frames via run_shell then use understand_image",
        })
        .to_string())
    }
}

pub struct UnderstandUiDesignTool {
    pub ctx: MultimodalCtx,
}
impl Tool for UnderstandUiDesignTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "understand_ui_design",
            "Analyze a UI design mockup/screenshot and optionally generate frontend code.",
            vec![
                ToolParameter::new("design_path", ParamType::String, true, "Design image path(s)", &["design", "path", "image", "file"]),
                ToolParameter::new("prompt", ParamType::String, false, "Analysis request", &["question", "query"]),
                ToolParameter::new("generate_code", ParamType::Boolean, false, "Generate HTML/CSS", &["code", "gen_code"]),
            ],
        )
    }
    fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let paths_s = args.get("design_path").and_then(|v| v.as_str()).unwrap_or("");
        let gen = args.get("generate_code").and_then(|v| v.as_bool()).unwrap_or(true);
        let base = args
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("Analyze this UI design: layout, colors, components.");
        let prompt = if gen {
            format!("{base}\n\nThen generate corresponding HTML/CSS code.")
        } else {
            base.to_string()
        };
        let paths = split_paths(paths_s);
        if paths.is_empty() {
            return Ok(json!({"success": false, "error": "no design_path"}).to_string());
        }
        match self.ctx.vision_chat(&prompt, &paths) {
            Ok(analysis) => Ok(json!({"success": true, "analysis": analysis}).to_string()),
            Err(e) => Ok(json!({"success": false, "error": e.to_string()}).to_string()),
        }
    }
}

pub struct ImageConsistencyTool {
    pub ctx: MultimodalCtx,
}
impl Tool for ImageConsistencyTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "analyze_image_consistency",
            "Analyze consistency (people/objects) across multiple images.",
            vec![
                ToolParameter::new("image_paths", ParamType::String, true, "Comma-separated image paths", &["images", "paths", "files", "image_path"]),
                ToolParameter::new("prompt", ParamType::String, false, "Consistency question", &["question", "query"]),
            ],
        )
    }
    fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let paths_s = args.get("image_paths").and_then(|v| v.as_str()).unwrap_or("");
        let prompt = args
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("Are these images consistent (same person/object/style)?");
        let paths = split_paths(paths_s);
        if paths.len() < 2 {
            return Ok(json!({"success": false, "error": "need at least 2 images"}).to_string());
        }
        match self.ctx.vision_chat(prompt, &paths) {
            Ok(analysis) => Ok(json!({"success": true, "analysis": analysis, "images_count": paths.len()}).to_string()),
            Err(e) => Ok(json!({"success": false, "error": e.to_string()}).to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ctx() -> MultimodalCtx {
        MultimodalCtx {
            model: None,
            project_path: std::env::temp_dir(),
            timeout_secs: 5,
        }
    }

    #[test]
    fn split_paths_works() {
        assert_eq!(split_paths("a.png, b.jpg"), vec!["a.png", "b.jpg"]);
        assert_eq!(split_paths("").len(), 0);
    }

    #[test]
    fn understand_image_no_path() {
        let t = UnderstandImageTool { ctx: ctx() };
        let cancel = AtomicBool::new(false);
        let out = t.call(&json!({"image_path": ""}), &cancel).unwrap();
        assert_eq!(serde_json::from_str::<Value>(&out).unwrap()["success"], false);
    }

    #[test]
    fn understand_image_no_model() {
        let d = std::env::temp_dir().join(format!("mm_{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("x.png"), b"fakepng").unwrap();
        let c = MultimodalCtx {
            model: None,
            project_path: d,
            timeout_secs: 5,
        };
        let t = UnderstandImageTool { ctx: c };
        let cancel = AtomicBool::new(false);
        let out = t.call(&json!({"image_path": "x.png"}), &cancel).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["success"], false);
        assert!(v["error"].as_str().unwrap().contains("multimodal model"));
    }

    #[test]
    fn consistency_needs_two() {
        let t = ImageConsistencyTool { ctx: ctx() };
        let cancel = AtomicBool::new(false);
        let out = t.call(&json!({"image_paths": "a.png"}), &cancel).unwrap();
        assert!(out.contains("at least 2"));
    }
}
