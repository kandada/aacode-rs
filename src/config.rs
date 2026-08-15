// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Configuration for the agent.
//!
//! Resolution priority (highest first):
//!   inline taskJson  >  environment variables  >  aacode_config.json  >  defaults
//!
//! Ported from the Python `config.py` (ModelConfig auto-detection of base_url
//! and gateway based on the model name).

use serde::{Deserialize, Serialize};

/// LLM gateway flavor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Gateway {
    Openai,
    Anthropic,
}

impl Default for Gateway {
    fn default() -> Self {
        Gateway::Openai
    }
}

impl Gateway {
    pub fn as_str(&self) -> &'static str {
        match self {
            Gateway::Openai => "openai",
            Gateway::Anthropic => "anthropic",
        }
    }

    pub fn parse(s: &str) -> Gateway {
        match s.trim().to_lowercase().as_str() {
            "anthropic" => Gateway::Anthropic,
            _ => Gateway::Openai,
        }
    }
}

/// Model / LLM connection configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default = "default_model_name")]
    pub name: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub gateway: Gateway,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub multimodal: bool,
    /// Per-request deadline (secs). Falls back to config.timeouts.model_request.
    #[serde(default)]
    pub request_timeout_secs: Option<u64>,
}

fn default_model_name() -> String {
    "deepseek-chat".to_string()
}
fn default_temperature() -> f32 {
    0.1
}
fn default_max_tokens() -> u32 {
    8192
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            name: default_model_name(),
            api_key: None,
            base_url: None,
            gateway: Gateway::Openai,
            temperature: default_temperature(),
            max_tokens: default_max_tokens(),
            multimodal: false,
            request_timeout_secs: None,
        }
    }
}

impl ModelConfig {
    /// Resolve the effective base URL. If explicitly set, use it. Otherwise
    /// infer from the model name + gateway, mirroring the Python behavior.
    pub fn resolved_base_url(&self) -> String {
        if let Some(url) = &self.base_url {
            if !url.trim().is_empty() {
                return url.clone();
            }
        }
        let m = self.name.to_lowercase();
        let is_kimi = m.contains("kimi") || m.contains("moonshot");
        let is_minimax = m.contains("minimax");
        let is_deepseek = m.contains("deepseek");
        let is_claude = m.contains("claude");

        match self.gateway {
            Gateway::Anthropic => {
                if is_minimax {
                    "https://api.minimax.chat/anthropic".into()
                } else if is_deepseek {
                    "https://api.deepseek.com/anthropic".into()
                } else if is_kimi {
                    "https://api.moonshot.cn/anthropic".into()
                } else {
                    "https://api.anthropic.com".into()
                }
            }
            Gateway::Openai => {
                if is_minimax {
                    "https://api.minimax.chat/v1".into()
                } else if is_deepseek {
                    "https://api.deepseek.com/v1".into()
                } else if is_kimi {
                    "https://api.moonshot.cn/v1".into()
                } else if is_claude {
                    "https://api.openai.com/v1".into()
                } else {
                    "https://api.openai.com/v1".into()
                }
            }
        }
    }

    /// Apply environment variable overrides (env wins over file/defaults).
    pub fn apply_env(&mut self) {
        if let Ok(k) = std::env::var("LLM_API_KEY").or_else(|_| std::env::var("OPENAI_API_KEY")) {
            if !k.is_empty() {
                self.api_key = Some(k);
            }
        }
        if let Ok(u) = std::env::var("LLM_API_URL").or_else(|_| std::env::var("OPENAI_BASE_URL")) {
            if !u.is_empty() {
                self.base_url = Some(u);
            }
        }
        if let Ok(n) = std::env::var("LLM_MODEL_NAME") {
            if !n.is_empty() {
                self.name = n;
            }
        }
        if let Ok(g) = std::env::var("LLM_GATEWAY") {
            if !g.is_empty() {
                self.gateway = Gateway::parse(&g);
            }
        }
        if let Ok(mm) = std::env::var("LLM_MULTIMODAL") {
            let l = mm.to_lowercase();
            self.multimodal = matches!(l.as_str(), "true" | "1" | "yes" | "on");
        }
    }
}

/// Timeouts (seconds).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timeouts {
    #[serde(default = "d_shell")]
    pub shell_command: u64,
    #[serde(default = "d_shell_idle")]
    pub shell_idle: u64,
    #[serde(default = "d_web")]
    pub web_request: u64,
    #[serde(default = "d_model")]
    pub model_request: u64,
}
fn d_shell() -> u64 {
    120
}
fn d_shell_idle() -> u64 {
    30
}
fn d_web() -> u64 {
    15
}
fn d_model() -> u64 {
    300
}
impl Default for Timeouts {
    fn default() -> Self {
        Timeouts {
            shell_command: d_shell(),
            shell_idle: d_shell_idle(),
            web_request: d_web(),
            model_request: d_model(),
        }
    }
}

/// Output / truncation limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Limits {
    #[serde(default = "d_tool_output")]
    pub tool_output_chars: usize,
    #[serde(default = "d_display_preview")]
    pub display_preview_chars: usize,
    #[serde(default = "d_max_retries")]
    pub max_retries: u32,
}
fn d_tool_output() -> usize {
    24000
}
fn d_display_preview() -> usize {
    3000
}
fn d_max_retries() -> u32 {
    2
}
impl Default for Limits {
    fn default() -> Self {
        Limits {
            tool_output_chars: d_tool_output(),
            display_preview_chars: d_display_preview(),
            max_retries: d_max_retries(),
        }
    }
}

/// Dangerous-command handling policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DangerAction {
    Log,
    Reject,
    Ask,
}
impl Default for DangerAction {
    fn default() -> Self {
        DangerAction::Log
    }
}

/// Safety configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyConfig {
    #[serde(default)]
    pub dangerous_command_action: DangerAction,
}
impl Default for SafetyConfig {
    fn default() -> Self {
        SafetyConfig {
            dangerous_command_action: DangerAction::Log,
        }
    }
}

/// Context / compaction configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    #[serde(default = "d_max_ctx")]
    pub max_context_tokens: usize,
    #[serde(default = "d_trigger")]
    pub compact_trigger_tokens: usize,
    #[serde(default = "d_protect_first")]
    pub protect_first_rounds: usize,
    #[serde(default = "d_keep_last")]
    pub keep_last_rounds: usize,
    #[serde(default = "d_protect_user")]
    pub protect_last_user_rounds: usize,
}
fn d_max_ctx() -> usize {
    262144
}
// Aligned with the Python aacode default (config.py: compact_trigger_tokens).
fn d_trigger() -> usize {
    256000
}
fn d_protect_first() -> usize {
    1
}
fn d_keep_last() -> usize {
    10
}
fn d_protect_user() -> usize {
    2
}
impl Default for ContextConfig {
    fn default() -> Self {
        ContextConfig {
            max_context_tokens: d_max_ctx(),
            compact_trigger_tokens: d_trigger(),
            protect_first_rounds: d_protect_first(),
            keep_last_rounds: d_keep_last(),
            protect_last_user_rounds: d_protect_user(),
        }
    }
}

/// Search engine configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchConfig {
    #[serde(default)]
    pub searxng_url: Option<String>,
    #[serde(default)]
    pub brave_api_key: Option<String>,
    #[serde(default)]
    pub google_cse_key: Option<String>,
    #[serde(default)]
    pub google_cse_cx: Option<String>,
    #[serde(default)]
    pub bing_api_key: Option<String>,
    #[serde(default)]
    pub serpapi_key: Option<String>,
}

impl SearchConfig {
    pub fn apply_env(&mut self) {
        if let Ok(u) = std::env::var("SEARCHXNG_URL").or_else(|_| std::env::var("SEARXNG_URL")) {
            if !u.is_empty() {
                self.searxng_url = Some(u);
            }
        }
        if let Ok(k) = std::env::var("BRAVE_API_KEY") {
            if !k.is_empty() {
                self.brave_api_key = Some(k);
            }
        }
        if let Ok(k) = std::env::var("SERPAPI_KEY") {
            if !k.is_empty() {
                self.serpapi_key = Some(k);
            }
        }
    }
}

/// Shell execution backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ShellBackendChoice {
    /// Real OS shell (sh -c / cmd /C). Natural on desktop.
    Native,
    /// fastshell sandbox engine (VFS jail + built-in commands + embedded CPython).
    Fastshell,
    /// Pick based on target OS: native on desktop, fastshell on mobile.
    Auto,
}

impl Default for ShellBackendChoice {
    fn default() -> Self {
        ShellBackendChoice::Auto
    }
}

/// Skills discovery configuration (host-provided, platform-agnostic).
///
/// When `user_dir` is set (mobile hosts pass an absolute path inside their
/// sandbox), discovery switches to "builtin + user dir" mode and project
/// directories are NOT scanned. When unset, the legacy per-project scanning
/// (`<project>/skills` + `<project>/.aacode/skills`) is preserved unchanged.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SkillsConfig {
    /// Absolute path of the user-managed skills directory.
    #[serde(default)]
    pub user_dir: Option<String>,
    /// Host-declared extra builtins beyond skill_creator. These are compiled
    /// into the binary but only injected when the host explicitly names them
    /// here (e.g. `["agent_cron"]`). Empty by default — desktop/CLI hosts
    /// that don't pass this field never see platform-specific builtins.
    #[serde(default)]
    pub extra_builtins: Vec<String>,
    /// VFS-internal path for `{SKILLS_DIR}` substitution in skill prompts.
    /// Computed at runtime from `user_dir` relative to the shell sandbox root.
    ///
    /// The path starts with `/` (e.g. `/skills`) because the VFS sandbox
    /// interprets leading-`/` paths as relative to the VFS root rather than
    /// the real filesystem root. This keeps prompts short and platform-neutral.
    ///
    /// When `None`, substitution falls back to `user_dir` (legacy absolute path).
    /// Marked `#[serde(skip)]` because the host never sets it — the shell
    /// root widening logic in ffi.rs computes it.
    #[serde(skip, default)]
    pub vfs_skills_dir: Option<String>,
}

/// The complete agent configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    #[serde(default)]
    pub model: ModelConfig,
    /// Optional dedicated multimodal (vision) model config.
    #[serde(default)]
    pub multimodal: Option<ModelConfig>,
    #[serde(default = "d_max_iter")]
    pub max_iterations: u32,
    #[serde(default)]
    pub plan_first: bool,
    /// Which shell backend to use. Defaults to Auto (native on desktop,
    /// fastshell sandbox on mobile).
    #[serde(default)]
    pub shell_backend: ShellBackendChoice,
    #[serde(default)]
    pub timeouts: Timeouts,
    #[serde(default)]
    pub limits: Limits,
    #[serde(default)]
    pub safety: SafetyConfig,
    #[serde(default)]
    pub context: ContextConfig,
    #[serde(default)]
    pub search: SearchConfig,
    #[serde(default)]
    pub skills: SkillsConfig,
}

fn d_max_iter() -> u32 {
    300
}

impl Default for AgentConfig {
    fn default() -> Self {
        AgentConfig {
            model: ModelConfig::default(),
            multimodal: None,
            max_iterations: d_max_iter(),
            plan_first: false,
            shell_backend: ShellBackendChoice::Auto,
            timeouts: Timeouts::default(),
            limits: Limits::default(),
            safety: SafetyConfig::default(),
            context: ContextConfig::default(),
            search: SearchConfig::default(),
            skills: SkillsConfig::default(),
        }
    }
}

impl AgentConfig {
    /// Apply environment overrides to the model + search config.
    pub fn apply_env(&mut self) {
        self.model.apply_env();
        // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0
        self.search.apply_env();
        if let Some(mm) = self.multimodal.as_mut() {
            mm.apply_env();
        }
        // AACODE_SHELL_BACKEND = native | fastshell | auto
        if let Ok(b) = std::env::var("AACODE_SHELL_BACKEND") {
            self.shell_backend = match b.trim().to_lowercase().as_str() {
                "native" => ShellBackendChoice::Native,
                "fastshell" => ShellBackendChoice::Fastshell,
                _ => ShellBackendChoice::Auto,
            };
        }
        // AACODE_SKILLS_DIR = absolute path of the user skills directory.
        // Gives desktop/CLI hosts the same "builtin + user dir" skills mode
        // that mobile hosts enable via task_json {"skills":{"user_dir":…}}.
        if let Ok(d) = std::env::var("AACODE_SKILLS_DIR") {
            let d = d.trim();
            if !d.is_empty() {
                self.skills.user_dir = Some(d.to_string());
            }
        }
    }

    /// Validate required fields; returns a list of human-readable problems.
    pub fn validate(&self) -> Vec<String> {
        // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0
        let mut errs = Vec::new();
        let has_key = self
            .model
            .api_key
            .as_ref()
            .map(|k| !k.trim().is_empty())
            .unwrap_or(false);
        if !has_key {
            errs.push(
                "LLM API key not configured (set model.api_key or env LLM_API_KEY)".to_string(),
            );
        }
        errs
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn task_json_context_trigger_parses() {
        // The Android app sends {"context":{"compact_trigger_tokens":N}} in
        // the task JSON — must round-trip into ContextConfig.
        let v: serde_json::Value = serde_json::json!({
            "model": {"name": "m", "api_key": "k"},
            "context": {"compact_trigger_tokens": 123456}
        });
        let cfg: super::AgentConfig = serde_json::from_value(v).unwrap();
        assert_eq!(cfg.context.compact_trigger_tokens, 123456);
        // Default aligned with the Python aacode config (256000).
        assert_eq!(super::ContextConfig::default().compact_trigger_tokens, 256000);
    }

    #[test]
    fn task_json_skills_user_dir_parses() {
        // The Android app sends {"skills":{"user_dir":"/abs/path"}} — must
        // round-trip into SkillsConfig; absent → None (legacy mode).
        let v: serde_json::Value = serde_json::json!({
            "model": {"name": "m", "api_key": "k"},
            "skills": {"user_dir": "/data/data/app/files/sandbox/skills"}
        });
        let cfg: super::AgentConfig = serde_json::from_value(v).unwrap();
        assert_eq!(
            cfg.skills.user_dir.as_deref(),
            Some("/data/data/app/files/sandbox/skills")
        );
        let v2: serde_json::Value = serde_json::json!({"model": {"name": "m", "api_key": "k"}});
        let cfg2: super::AgentConfig = serde_json::from_value(v2).unwrap();
        assert!(cfg2.skills.user_dir.is_none());
    }

    #[test]
    fn env_skills_dir_applies() {
        std::env::set_var("AACODE_SKILLS_DIR", "/tmp/aacode_env_skills");
        let mut cfg = super::AgentConfig::default();
        cfg.apply_env();
        std::env::remove_var("AACODE_SKILLS_DIR");
        assert_eq!(cfg.skills.user_dir.as_deref(), Some("/tmp/aacode_env_skills"));
    }

    use super::*;

    #[test]
    fn base_url_inference() {
        let mut m = ModelConfig {
            name: "deepseek-chat".into(),
            gateway: Gateway::Openai,
            ..Default::default()
        };
        assert_eq!(m.resolved_base_url(), "https://api.deepseek.com/v1");

        m.name = "kimi-k2".into();
        assert_eq!(m.resolved_base_url(), "https://api.moonshot.cn/v1");

        m.name = "MiniMax-M2".into();
        m.gateway = Gateway::Anthropic;
        assert_eq!(m.resolved_base_url(), "https://api.minimax.chat/anthropic");

        m.name = "claude-3".into();
        m.gateway = Gateway::Anthropic;
        assert_eq!(m.resolved_base_url(), "https://api.anthropic.com");
    }

    #[test]
    fn explicit_base_url_wins() {
        let m = ModelConfig {
            name: "deepseek-chat".into(),
            base_url: Some("http://localhost:9999/v1".into()),
            ..Default::default()
        };
        assert_eq!(m.resolved_base_url(), "http://localhost:9999/v1");
    }

    #[test]
    fn config_json_roundtrip() {
        let cfg = AgentConfig::default();
        let s = serde_json::to_string(&cfg).unwrap();
        let back: AgentConfig = serde_json::from_str(&s).unwrap();
        assert_eq!(back.max_iterations, 300);
        assert_eq!(back.model.name, "deepseek-chat");
    }

    #[test]
    fn partial_json_uses_defaults() {
        let cfg: AgentConfig =
            serde_json::from_str(r#"{"model":{"name":"gpt-4","api_key":"x"}}"#).unwrap();
        assert_eq!(cfg.model.name, "gpt-4");
        assert_eq!(cfg.max_iterations, 300);
        assert_eq!(cfg.model.temperature, 0.1);
    }

    #[test]
    fn validate_requires_key() {
        let cfg = AgentConfig::default();
        assert!(!cfg.validate().is_empty());
        let cfg2: AgentConfig =
            serde_json::from_str(r#"{"model":{"api_key":"sk-x"}}"#).unwrap();
        assert!(cfg2.validate().is_empty());
    }
}
