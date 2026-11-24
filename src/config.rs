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
    #[serde(
        default = "default_max_tokens",
        deserialize_with = "deserialize_u32_lenient"
    )]
    pub max_tokens: u32,
    /// Model context window (input capacity). When absent, use context.max_context_tokens.
    #[serde(default, deserialize_with = "deserialize_opt_usize_lenient")]
    pub context_window: Option<usize>,
    /// Canonical per-route output cap. Falls back to max_tokens.
    #[serde(default, deserialize_with = "deserialize_opt_usize_lenient")]
    pub max_output_tokens: Option<usize>,
    /// Optional pricing table for cost accounting:
    /// {"input": <per-token>, "cache_read": <per-token>, "output": <per-token>}
    #[serde(default)]
    pub pricing: Option<serde_json::Value>,
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
    32_000
}

fn deserialize_u32_lenient<'de, D>(deserializer: D) -> std::result::Result<u32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    Ok(match value {
        serde_json::Value::Number(n) => n.as_u64().unwrap_or(0) as u32,
        serde_json::Value::String(s) => s.trim().parse().unwrap_or(0),
        _ => 0,
    })
}

fn deserialize_usize_lenient<'de, D>(deserializer: D) -> std::result::Result<usize, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    Ok(match value {
        serde_json::Value::Number(n) => n.as_u64().unwrap_or(0) as usize,
        serde_json::Value::String(s) => s.trim().parse().unwrap_or(0),
        _ => 0,
    })
}

fn deserialize_f64_lenient<'de, D>(deserializer: D) -> std::result::Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    Ok(match value {
        serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0),
        serde_json::Value::String(s) => s.trim().parse().unwrap_or(0.0),
        _ => 0.0,
    })
}

fn deserialize_opt_usize_lenient<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<usize>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    let n = match value {
        serde_json::Value::Number(n) => n.as_u64().unwrap_or(0) as usize,
        serde_json::Value::String(s) => s.trim().parse().unwrap_or(0),
        _ => 0,
    };
    Ok((n > 0).then_some(n))
}

fn deserialize_string_lenient<'de, D>(deserializer: D) -> std::result::Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    Ok(match value {
        serde_json::Value::String(s) => s,
        serde_json::Value::Number(n) => n.to_string(),
        _ => String::new(),
    })
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
            context_window: None,
            max_output_tokens: None,
            pricing: None,
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
    /// Upper bound for a foreground run_shell wait before promotion to a job.
    #[serde(default = "d_shell_foreground_max")]
    pub shell_foreground_max: u64,
    /// Default total lifetime for background shell jobs.
    #[serde(default = "d_shell_background")]
    pub shell_background: u64,
    #[serde(default = "d_web")]
    pub web_request: u64,
    #[serde(default = "d_model")]
    pub model_request: u64,
    /// Default timeout for code execution / tests.
    #[serde(default = "d_code_execution")]
    pub code_execution: u64,
    #[serde(default = "d_skill_timeout")]
    pub skill_discovery: u64,
    #[serde(default = "d_subagent_timeout")]
    pub subagent: u64,
}
fn d_shell() -> u64 {
    120
}
fn d_shell_idle() -> u64 {
    120
}
fn d_shell_foreground_max() -> u64 {
    600
}
fn d_shell_background() -> u64 {
    1_800
}
fn d_web() -> u64 {
    60
}
fn d_model() -> u64 {
    300
}
fn d_code_execution() -> u64 {
    300
}
fn d_skill_timeout() -> u64 {
    5
}
fn d_subagent_timeout() -> u64 {
    600
}
impl Default for Timeouts {
    fn default() -> Self {
        Timeouts {
            shell_command: d_shell(),
            shell_idle: d_shell_idle(),
            shell_foreground_max: d_shell_foreground_max(),
            shell_background: d_shell_background(),
            web_request: d_web(),
            model_request: d_model(),
            code_execution: d_code_execution(),
            skill_discovery: d_skill_timeout(),
            subagent: d_subagent_timeout(),
        }
    }
}

/// Output / truncation limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Limits {
    /// Observation-layer cap. 0 = no truncation for plain text (only binary is
    /// externalized). `run_shell` has its own default cap (see `ShellTool`).
    #[serde(default = "d_tool_output")]
    pub tool_output_chars: usize,
    /// `run_shell` line cap. 0 = unset → `ShellTool` default (500).
    #[serde(default = "d_tool_output_lines")]
    pub tool_output_lines: usize,
    #[serde(default = "d_display_preview")]
    pub display_preview_chars: usize,
    #[serde(default = "d_max_retries")]
    pub max_retries: u32,
}
fn d_tool_output() -> usize {
    // 0 = unlimited for the observation layer. `run_shell` truncation defaults
    // to `ShellTool::DEFAULT_MAX_OUTPUT_CHARS` (20000) unless
    // `tool_output_chars` is set; `full_output=true` opts out per call.
    0
}
fn d_tool_output_lines() -> usize {
    500
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
            tool_output_lines: d_tool_output_lines(),
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

/// Policy for commands that are not on PATH (native shell only).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UnknownAction {
    /// Allow (default): an unknown command simply fails in the shell.
    Allow,
    /// Ask: no interactive channel exists in aacode-rs; treated as allow.
    Ask,
    /// Reject before execution.
    Reject,
}
impl Default for UnknownAction {
    fn default() -> Self {
        UnknownAction::Allow
    }
}

/// Agent permission mode.
///
/// `full_access` is the default and keeps existing behaviour. `read_only`
/// blocks any tool that mutates files/state or executes code; browsing,
/// reading, searching and waiting remain available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionMode {
    FullAccess,
    ReadOnly,
}
impl Default for PermissionMode {
    fn default() -> Self {
        PermissionMode::FullAccess
    }
}

/// Safety configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyConfig {
    #[serde(default)]
    pub dangerous_command_action: DangerAction,
    /// Unknown-command policy (native shell only; `allow` by default).
    #[serde(default)]
    pub unknown_command_action: UnknownAction,
    /// `full_access` (default) or `read_only`.
    #[serde(default)]
    pub permission_mode: PermissionMode,
}
impl Default for SafetyConfig {
    fn default() -> Self {
        SafetyConfig {
            dangerous_command_action: DangerAction::Log,
            unknown_command_action: UnknownAction::default(),
            permission_mode: PermissionMode::default(),
        }
    }
}

/// Context / compaction configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    #[serde(default = "d_max_ctx", deserialize_with = "deserialize_usize_lenient")]
    pub max_context_tokens: usize,
    #[serde(default = "d_trigger", deserialize_with = "deserialize_usize_lenient")]
    pub compact_trigger_tokens: usize,
    #[serde(
        default = "d_protect_first",
        deserialize_with = "deserialize_usize_lenient"
    )]
    pub protect_first_rounds: usize,
    #[serde(
        default = "d_keep_last",
        deserialize_with = "deserialize_usize_lenient"
    )]
    pub keep_last_rounds: usize,
    #[serde(
        default = "d_protect_user",
        deserialize_with = "deserialize_usize_lenient"
    )]
    pub protect_last_user_rounds: usize,
    /// Retrieval-style context control (no summary). Advance a frozen cutoff
    /// when the model view exceeds this many tokens.
    #[serde(
        default = "d_retrieval_trigger",
        deserialize_with = "deserialize_usize_lenient"
    )]
    pub retrieval_trigger_tokens: usize,
    /// Budget of recent verbatim history kept after the cutoff.
    #[serde(
        default = "d_retrieval_keep",
        deserialize_with = "deserialize_usize_lenient"
    )]
    pub retrieval_keep_tokens: usize,
    /// Always try to keep the newest N real user messages.
    #[serde(
        default = "d_retrieval_recent",
        deserialize_with = "deserialize_usize_lenient"
    )]
    pub retrieval_keep_recent_user_messages: usize,
    /// Always try to keep at least this many recent rounds.
    #[serde(
        default = "d_retrieval_rounds",
        deserialize_with = "deserialize_usize_lenient"
    )]
    pub retrieval_keep_min_rounds: usize,
    /// Quality-first trigger as a fraction of the model window (default 0.8).
    #[serde(
        default = "d_window_ratio",
        deserialize_with = "deserialize_f64_lenient"
    )]
    pub window_ratio: f64,
    /// Reserve for the next model output + protocol overhead.
    #[serde(default = "d_headroom", deserialize_with = "deserialize_usize_lenient")]
    pub headroom_tokens: usize,
    /// Verbatim tail budget as a fraction of (window - output).
    #[serde(
        default = "d_retain_ratio",
        deserialize_with = "deserialize_f64_lenient"
    )]
    pub retain_ratio: f64,
    /// Explicit tail budget; overrides retain_ratio when set.
    #[serde(default, deserialize_with = "deserialize_opt_usize_lenient")]
    pub retain_tokens: Option<usize>,
    /// Optional cost-based trigger (prompt tokens). 0 = disabled.
    #[serde(default)]
    pub cost_ceiling_tokens: usize,
    /// Spill oversized tool results before they enter model-visible history.
    #[serde(
        default = "d_inline_tool_tokens",
        deserialize_with = "deserialize_usize_lenient"
    )]
    pub inline_tool_tokens: usize,
    #[serde(
        default = "d_inline_head_chars",
        deserialize_with = "deserialize_usize_lenient"
    )]
    pub inline_head_chars: usize,
    #[serde(
        default = "d_inline_tail_chars",
        deserialize_with = "deserialize_usize_lenient"
    )]
    pub inline_tail_chars: usize,
    /// Middle-prune old tool results when the view grows.
    #[serde(default = "d_prune_enabled")]
    pub prune_enabled: bool,
    #[serde(
        default = "d_prune_threshold",
        deserialize_with = "deserialize_usize_lenient"
    )]
    pub prune_threshold_chars: usize,
    #[serde(
        default = "d_prune_head",
        deserialize_with = "deserialize_usize_lenient"
    )]
    pub prune_head_chars: usize,
    #[serde(
        default = "d_prune_tail",
        deserialize_with = "deserialize_usize_lenient"
    )]
    pub prune_tail_chars: usize,
    #[serde(
        default = "d_prune_min_rounds",
        deserialize_with = "deserialize_usize_lenient"
    )]
    pub prune_min_rounds: usize,
}

fn d_window_ratio() -> f64 {
    0.8
}
fn d_headroom() -> usize {
    65536
}
fn d_retain_ratio() -> f64 {
    0.16
}
fn d_inline_tool_tokens() -> usize {
    6_000
}
fn d_inline_head_chars() -> usize {
    8_000
}
fn d_inline_tail_chars() -> usize {
    4_000
}
fn d_prune_enabled() -> bool {
    true
}
fn d_prune_threshold() -> usize {
    8_192
}
fn d_prune_head() -> usize {
    4_096
}
fn d_prune_tail() -> usize {
    1_024
}
fn d_prune_min_rounds() -> usize {
    4
}
fn d_max_ctx() -> usize {
    FALLBACK_CONTEXT_WINDOW
}
const FALLBACK_CONTEXT_WINDOW: usize = 1_000_000;
const FALLBACK_MAX_OUTPUT: usize = 32_000;
const MIN_CONTEXT_WINDOW: usize = 8_192;
const SAFETY_MARGIN_TOKENS: usize = 4_096;
fn d_retrieval_trigger() -> usize {
    100000
}
fn d_retrieval_keep() -> usize {
    20000
}
fn d_retrieval_recent() -> usize {
    2
}
fn d_retrieval_rounds() -> usize {
    4
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
            retrieval_trigger_tokens: d_retrieval_trigger(),
            retrieval_keep_tokens: d_retrieval_keep(),
            retrieval_keep_recent_user_messages: d_retrieval_recent(),
            retrieval_keep_min_rounds: d_retrieval_rounds(),
            window_ratio: d_window_ratio(),
            headroom_tokens: d_headroom(),
            retain_ratio: d_retain_ratio(),
            retain_tokens: None,
            cost_ceiling_tokens: 0,
            inline_tool_tokens: d_inline_tool_tokens(),
            inline_head_chars: d_inline_head_chars(),
            inline_tail_chars: d_inline_tail_chars(),
            prune_enabled: d_prune_enabled(),
            prune_threshold_chars: d_prune_threshold(),
            prune_head_chars: d_prune_head(),
            prune_tail_chars: d_prune_tail(),
            prune_min_rounds: d_prune_min_rounds(),
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
    /// Tool schema profile: core | auto | web | full.
    #[serde(
        default = "d_tool_profile",
        deserialize_with = "deserialize_string_lenient"
    )]
    pub tool_profile: String,
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
    // Compatibility aliases used by iOS's flat TaskConfig JSON. Android sends
    // the nested context object; both shapes are accepted.
    #[serde(default, deserialize_with = "deserialize_opt_usize_lenient")]
    pub context_window: Option<usize>,
    #[serde(default)]
    pub window_ratio: Option<f64>,
    #[serde(default, deserialize_with = "deserialize_opt_usize_lenient")]
    pub headroom_tokens: Option<usize>,
    #[serde(default)]
    pub retain_ratio: Option<f64>,
    #[serde(default, deserialize_with = "deserialize_opt_usize_lenient")]
    pub retain_tokens: Option<usize>,
    #[serde(default, deserialize_with = "deserialize_opt_usize_lenient")]
    pub cost_ceiling_tokens: Option<usize>,
    #[serde(default)]
    pub timeout: Option<u64>,
}

fn d_max_iter() -> u32 {
    300
}
fn d_tool_profile() -> String {
    "auto".to_string()
}

impl Default for AgentConfig {
    fn default() -> Self {
        AgentConfig {
            model: ModelConfig::default(),
            multimodal: None,
            max_iterations: d_max_iter(),
            tool_profile: d_tool_profile(),
            plan_first: false,
            shell_backend: ShellBackendChoice::Auto,
            timeouts: Timeouts::default(),
            limits: Limits::default(),
            safety: SafetyConfig::default(),
            context: ContextConfig::default(),
            search: SearchConfig::default(),
            skills: SkillsConfig::default(),
            context_window: None,
            window_ratio: None,
            headroom_tokens: None,
            retain_ratio: None,
            retain_tokens: None,
            cost_ceiling_tokens: None,
            timeout: None,
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
        if let Ok(profile) = std::env::var("AACODE_TOOL_PROFILE") {
            if !profile.trim().is_empty() {
                self.tool_profile = profile.trim().to_lowercase();
            }
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

    /// Repair zero/null legacy config values before they reach the LLM client.
    pub fn apply_compat_defaults(&mut self) {
        if self.model.max_tokens == 0 {
            self.model.max_tokens = FALLBACK_MAX_OUTPUT as u32;
        }
        if self.model.context_window == Some(0) {
            self.model.context_window = None;
        }
        if self.model.max_output_tokens == Some(0) {
            self.model.max_output_tokens = None;
        }
        if self.context.max_context_tokens == 0 {
            self.context.max_context_tokens = FALLBACK_CONTEXT_WINDOW;
        }
        if !self.context.window_ratio.is_finite()
            || self.context.window_ratio <= 0.0
            || self.context.window_ratio > 1.0
        {
            self.context.window_ratio = d_window_ratio();
        }
        if !self.context.retain_ratio.is_finite()
            || self.context.retain_ratio <= 0.0
            || self.context.retain_ratio > 1.0
        {
            self.context.retain_ratio = d_retain_ratio();
        }
        if self.context.retain_tokens == Some(0) {
            self.context.retain_tokens = None;
        }
        if self.context.max_context_tokens < MIN_CONTEXT_WINDOW {
            self.context.max_context_tokens = MIN_CONTEXT_WINDOW;
        }
        if self
            .model
            .context_window
            .map(|v| v > 0 && v < MIN_CONTEXT_WINDOW)
            .unwrap_or(false)
        {
            self.model.context_window = Some(MIN_CONTEXT_WINDOW);
        }
        let window = self.effective_context_window();
        let output_ceiling = window
            .saturating_sub(SAFETY_MARGIN_TOKENS)
            .max(1)
            .min(FALLBACK_MAX_OUTPUT);
        let output = self.effective_max_output().clamp(1, output_ceiling.max(1));
        if self.model.max_output_tokens.is_some() {
            self.model.max_output_tokens = Some(output);
        } else {
            self.model.max_tokens = output as u32;
        }
        if !self.model.temperature.is_finite() {
            self.model.temperature = default_temperature();
        } else {
            self.model.temperature = self.model.temperature.clamp(0.0, 2.0);
        }
        if self.context.headroom_tokens == 0 {
            self.context.headroom_tokens = d_headroom();
        } else if self.context.headroom_tokens >= window {
            self.context.headroom_tokens = (window / 16).max(1);
        }
        if self.tool_profile.trim().is_empty() {
            self.tool_profile = d_tool_profile();
        }
    }

    /// Apply flat compatibility fields (iOS TaskConfig) to the nested policy.
    pub fn apply_flat_context(&mut self) {
        if let Some(v) = self.context_window {
            if v > 0 {
                self.model.context_window = Some(v);
            }
        }
        if let Some(v) = self.window_ratio {
            self.context.window_ratio = v;
        }
        if let Some(v) = self.headroom_tokens {
            self.context.headroom_tokens = v;
        }
        if let Some(v) = self.retain_ratio {
            self.context.retain_ratio = v;
        }
        if let Some(v) = self.retain_tokens {
            self.context.retain_tokens = Some(v);
        }
        if let Some(v) = self.cost_ceiling_tokens {
            self.context.cost_ceiling_tokens = v;
        }
        if let Some(v) = self.timeout {
            if v > 0 {
                self.timeouts.model_request = v;
            }
        }
    }

    /// Effective context window for this route.
    ///
    /// Legacy configs sometimes persisted 0 for fields that were added later.
    /// Treat 0 as unset; otherwise the old value becomes a 1-token window and
    /// compaction fires on the first request.
    pub fn effective_context_window(&self) -> usize {
        self.model
            .context_window
            .filter(|&v| v > 0)
            .or_else(|| {
                (self.context.max_context_tokens > 0).then_some(self.context.max_context_tokens)
            })
            .unwrap_or(FALLBACK_CONTEXT_WINDOW)
            .max(1)
    }

    /// Effective max output for this route.
    pub fn effective_max_output(&self) -> usize {
        let max_tokens = self.model.max_tokens as usize;
        self.model
            .max_output_tokens
            .filter(|&v| v > 0)
            .or_else(|| (max_tokens > 0).then_some(max_tokens.min(FALLBACK_MAX_OUTPUT)))
            .unwrap_or(FALLBACK_MAX_OUTPUT)
            .max(1)
    }

    fn effective_window_ratio(&self) -> f64 {
        let ratio = self.context.window_ratio;
        if ratio.is_finite() && ratio > 0.0 && ratio <= 1.0 {
            ratio
        } else {
            d_window_ratio()
        }
    }

    fn effective_retain_ratio(&self) -> f64 {
        let ratio = self.context.retain_ratio;
        if ratio.is_finite() && ratio > 0.0 && ratio <= 1.0 {
            ratio
        } else {
            d_retain_ratio()
        }
    }

    /// Quality-first trigger: min(window*ratio, window-output-headroom).
    pub fn effective_trigger(&self) -> usize {
        let window = self.effective_context_window();
        let output = self.effective_max_output();
        let hard_cap = window
            .saturating_sub(output)
            .saturating_sub(self.context.headroom_tokens)
            .max(1);
        // An explicitly-set retrieval trigger wins, but never above the
        // provider-safe input budget (otherwise compaction would fire after
        // the hard limit and every request would fail).
        let configured = self.context.retrieval_trigger_tokens;
        if configured > 0 && configured != d_retrieval_trigger() {
            let mut trigger = configured.min(hard_cap).max(1);
            if self.context.cost_ceiling_tokens > 0 {
                trigger = trigger.min(self.context.cost_ceiling_tokens.max(1));
            }
            return trigger;
        }
        let pressure = ((window as f64) * self.effective_window_ratio().clamp(0.1, 0.95)) as usize;
        let available = window
            .saturating_sub(output)
            .saturating_sub(self.context.headroom_tokens);
        let trigger = pressure.min(available.max(1));
        if self.context.cost_ceiling_tokens > 0 {
            trigger.min(self.context.cost_ceiling_tokens).max(1)
        } else {
            trigger
        }
    }

    /// Provider-safe input budget. Always leaves room for the requested
    /// output plus protocol headroom, so a request never consumes the whole
    /// model window.
    pub fn effective_hard_input(&self) -> usize {
        let window = self.effective_context_window();
        let output = self.effective_max_output();
        let available = window.saturating_sub(output).max(1);
        window
            .saturating_sub(output)
            .saturating_sub(self.context.headroom_tokens.max(SAFETY_MARGIN_TOKENS))
            .max(window / 2)
            .min(available)
            .max(1)
    }

    /// Effective tail budget after a cutoff. A tail at/above the trigger would
    /// make cutoff advancement a no-op, so cap it below the trigger.
    pub fn effective_retain(&self) -> usize {
        let trigger = self.effective_trigger();
        let raw = if let Some(tokens) = self.context.retain_tokens.filter(|&v| v > 0) {
            tokens.max(1)
        } else {
            let window = self.effective_context_window();
            let output = self.effective_max_output();
            let base = window.saturating_sub(output);
            ((base as f64) * self.effective_retain_ratio().clamp(0.01, 0.9)) as usize
        };
        raw.min(trigger.saturating_sub(1)).max(1)
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
        assert_eq!(
            super::ContextConfig::default().compact_trigger_tokens,
            256000
        );
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
        assert_eq!(
            cfg.skills.user_dir.as_deref(),
            Some("/tmp/aacode_env_skills")
        );
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
        let cfg2: AgentConfig = serde_json::from_str(r#"{"model":{"api_key":"sk-x"}}"#).unwrap();
        assert!(cfg2.validate().is_empty());
    }
    #[test]
    fn effective_context_policy_uses_model_window() {
        let mut cfg = AgentConfig::default();
        cfg.model.context_window = Some(1_000_000);
        cfg.model.max_output_tokens = Some(64_000);
        cfg.context.window_ratio = 0.8;
        cfg.context.headroom_tokens = 65_536;
        cfg.context.retain_ratio = 0.16;
        assert_eq!(cfg.effective_context_window(), 1_000_000);
        assert_eq!(cfg.effective_max_output(), 64_000);
        assert_eq!(cfg.effective_trigger(), 800_000);
        assert_eq!(cfg.effective_retain(), 149_760);
    }

    #[test]
    fn ios_flat_task_json_context_aliases_apply() {
        let v: serde_json::Value = serde_json::json!({
            "model": {"name": "m", "api_key": "k"},
            "context_window": 320000,
            "window_ratio": 0.75,
            "headroom_tokens": 12000,
            "retain_ratio": 0.2,
            "retain_tokens": 9000,
            "cost_ceiling_tokens": 300000,
            "timeout": 777
        });
        let mut cfg: super::AgentConfig = serde_json::from_value(v).unwrap();
        cfg.apply_flat_context();
        assert_eq!(cfg.effective_context_window(), 320000);
        assert_eq!(cfg.effective_trigger(), 240000);
        assert_eq!(cfg.effective_retain(), 9000);
        assert_eq!(cfg.context.cost_ceiling_tokens, 300000);
        assert_eq!(cfg.timeouts.model_request, 777);
    }

    #[test]
    fn flat_context_zero_values_do_not_override_defaults() {
        let v: serde_json::Value = serde_json::json!({
            "model": {"name": "m", "api_key": "k"},
            "context_window": 0,
            "timeout": 0
        });
        let mut cfg: super::AgentConfig = serde_json::from_value(v).unwrap();
        let before = cfg.effective_context_window();
        let timeout_before = cfg.timeouts.model_request;
        cfg.apply_flat_context();
        assert_eq!(cfg.effective_context_window(), before);
        assert_eq!(cfg.timeouts.model_request, timeout_before);
    }

    #[test]
    fn effective_trigger_honours_cost_ceiling() {
        let mut cfg = AgentConfig::default();
        cfg.model.context_window = Some(1_000_000);
        cfg.model.max_output_tokens = Some(8_000);
        cfg.context.window_ratio = 0.8;
        cfg.context.headroom_tokens = 0;
        cfg.context.cost_ceiling_tokens = 100_000;
        assert_eq!(cfg.effective_trigger(), 100_000);
    }

    #[test]
    fn effective_retain_uses_ratio_or_explicit_tokens() {
        let mut cfg = AgentConfig::default();
        cfg.model.context_window = Some(100_000);
        cfg.model.max_output_tokens = Some(8_000);
        cfg.context.retain_ratio = 0.2;
        assert_eq!(cfg.effective_retain(), 18_400);
        cfg.context.retain_tokens = Some(0);
        assert_eq!(
            cfg.effective_retain(),
            18_400,
            "zero retain is treated as unset, not a 1-token budget"
        );
        cfg.context.retain_tokens = Some(7_777);
        assert_eq!(cfg.effective_retain(), 7_777);
    }

    #[test]
    fn invalid_numeric_settings_are_clamped_not_fatal() {
        let mut cfg: AgentConfig = serde_json::from_value(serde_json::json!({
            "model": {
                "name": "bad",
                "api_key": "k",
                "temperature": 99.0,
                "context_window": "1",
                "max_output_tokens": "999999999"
            },
            "context": {
                "max_context_tokens": 1,
                "headroom_tokens": 999999999,
                "window_ratio": -1,
                "retain_ratio": 3
            }
        }))
        .unwrap();
        cfg.apply_compat_defaults();
        let window = cfg.effective_context_window();
        let output = cfg.effective_max_output();
        assert!(window >= MIN_CONTEXT_WINDOW);
        assert!(output >= 1 && output <= FALLBACK_MAX_OUTPUT);
        assert!(cfg.context.headroom_tokens < window);
        assert!(cfg.effective_hard_input() >= 1);
        assert!(cfg.effective_hard_input() < window);
        assert_eq!(cfg.model.temperature, 2.0);
        assert_eq!(cfg.context.window_ratio, 0.8);
        assert_eq!(cfg.context.retain_ratio, 0.16);
        cfg.validate();
    }

    #[test]
    fn legacy_string_optional_numeric_fields_parse() {
        let mut cfg: AgentConfig = serde_json::from_value(serde_json::json!({
            "model": {
                "name": "legacy", "api_key": "k", "max_tokens": "0",
                "context_window": "0", "max_output_tokens": "0"
            },
            "context": {"retain_tokens": "0"},
            "context_window": "0",
            "retain_tokens": "0",
            "tool_profile": null
        }))
        .unwrap();
        cfg.apply_compat_defaults();
        assert_eq!(cfg.tool_profile, "auto");
        cfg.apply_flat_context();
        assert_eq!(cfg.model.max_tokens, 32_000);
        assert_eq!(cfg.model.context_window, None);
        assert_eq!(cfg.context.retain_tokens, None);
        assert_eq!(cfg.effective_context_window(), 1_000_000);
        assert_eq!(
            cfg.effective_retain(),
            (1_000_000usize.saturating_sub(32_000) as f64 * 0.16) as usize
        );
    }

    #[test]
    fn compat_defaults_repair_invalid_ratios_idempotently() {
        let mut cfg = AgentConfig::default();
        cfg.context.window_ratio = f64::NAN;
        cfg.context.retain_ratio = 2.0;
        cfg.context.retain_tokens = Some(0);
        cfg.apply_compat_defaults();
        assert_eq!(cfg.context.window_ratio, 0.8);
        assert_eq!(cfg.context.retain_ratio, 0.16);
        assert_eq!(cfg.context.retain_tokens, None);
        let first = cfg.clone();
        cfg.apply_compat_defaults();
        assert_eq!(cfg.context.window_ratio, first.context.window_ratio);
        assert_eq!(cfg.context.retain_ratio, first.context.retain_ratio);
        assert_eq!(cfg.model.max_tokens, first.model.max_tokens);
    }

    #[test]
    fn legacy_null_numeric_config_fields_parse_and_repair() {
        let mut cfg: AgentConfig = serde_json::from_value(serde_json::json!({
            "model": {"name": "legacy", "api_key": "k", "max_tokens": null},
            "context": {
                "max_context_tokens": null,
                "compact_trigger_tokens": null,
                "window_ratio": null,
                "headroom_tokens": null,
                "retain_ratio": null
            }
        }))
        .unwrap();
        cfg.apply_compat_defaults();
        assert_eq!(cfg.model.max_tokens, 32_000);
        assert_eq!(cfg.effective_context_window(), 1_000_000);
        assert!(cfg.effective_trigger() > 1);
    }

    #[test]
    fn compat_defaults_repair_zero_values_for_llm_request() {
        let mut cfg: AgentConfig = serde_json::from_value(serde_json::json!({
            "model": {"name": "legacy", "api_key": "k", "max_tokens": 0,
                      "context_window": 0, "max_output_tokens": 0},
            "context": {"max_context_tokens": 0, "retain_tokens": 0}
        }))
        .unwrap();
        cfg.apply_compat_defaults();
        assert_eq!(cfg.model.max_tokens, 32_000);
        assert_eq!(cfg.model.context_window, None);
        assert_eq!(cfg.model.max_output_tokens, None);
        assert_eq!(cfg.context.max_context_tokens, 1_000_000);
        assert_eq!(cfg.context.retain_tokens, None);
    }

    #[test]
    fn legacy_zero_model_fields_fall_back_to_safe_values() {
        let cfg: AgentConfig = serde_json::from_value(serde_json::json!({
            "model": {
                "name": "legacy",
                "api_key": "k",
                "context_window": 0,
                "max_output_tokens": 0,
                "max_tokens": 0
            },
            "context": {
                "max_context_tokens": 0,
                "window_ratio": 0,
                "retain_ratio": 0,
                "retain_tokens": 0
            }
        }))
        .unwrap();
        assert_eq!(cfg.effective_context_window(), 1_000_000);
        assert_eq!(cfg.effective_max_output(), 32_000);
        assert!(cfg.effective_trigger() > 1);
        assert!(cfg.effective_retain() > 1);
    }

    #[test]
    fn high_global_max_tokens_is_capped_to_safe_output_default() {
        let cfg: AgentConfig = serde_json::from_value(serde_json::json!({
            "model": {
                "name": "legacy",
                "api_key": "k",
                "context_window": 1_000_000,
                "max_output_tokens": 0,
                "max_tokens": 96000
            }
        }))
        .unwrap();
        assert_eq!(cfg.effective_max_output(), 32_000);
    }

    #[test]
    fn legacy_zero_model_window_uses_context_default() {
        let cfg: AgentConfig = serde_json::from_value(serde_json::json!({
            "model": {"name": "legacy", "api_key": "k", "context_window": 0, "max_output_tokens": 0},
            "context": {"max_context_tokens": 200000}
        }))
        .unwrap();
        assert_eq!(cfg.effective_context_window(), 200000);
        assert_eq!(cfg.effective_max_output(), 32_000);
    }

    #[test]
    fn zero_retain_override_uses_ratio_instead_of_one_token() {
        let mut cfg = AgentConfig::default();
        cfg.model.context_window = Some(100_000);
        cfg.model.max_output_tokens = Some(8_000);
        cfg.context.retain_ratio = 0.2;
        cfg.context.retain_tokens = Some(0);
        assert_eq!(cfg.effective_retain(), 18_400);
    }

    #[test]
    fn context_spill_and_prune_defaults_match_python() {
        let cfg = ContextConfig::default();
        assert_eq!(cfg.inline_tool_tokens, 6_000);
        assert_eq!(cfg.inline_head_chars, 8_000);
        assert_eq!(cfg.inline_tail_chars, 4_000);
        assert!(cfg.prune_enabled);
        assert_eq!(cfg.prune_threshold_chars, 8_192);
        assert_eq!(cfg.prune_head_chars, 4_096);
        assert_eq!(cfg.prune_tail_chars, 1_024);
        assert_eq!(cfg.prune_min_rounds, 4);
    }

    #[test]
    fn timeout_profiles_have_safe_defaults() {
        let cfg = AgentConfig::default();
        assert_eq!(cfg.timeouts.shell_command, 120);
        assert_eq!(cfg.timeouts.shell_idle, 120);
        assert_eq!(cfg.timeouts.shell_foreground_max, 600);
        assert_eq!(cfg.timeouts.shell_background, 1_800);
        assert_eq!(cfg.timeouts.web_request, 60);
        assert_eq!(cfg.timeouts.code_execution, 300);
        assert_eq!(cfg.timeouts.skill_discovery, 5);
        assert_eq!(cfg.timeouts.subagent, 600);
    }

    #[test]
    fn default_tool_profile_is_auto() {
        assert_eq!(AgentConfig::default().tool_profile, "auto");
    }

    #[test]
    fn flat_context_ignores_absent_fields() {
        let mut cfg = AgentConfig::default();
        cfg.context.window_ratio = 0.61;
        cfg.context.retain_ratio = 0.17;
        cfg.apply_flat_context();
        assert_eq!(cfg.context.window_ratio, 0.61);
        assert_eq!(cfg.context.retain_ratio, 0.17);
    }
    #[test]
    fn explicit_trigger_and_retain_are_capped_to_safe_budget() {
        let mut cfg = AgentConfig::default();
        cfg.model.context_window = Some(10_000);
        cfg.model.max_output_tokens = Some(1_000);
        cfg.context.headroom_tokens = 500;
        cfg.context.retrieval_trigger_tokens = 50_000;
        cfg.context.retain_tokens = Some(50_000);
        assert!(cfg.effective_trigger() <= 10_000 - 1_000 - 500);
        assert!(cfg.effective_retain() < cfg.effective_trigger());
    }

    #[test]
    fn cost_ceiling_caps_default_and_explicit_trigger() {
        let mut cfg = AgentConfig::default();
        cfg.context.cost_ceiling_tokens = 12_345;
        assert_eq!(cfg.effective_trigger(), 12_345);
        cfg.context.retrieval_trigger_tokens = 50_000;
        assert_eq!(cfg.effective_trigger(), 12_345);
    }

    #[test]
    fn explicit_retrieval_trigger_wins() {
        let mut cfg = AgentConfig::default();
        cfg.model.context_window = Some(1_000_000);
        cfg.context.retrieval_trigger_tokens = 1_234;
        assert_eq!(cfg.effective_trigger(), 1_234);
    }
}
