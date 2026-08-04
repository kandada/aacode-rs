// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! The ReAct loop — iterate model calls + tool execution until completion.
//!
//! Ported from Python `core/react_loop.py` (core control flow). Completion is
//! detected exactly as in Python: the model returns no tool_calls → the final
//! text is the summary and the loop ends.

use crate::agent::compact::{build_compact_view_cached, estimate_messages_tokens, CompactCache};
use crate::config::AgentConfig;
use crate::error::{AacodeError, Result};
use crate::llm::types::{ChatMessage, ToolCall};
use crate::llm::LlmClient;
use crate::session::{SessionManager, SessionMessage};
use crate::stream::EventSink;
use crate::tools::ToolRegistry;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

/// Outcome of a run.
#[derive(Debug, Clone, PartialEq)]
pub enum RunStatus {
    Completed,
    MaxIterations,
    Cancelled,
    Error(String),
}

pub struct RunResult {
    pub status: RunStatus,
    pub iterations: u32,
    pub final_text: String,
}

/// The ReAct loop driver. Borrows the collaborators for the duration of a run.
pub struct ReactLoop<'a> {
    pub llm: &'a dyn LlmClient,
    pub tools: &'a ToolRegistry,
    pub config: &'a AgentConfig,
    /// Preformatted native tools for the active gateway.
    pub native_tools: Vec<Value>,
}

impl<'a> ReactLoop<'a> {
    pub fn new(
        llm: &'a dyn LlmClient,
        tools: &'a ToolRegistry,
        config: &'a AgentConfig,
        native_tools: Vec<Value>,
    ) -> Self {
        ReactLoop {
            llm,
            tools,
            config,
            native_tools,
        }
    }

    /// Run the loop. `messages` starts with the system prompt + history + task.
    /// New messages are persisted to `session` incrementally.
    pub fn run(
        &self,
        mut messages: Vec<ChatMessage>,
        session: &mut SessionManager,
        emitter: &dyn EventSink,
        cancel: &AtomicBool,
    ) -> Result<RunResult> {
        let mut stale = StaleTracker::default();
        // Frozen compaction state: keeps the request prefix byte-stable
        // across iterations so the provider KV cache keeps hitting.
        let mut compact_cache: Option<CompactCache> = None;
        let session_id = session.current_session_id.clone().unwrap_or_default();

        for iteration in 0..self.config.max_iterations {
            if cancel.load(Ordering::SeqCst) {
                emitter.error("cancelled");
                return Ok(RunResult {
                    status: RunStatus::Cancelled,
                    iterations: iteration,
                    final_text: String::new(),
                });
            }

            // Build compact view for the model call (does not mutate messages).
            let (mut view, _compacted, _tokens) =
                build_compact_view_cached(&messages, &self.config.context, &mut compact_cache);
            // Last line of defense: never send a broken tool_calls pairing.
            crate::agent::sanitize::sanitize_history(&mut view);

            let resp = match self.chat_with_retry(&view, emitter, cancel) {
                Ok(r) => r,
                Err(AacodeError::Cancelled) => {
                    emitter.error("cancelled");
                    return Ok(RunResult {
                        status: RunStatus::Cancelled,
                        iterations: iteration,
                        final_text: String::new(),
                    });
                }
                Err(e) => {
                    let classified = classify_api_error(&e);
                    emitter.error(&classified);
                    return Ok(RunResult {
                        status: RunStatus::Error(classified),
                        iterations: iteration,
                        final_text: String::new(),
                    });
                }
            };

            // Completion: no tool calls.
            if resp.tool_calls.is_empty() {
                let assistant = ChatMessage {
                    role: "assistant".into(),
                    content: resp.text.clone(),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: resp.reasoning_content.clone(),
                };
                messages.push(assistant.clone());
                let _ = session.add_message(SessionMessage::from_chat(&assistant));
                emitter.done(&session_id);
                return Ok(RunResult {
                    status: RunStatus::Completed,
                    iterations: iteration + 1,
                    final_text: resp.text,
                });
            }

            // Stale-loop detection warning (fetch_url stale domains).
            // Persisted so cross-execute() prefix stays stable.
            if let Some(w) = stale.detect(&resp.tool_calls) {
                let warn_msg = ChatMessage::system(format!("[SYSTEM WARNING]: {w}"));
                let _ = session.add_message(SessionMessage::from_chat(&warn_msg));
                messages.push(warn_msg);
            }

            // Append the assistant message carrying tool_calls.
            let assistant = ChatMessage::assistant_with_tools(resp.text.clone(), resp.tool_calls.clone());
            let mut assistant_msg = assistant.clone();
            assistant_msg.reasoning_content = resp.reasoning_content.clone();
            messages.push(assistant_msg.clone());
            let _ = session.add_message(SessionMessage::from_chat(&assistant_msg));

            // Execute each tool call.
            for tc in &resp.tool_calls {
                if cancel.load(Ordering::SeqCst) {
                    emitter.error("cancelled");
                    return Ok(RunResult {
                        status: RunStatus::Cancelled,
                        iterations: iteration + 1,
                        final_text: String::new(),
                    });
                }
                let args = tc.parsed_args();
                let observation = self.execute_with_retry(&tc.name, args, emitter, cancel);
                // Record for stale detection (fetch_url).
                if tc.name == "fetch_url" {
                    let url = tc
                        .parsed_args()
                        .get("url")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    stale.record_fetch(&url, &observation);
                }
                // Emit observation segment (preview) + persist full result.
                emitter.seg("observation", &preview(&observation, self.config.limits.display_preview_chars));
                let tool_msg = ChatMessage::tool_result(tc.id.clone(), observation);
                messages.push(tool_msg.clone());
                let _ = session.add_message(SessionMessage::from_chat(&tool_msg));
            }

            // Context growth check (informational; compact view is built each iter).
            let _ = estimate_messages_tokens(&messages);
        }

        emitter.done(&session_id);
        Ok(RunResult {
            status: RunStatus::MaxIterations,
            iterations: self.config.max_iterations,
            final_text: String::new(),
        })
    }

    /// Call the model with retry on transient (network/5xx/timeout) errors.
    fn chat_with_retry(
        &self,
        view: &[ChatMessage],
        emitter: &dyn EventSink,
        cancel: &AtomicBool,
    ) -> Result<crate::llm::types::LlmResponse> {
        let max = self.config.limits.max_retries.max(1);
        let mut last: Option<AacodeError> = None;
        for attempt in 0..max {
            if cancel.load(Ordering::SeqCst) {
                return Err(AacodeError::Cancelled);
            }
            match self.llm.chat_stream(view, &self.native_tools, emitter, cancel) {
                Ok(r) => return Ok(r),
                Err(AacodeError::Cancelled) => return Err(AacodeError::Cancelled),
                Err(e) => {
                    let retryable = e.is_retryable();
                    // Surface the retry to the UI (tool_progress renders as a
                    // live status line) — a silent backoff looks like a hang,
                    // especially on flaky mobile networks.
                    if retryable && attempt + 1 < max {
                        emitter.tool_progress(
                            "building",
                            &format!("llm retry {}/{} ({})", attempt + 2, max, brief(&e)),
                            0,
                        );
                    }
                    last = Some(e);
                    if !retryable || attempt + 1 >= max {
                        break;
                    }
                    // exponential backoff: 1s, 2s, 4s, ...
                    let delay = 1u64 << attempt.min(4);
                    std::thread::sleep(std::time::Duration::from_secs(delay));
                }
            }
        }
        Err(last.unwrap_or_else(|| AacodeError::Api("model call failed".into())))
    }

    /// Execute a tool, retrying transient failures per config.limits.max_retries.
    fn execute_with_retry(
        &self,
        name: &str,
        args: Value,
        _emitter: &dyn EventSink,
        cancel: &AtomicBool,
    ) -> String {
        let max_retries = self.config.limits.max_retries.max(1);
        let mut last = String::new();
        for attempt in 0..max_retries {
            let obs = self.tools.execute(name, args.clone(), cancel);
            let lower = obs.to_lowercase();
            let retryable = lower.contains("timeout")
                || lower.contains("connection")
                || lower.contains("temporary");
            last = obs;
            if !retryable || attempt + 1 >= max_retries {
                break;
            }
        }
        last
    }
}

/// Classify an LLM/API error into a friendly, actionable message.
fn classify_api_error(e: &AacodeError) -> String {
    let s = e.to_string();
    let l = s.to_lowercase();
    if l.contains("401") || l.contains("authentication") || l.contains("invalid api key") {
        format!("API authentication failed (check LLM_API_KEY / endpoint): {s}")
    } else if l.contains("quota") || l.contains("rate limit") || l.contains("429") {
        format!("API quota/rate limit reached (check usage or wait): {s}")
    } else if l.contains("connection") || l.contains("timeout") || l.contains("network") {
        format!("Network error reaching the model (check connectivity/endpoint): {s}")
    } else if l.contains("config") || l.contains("api key not configured") {
        format!("Configuration error: {s}")
    } else {
        s
    }
}

/// Truncate an observation to a preview length for display (agent still gets full).
fn preview(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let head: String = text.chars().take(max).collect();
    format!(
        "{head}...\n\n(Display truncated, {} chars total. Agent received full content.)",
        text.chars().count()
    )
}

/// Tracks fetch_url results per domain to detect stale loops (mirrors Python).
#[derive(Default)]
struct StaleTracker {
    by_domain: HashMap<String, Vec<bool>>, // has_content flags
    warned: std::collections::HashSet<String>,
}

impl StaleTracker {
    fn detect(&mut self, tool_calls: &[ToolCall]) -> Option<String> {
        // Only meaningful when the model is issuing fetch_url calls.
        if !tool_calls.iter().any(|tc| tc.name == "fetch_url") {
            return None;
        }
        // Find a domain with 3 recent no-content fetches that we haven't warned about.
        let mut candidate: Option<String> = None;
        for (domain, entries) in &self.by_domain {
            if entries.len() < 3 {
                continue;
            }
            let last3 = &entries[entries.len() - 3..];
            if last3.iter().all(|has| !has) {
                let key = format!("stale_{domain}");
                if !self.warned.contains(&key) {
                    candidate = Some(domain.clone());
                    break;
                }
            }
        }
        let domain = candidate?;
        self.warned.insert(format!("stale_{domain}"));
        Some(format!(
            "Last 3 fetch_url calls to '{domain}' returned unreadable content. Try search_web or proceed with existing knowledge."
        ))
    }

    fn record_fetch(&mut self, url: &str, observation: &str) {
        let domain = domain_of(url);
        // strip tags, check readable length
        let text: String = observation.chars().filter(|c| *c != '<' && *c != '>').collect();
        let has_content = text.trim().len() >= 200;
        let e = self.by_domain.entry(domain).or_default();
        e.push(has_content);
        if e.len() > 5 {
            let drain = e.len() - 5;
            e.drain(0..drain);
        }
    }
}

fn domain_of(url: &str) -> String {
    // naive scheme://host/... parse
    let after_scheme = url.split("://").nth(1).unwrap_or(url);
    after_scheme
        .split('/')
        .next()
        .unwrap_or(after_scheme)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AgentConfig;
    use crate::llm::types::LlmResponse;
    use crate::stream::CollectingSink;
    use crate::tools::registry::Tool;
    use crate::tools::schema::{ParamType, ToolParameter, ToolSchema};
    use std::sync::Mutex;

    // A scripted LLM that returns a queue of responses.
    struct ScriptedLlm {
        responses: Mutex<Vec<LlmResponse>>,
    }
    impl LlmClient for ScriptedLlm {
        fn chat_stream(
            &self,
            _m: &[ChatMessage],
            _t: &[Value],
            emitter: &dyn EventSink,
            _c: &AtomicBool,
        ) -> Result<LlmResponse> {
            let mut q = self.responses.lock().unwrap();
            if q.is_empty() {
                return Err(AacodeError::Api("no more scripted responses".into()));
            }
            let r = q.remove(0);
            emitter.seg("thought", &r.text);
            Ok(r)
        }
        fn validate(&self) -> Result<()> {
            Ok(())
        }
    }

    struct EchoTool;
    impl Tool for EchoTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema::new(
                "echo",
                "echo",
                vec![ToolParameter::new("text", ParamType::String, true, "t", &[])],
            )
        }
        fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
            Ok(format!("echoed: {}", args["text"].as_str().unwrap_or("")))
        }
    }

    fn tmp_proj() -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "aacode_react_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn tc(name: &str, args: &str) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: name.into(),
            arguments: args.into(),
        }
    }

    #[test]
    fn completes_when_no_tool_calls() {
        let llm = ScriptedLlm {
            responses: Mutex::new(vec![LlmResponse {
                text: "all done".into(),
                ..Default::default()
            }]),
        };
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(EchoTool));
        let cfg = AgentConfig::default();
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);

        let proj = tmp_proj();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);

        let msgs = vec![ChatMessage::system("sys"), ChatMessage::user("hi")];
        let res = loop_.run(msgs, &mut sm, &sink, &cancel).unwrap();
        assert_eq!(res.status, RunStatus::Completed);
        assert_eq!(res.final_text, "all done");
        assert!(sink.lines().iter().any(|l| l.contains(r#""type":"done""#)));
    }

    #[test]
    fn executes_tool_then_completes() {
        let llm = ScriptedLlm {
            responses: Mutex::new(vec![
                LlmResponse {
                    text: "let me echo".into(),
                    tool_calls: vec![tc("echo", "{\"text\":\"hey\"}")],
                    ..Default::default()
                },
                LlmResponse {
                    text: "done".into(),
                    ..Default::default()
                },
            ]),
        };
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(EchoTool));
        let cfg = AgentConfig::default();
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);

        let proj = tmp_proj();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);

        let msgs = vec![ChatMessage::system("sys"), ChatMessage::user("echo hey")];
        let res = loop_.run(msgs, &mut sm, &sink, &cancel).unwrap();
        assert_eq!(res.status, RunStatus::Completed);
        // observation emitted
        assert!(sink.lines().iter().any(|l| l.contains("echoed: hey")));
        // session persisted assistant(tool_calls) + tool + final assistant
        assert!(sm.messages.iter().any(|m| m.tool_calls.is_some()));
        assert!(sm.messages.iter().any(|m| m.role == "tool"));
    }

    #[test]
    fn respects_cancel_before_start() {
        let llm = ScriptedLlm {
            responses: Mutex::new(vec![]),
        };
        let reg = ToolRegistry::new();
        let cfg = AgentConfig::default();
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);
        let proj = tmp_proj();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(true);
        let res = loop_
            .run(vec![ChatMessage::user("x")], &mut sm, &sink, &cancel)
            .unwrap();
        assert_eq!(res.status, RunStatus::Cancelled);
    }

    #[test]
    fn max_iterations_reached() {
        // Always returns a tool call → never completes.
        let responses: Vec<LlmResponse> = (0..5)
            .map(|_| LlmResponse {
                text: "again".into(),
                tool_calls: vec![tc("echo", "{\"text\":\"x\"}")],
                ..Default::default()
            })
            .collect();
        let llm = ScriptedLlm {
            responses: Mutex::new(responses),
        };
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(EchoTool));
        let mut cfg = AgentConfig::default();
        cfg.max_iterations = 3;
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);
        let proj = tmp_proj();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let res = loop_
            .run(vec![ChatMessage::user("x")], &mut sm, &sink, &cancel)
            .unwrap();
        assert_eq!(res.status, RunStatus::MaxIterations);
        assert_eq!(res.iterations, 3);
    }

    #[test]
    fn domain_parse() {
        assert_eq!(domain_of("https://example.com/path"), "example.com");
        assert_eq!(domain_of("http://a.b.c/x/y"), "a.b.c");
    }

    #[test]
    fn classify_errors() {
        assert!(classify_api_error(&AacodeError::Api("HTTP 401 no".into())).contains("authentication"));
        assert!(classify_api_error(&AacodeError::Api("rate limit".into())).contains("quota"));
        assert!(classify_api_error(&AacodeError::Network("reset".into())).contains("Network"));
        assert!(classify_api_error(&AacodeError::Config("no key".into())).contains("Configuration"));
    }

    // An LLM that fails N times with a retryable error, then succeeds.
    struct FlakyLlm {
        remaining_failures: Mutex<u32>,
    }
    impl LlmClient for FlakyLlm {
        fn chat_stream(
            &self,
            _m: &[ChatMessage],
            _t: &[Value],
            _e: &dyn EventSink,
            _c: &AtomicBool,
        ) -> Result<LlmResponse> {
            let mut n = self.remaining_failures.lock().unwrap();
            if *n > 0 {
                *n -= 1;
                return Err(AacodeError::Network("temporary reset".into()));
            }
            Ok(LlmResponse {
                text: "recovered".into(),
                ..Default::default()
            })
        }
        fn validate(&self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn retries_transient_llm_errors() {
        let llm = FlakyLlm {
            remaining_failures: Mutex::new(1),
        };
        let reg = ToolRegistry::new();
        let mut cfg = AgentConfig::default();
        cfg.limits.max_retries = 3;
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        // Directly exercise chat_with_retry via a completing run.
        let proj = tmp_proj();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        let res = loop_
            .run(vec![ChatMessage::user("x")], &mut sm, &sink, &cancel)
            .unwrap();
        assert_eq!(res.status, RunStatus::Completed);
        assert_eq!(res.final_text, "recovered");
    }
}

/// Short one-line error description for UI status lines.
fn brief(e: &crate::error::AacodeError) -> String {
    let s = e.to_string();
    let first = s.lines().next().unwrap_or(&s);
    let mut out: String = first.chars().take(60).collect();
    if first.chars().count() > 60 {
        out.push('…');
    }
    out
}
