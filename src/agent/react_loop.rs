// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! The ReAct loop — iterate model calls + tool execution until completion.
//!
//! Ported from Python `core/react_loop.py` (core control flow). Completion is
//! detected exactly as in Python: the model returns no tool_calls → the final
//! text is the summary and the loop ends.

use crate::agent::compact::estimate_messages_tokens;
use crate::agent::retrieval::RetrievalState;
use crate::config::AgentConfig;
use crate::error::{AacodeError, Result};
use crate::llm::types::{ChatMessage, ToolCall};
use crate::llm::LlmClient;
use crate::session::{SessionManager, SessionMessage};
use crate::stream::{CollectingSink, EventSink};
use crate::tools::ToolRegistry;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::Notify;

/// Outcome of a run.
#[derive(Debug, Clone, PartialEq)]
pub enum RunStatus {
    Completed,
    MaxIterations,
    Cancelled,
    Error(String),
}

impl RunStatus {
    /// Stable wire name for the status (used by the `done` event and the
    /// terminal JSON returned by the FFI `wait`).
    pub fn as_str(&self) -> &'static str {
        match self {
            RunStatus::Completed => "completed",
            RunStatus::MaxIterations => "max_iterations",
            RunStatus::Cancelled => "cancelled",
            RunStatus::Error(_) => "error",
        }
    }
}

pub struct RunResult {
    pub status: RunStatus,
    pub iterations: u32,
    pub final_text: String,
}

impl RunResult {
    /// Serialize the terminal outcome as JSON (shared by `done` enrichment and
    /// the FFI `wait` return value).
    pub fn to_result_json(&self) -> serde_json::Value {
        serde_json::json!({
            "status": self.status.as_str(),
            "iterations": self.iterations,
            "final_text": self.final_text,
        })
    }
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
    pub async fn run(
        &self,
        mut messages: Vec<ChatMessage>,
        session: &mut SessionManager,
        emitter: &dyn EventSink,
        cancel: &AtomicBool,
    ) -> Result<RunResult> {
        let cancel_notify = Arc::new(Notify::new());
        let mut stale = StaleTracker::default();
        let mut repeat_guard = RepeatTracker::default();
        // Retrieval cutoff (replaces summary compaction): stored messages stay
        // complete; only the model view is trimmed. The cutoff is frozen so the
        // request prefix stays byte-stable for provider prompt caching.
        let project_path = session.project_path();
        let mut retrieval = RetrievalState::from_value(session.context_checkpoint());
        let session_id = session.current_session_id.clone().unwrap_or_default();
        // Per-model context policy: the configured trigger is capped by the
        // model window, output cap, and headroom so compaction fires before the
        // provider limit instead of relying on a fixed default.
        let mut context_cfg = self.config.context.clone();
        context_cfg.max_context_tokens = self.config.effective_context_window();
        context_cfg.retrieval_trigger_tokens = self
            .config
            .effective_trigger()
            .min(self.config.effective_hard_input());
        context_cfg.retrieval_keep_tokens = self.config.effective_retain();
        // Usage telemetry (token counts only; never prompt/response text).
        let usage_tracker = crate::usage::UsageTracker::new(&session.project_path());
        let run_uuid = uuid::Uuid::new_v4().simple().to_string();
        let run_id = &run_uuid[..12.min(run_uuid.len())];
        // Track the last response text so MaxIterations can return it.
        let mut last_text = String::new();

        for iteration in 0..self.config.max_iterations {
            if cancel.load(Ordering::SeqCst) {
                emitter.error("cancelled");
                return Ok(RunResult {
                    status: RunStatus::Cancelled,
                    iterations: iteration,
                    final_text: String::new(),
                });
            }

            // Build the model view (does not mutate stored messages). Older
            // history is omitted at a frozen cutoff and stays retrievable via
            // the history tools.
            let (mut view, compacted, mut tokens, checkpoint_changed) =
                retrieval.model_view(&messages, &context_cfg, &project_path, &session_id);
            if checkpoint_changed {
                let _ = session.set_context_checkpoint(retrieval.to_value());
            }
            // Enforce hard context limit: if even the compacted view exceeds
            // max_context_tokens the request would be rejected or silently
            // truncated by the API — fail fast instead.
            if tokens > context_cfg.max_context_tokens {
                return Err(AacodeError::Api(format!(
                    "context too large: {} tokens exceeds limit of {} ({} messages). Try reducing history or restarting the session.",
                    tokens,
                    context_cfg.max_context_tokens,
                    messages.len(),
                )));
            }
            // Only re-sanitize after compaction — uncompacted views were
            // already sanitized in MainAgent (history repair) and the ReAct
            // loop never creates broken tool_calls/tool pairings on its own.
            if compacted {
                crate::agent::sanitize::sanitize_history(&mut view);
            }

            let mut response = self
                .chat_with_retry(&view, emitter, cancel, &cancel_notify)
                .await;
            if let Err(e) = &response {
                if is_context_overflow_error(e) {
                    let reported_limit =
                        context_overflow_limit(e).unwrap_or_else(|| context_cfg.max_context_tokens);
                    let effective_limit = reported_limit.min(context_cfg.max_context_tokens);
                    let safe_budget = effective_limit
                        .saturating_sub(self.config.effective_max_output())
                        .saturating_sub(context_cfg.headroom_tokens)
                        .max(1);
                    let target = safe_budget
                        .min(context_cfg.retrieval_trigger_tokens.max(1))
                        .min(tokens.saturating_mul(3) / 4)
                        .max(1);
                    let mut retry_cfg = context_cfg.clone();
                    retry_cfg.retrieval_trigger_tokens = target;
                    retry_cfg.retrieval_keep_tokens =
                        context_cfg.retrieval_keep_tokens.min(target / 2).max(1);
                    let (retry_view, retry_changed, retry_tokens, _) =
                        retrieval.model_view(&messages, &retry_cfg, &project_path, &session_id);
                    if retry_view.len() < view.len() {
                        if retry_changed {
                            let _ = session.set_context_checkpoint(retrieval.to_value());
                        }
                        emitter.tool_progress(
                            "compacting",
                            "context overflow: retrying with shorter history",
                            0,
                        );
                        response = self
                            .chat_with_retry(&retry_view, emitter, cancel, &cancel_notify)
                            .await;
                        view = retry_view;
                        tokens = retry_tokens;
                    }
                }
            }
            let resp = match response {
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

            // Record provider usage for this call (JSONL; no text). Missing
            // usage (gateways that don't report it) is recorded too, so the
            // estimator can be calibrated against real counts later.
            {
                let gateway = match self.config.model.gateway {
                    crate::config::Gateway::Anthropic => "anthropic",
                    crate::config::Gateway::Openai => "openai",
                };
                let mut event = crate::usage::usage_event(
                    &session_id,
                    run_id,
                    iteration + 1,
                    gateway,
                    &self.config.model.name,
                    tokens,
                    view.len(),
                    compacted,
                    self.config.model.pricing.as_ref(),
                    resp.usage.as_ref(),
                );
                // Request manifest (no prompt text): the actual exposed tool
                // names plus the retrieval cutoff. Makes cache misses and tool
                // drift diagnosable from the usage log alone.
                let tools: Vec<String> = self
                    .native_tools
                    .iter()
                    .filter_map(|t| {
                        t.get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|n| n.as_str())
                            .map(str::to_string)
                    })
                    .collect();
                event["tools"] = serde_json::json!(tools);
                let system_hash = view
                    .first()
                    .map(|m| {
                        let mut h = Sha256::new();
                        h.update(m.content.as_bytes());
                        format!("{:x}", h.finalize())[..16].to_string()
                    })
                    .unwrap_or_default();
                let tools_hash = {
                    let mut h = Sha256::new();
                    if let Ok(raw) = serde_json::to_string(&self.native_tools) {
                        h.update(raw.as_bytes());
                    }
                    format!("{:x}", h.finalize())[..16].to_string()
                };
                event["system_hash"] = serde_json::json!(system_hash);
                event["tools_hash"] = serde_json::json!(tools_hash);
                let context_for_display = resp
                    .usage
                    .as_ref()
                    .map(|u| u.input_tokens as usize)
                    .filter(|v| *v > 0)
                    .unwrap_or(tokens);
                event["context_tokens"] = serde_json::json!(context_for_display);
                event["context_window"] = serde_json::json!(context_cfg.max_context_tokens);
                event["context_trigger_tokens"] = serde_json::json!(context_cfg.max_context_tokens);
                if let Some(cp) = retrieval.checkpoint.as_ref() {
                    event["retrieval_cutoff"] = serde_json::json!(cp.cutoff_index);
                    event["retrieval_stub_chars"] = serde_json::json!(cp.stub.chars().count());
                }
                let _ = session.record_usage(&event);
                if let Some(summary) = session.current_summary() {
                    event["session_usage"] = serde_json::json!({
                        "calls": summary.calls,
                        "cache_hit_rate": summary.cache_hit_rate,
                        "total_usage_tokens": summary.total_usage_tokens,
                        "uncached_input_tokens": summary.uncached_input_tokens,
                        "cache_read_tokens": summary.cache_read_tokens,
                        "cache_write_tokens": summary.cache_write_tokens,
                        "output_tokens": summary.output_tokens,
                        "reasoning_tokens": summary.reasoning_tokens,
                        "cost": summary.cost,
                        "context_tokens": summary.context_tokens,
                        "context_window": summary.context_window,
                        "context_trigger_tokens": summary.context_trigger_tokens,
                    });
                }
                emitter.usage(&event);
                if usage_tracker.record(&event) && usage_tracker.verbose {
                    if let Some(u) = &resp.usage {
                        eprintln!(
                            "📊 usage iter {}: input={} (cache {:.1}%) output={} reasoning={}",
                            iteration + 1,
                            u.input_tokens,
                            u.cache_hit_rate() * 100.0,
                            u.output_tokens,
                            u.reasoning_tokens
                        );
                    }
                }
            }

            last_text = resp.text.clone();

            // Completion: no tool calls.
            if resp.tool_calls.is_empty() {
                // If the response was truncated (max_tokens / length), the
                // model may have intended to output tool calls that got cut
                // off. Inject a continuation so the model can keep going.
                if resp.is_truncated() {
                    let assistant = ChatMessage {
                        role: "assistant".into(),
                        content: last_text.clone(),
                        tool_calls: None,
                        tool_call_id: None,
                        reasoning_content: resp.reasoning_content.clone(),
                    };
                    messages.push(assistant.clone());
                    let _ = session.add_message(SessionMessage::from_chat(&assistant));
                    let cont = ChatMessage::user("continue");
                    messages.push(cont.clone());
                    let _ = session.add_message(SessionMessage::from_chat(&cont));
                    continue;
                }

                // An empty completion is not a successful task. Without this
                // guard, a failed gateway/stream is reported as "completed"
                // with a blank final answer and persisted as an empty turn.
                if last_text.trim().is_empty() {
                    let msg = "provider returned an empty final response (no text, no tool calls)";
                    emitter.error(msg);
                    return Ok(RunResult {
                        status: RunStatus::Error(msg.into()),
                        iterations: iteration + 1,
                        final_text: String::new(),
                    });
                }

                let assistant = ChatMessage {
                    role: "assistant".into(),
                    content: last_text.clone(),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: resp.reasoning_content.clone(),
                };
                messages.push(assistant.clone());
                let _ = session.add_message(SessionMessage::from_chat(&assistant));
                // Flush before the terminal event so a host re-reading the
                // session right after `done` sees the final assistant message.
                let _ = session.flush();
                emitter.done_result(&session_id, "completed", iteration + 1, &last_text);
                return Ok(RunResult {
                    status: RunStatus::Completed,
                    iterations: iteration + 1,
                    final_text: last_text,
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
            let assistant =
                ChatMessage::assistant_with_tools(resp.text.clone(), resp.tool_calls.clone());
            let mut assistant_msg = assistant.clone();
            assistant_msg.reasoning_content = resp.reasoning_content.clone();
            messages.push(assistant_msg.clone());
            let _ = session.add_message(SessionMessage::from_chat(&assistant_msg));

            let repeat_warning = repeat_guard.observe(&resp.tool_calls);

            // Execute tool calls. A batch containing a non-reentrant lane
            // (shell/browser/python/delegate) runs sequentially; otherwise
            // independent calls run concurrently. `join_all` preserves input
            // order, so tool results are always submitted in the model's
            // original call order regardless of completion timing.
            let tool_count = resp.tool_calls.len();
            let has_lane = resp
                .tool_calls
                .iter()
                .any(|tc| self.tools.lane_of(&tc.name).is_some());
            if tool_count > 1 && !has_lane {
                use futures::future::join_all;
                let cancel_arc = Arc::new(AtomicBool::new(false));
                let cancel_ref = cancel as &AtomicBool; // reference from outer scope

                let futures: Vec<_> = resp
                    .tool_calls
                    .iter()
                    .map(|tc| {
                        let name = tc.name.clone();
                        let args = tc.parsed_args();
                        let cancel = cancel_arc.clone();
                        async move {
                            let started = std::time::Instant::now();
                            if cancel_ref.load(Ordering::SeqCst) || cancel.load(Ordering::SeqCst) {
                                return (
                                    tc.id.clone(),
                                    tc.name.clone(),
                                    "cancelled".to_string(),
                                    0u64,
                                );
                            }
                            let obs = self
                                .execute_with_retry(&name, args, emitter, cancel.as_ref())
                                .await;
                            (
                                tc.id.clone(),
                                tc.name.clone(),
                                obs,
                                started.elapsed().as_millis() as u64,
                            )
                        }
                    })
                    .collect();

                let results = join_all(futures).await;

                for (tc_id, tc_name, observation, duration_ms) in results {
                    if cancel.load(Ordering::SeqCst) {
                        emitter.error("cancelled");
                        return Ok(RunResult {
                            status: RunStatus::Cancelled,
                            iterations: iteration + 1,
                            final_text: String::new(),
                        });
                    }
                    if tc_name == "fetch_url" {
                        let url = resp
                            .tool_calls
                            .iter()
                            .find(|tc| tc.id == tc_id)
                            .and_then(|tc| {
                                tc.parsed_args()
                                    .get("url")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string())
                            })
                            .unwrap_or_default();
                        stale.record_fetch(&url, &observation);
                    }
                    let (status, code) = classify_tool_status(&observation);
                    emitter.tool_status(
                        &tc_id,
                        &tc_name,
                        status,
                        duration_ms,
                        code,
                        if code.is_some() {
                            Some(observation.as_str())
                        } else {
                            None
                        },
                    );
                    emitter.seg_observation(&observation, self.config.limits.display_preview_chars);
                    let observation = spill_tool_observation(
                        &project_path,
                        &session_id,
                        &tc_id,
                        observation,
                        &self.config.context,
                    );
                    let tool_msg = ChatMessage::tool_result(tc_id, observation);
                    messages.push(tool_msg.clone());
                    let _ = session.add_message(SessionMessage::from_chat(&tool_msg));
                }
            } else {
                // Single tool: sequential path
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
                    let started = std::time::Instant::now();
                    let observation = self
                        .execute_with_retry(&tc.name, args, emitter, cancel)
                        .await;
                    let duration_ms = started.elapsed().as_millis() as u64;
                    if tc.name == "fetch_url" {
                        let url = tc
                            .parsed_args()
                            .get("url")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        stale.record_fetch(&url, &observation);
                    }
                    let (status, code) = classify_tool_status(&observation);
                    emitter.tool_status(
                        &tc.id,
                        &tc.name,
                        status,
                        duration_ms,
                        code,
                        if code.is_some() {
                            Some(observation.as_str())
                        } else {
                            None
                        },
                    );
                    emitter.seg_observation(&observation, self.config.limits.display_preview_chars);
                    let observation = spill_tool_observation(
                        &project_path,
                        &session_id,
                        &tc.id,
                        observation,
                        &self.config.context,
                    );
                    let tool_msg = ChatMessage::tool_result(tc.id.clone(), observation);
                    messages.push(tool_msg.clone());
                    let _ = session.add_message(SessionMessage::from_chat(&tool_msg));
                }
            }

            // Warn after the tool result pair so the conversation stays valid.
            if let Some(w) = repeat_warning {
                let warn_msg = ChatMessage::system(format!("[SYSTEM WARNING]: {w}"));
                let _ = session.add_message(SessionMessage::from_chat(&warn_msg));
                messages.push(warn_msg);
            }

            // Context growth check (informational; compact view is built each iter).
            let _ = estimate_messages_tokens(&messages);
        }

        let _ = session.flush();
        emitter.done_result(
            &session_id,
            "max_iterations",
            self.config.max_iterations,
            &last_text,
        );
        Ok(RunResult {
            status: RunStatus::MaxIterations,
            iterations: self.config.max_iterations,
            final_text: last_text,
        })
    }

    /// Call the model with retry on transient (network/5xx/timeout) errors.
    async fn chat_with_retry(
        &self,
        view: &[ChatMessage],
        emitter: &dyn EventSink,
        cancel: &AtomicBool,
        cancel_notify: &Notify,
    ) -> Result<crate::llm::types::LlmResponse> {
        let max = self.config.limits.max_retries.max(1);
        let mut last: Option<AacodeError> = None;
        for attempt in 0..max {
            if cancel.load(Ordering::SeqCst) {
                return Err(AacodeError::Cancelled);
            }
            // On retry in pipe mode (mobile FFI), buffer output in a
            // CollectingSink so partial content from a failed stream never
            // leaks to the client (avoids "回退" — segment content shrinking
            // mid-stream). Only replay when the stream succeeds.
            // First attempt still streams directly for real-time UX.
            // TTY mode (CLI) never buffers — it has no segment tracking
            // and stale partial text is harmless.
            let use_buffered = attempt > 0 && !emitter.is_tty();
            // Compute result first — LlmResponse is self-contained (all
            // owned), so the CollectingSink borrow ends before we move it.
            let (result, buffered): (_, Option<CollectingSink>) = if use_buffered {
                let b = CollectingSink::new(false);
                let fut = self.llm.chat_stream(view, &self.native_tools, &b, cancel);
                let r = tokio::select! {
                    r = fut => r,
                    _ = cancel_checker(cancel, cancel_notify) => Err(AacodeError::Cancelled),
                };
                (r, Some(b))
            } else {
                let fut = self
                    .llm
                    .chat_stream(view, &self.native_tools, emitter, cancel);
                let r = tokio::select! {
                    r = fut => r,
                    _ = cancel_checker(cancel, cancel_notify) => Err(AacodeError::Cancelled),
                };
                (r, None)
            };

            match result {
                Ok(r) => {
                    if let Some(collected) = buffered {
                        for line in collected.lines() {
                            emitter.emit_line(&line);
                        }
                    }
                    return Ok(r);
                }
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
                    // exponential backoff: 1s, 2s, 4s, ... with ±25% jitter
                    // so concurrent retries don't form a thundering herd.
                    let delay = 1u64 << attempt.min(4);
                    let jitter = {
                        let nanos = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .subsec_nanos() as u64;
                        let half_range = (delay * 250).max(1); // 250ms * delay
                        nanos % (half_range * 2).max(1)
                    };
                    let actual = if jitter > 0 {
                        (delay * 1000 + jitter - delay * 250).max(100)
                    } else {
                        delay * 1000
                    };
                    tokio::time::sleep(std::time::Duration::from_millis(actual)).await;
                }
            }
        }
        Err(last.unwrap_or_else(|| AacodeError::Api("model call failed".into())))
    }

    /// Execute a tool, retrying transient failures per config.limits.max_retries.
    async fn execute_with_retry(
        &self,
        name: &str,
        args: Value,
        _emitter: &dyn EventSink,
        cancel: &AtomicBool,
    ) -> String {
        // Read-only mode: block mutating/executing tools before any dispatch.
        // The structured observation tells the model why (and is classified as
        // a tool error by `tool_status`).
        if self.config.safety.permission_mode == crate::config::PermissionMode::ReadOnly {
            let target_is_read_only = if name == "call_tool" {
                args.get("name")
                    .and_then(|v| v.as_str())
                    .map(crate::tools::registry::is_read_only)
                    .unwrap_or(false)
            } else {
                crate::tools::registry::is_read_only(name)
            };
            if !target_is_read_only {
                return serde_json::json!({
                    "success": false,
                    "error": format!(
                        "read-only mode: tool '{name}' is not allowed (switch to full access to run it)"
                    ),
                    "tool": name,
                })
                .to_string();
            }
        }
        let max_retries = self.config.limits.max_retries.max(1);
        let mut last = String::new();
        for attempt in 0..max_retries {
            if cancel.load(Ordering::SeqCst) {
                return "cancelled".to_string();
            }
            // Race the tool against cancellation so a long/blocking tool (e.g.
            // a 60s `run_shell`) does not delay a cancel. The tool future is
            // dropped on cancel; a `spawn_blocking` command may finish in the
            // background but the agent stops promptly (and releases the session
            // guard). The local Notify is never notified → 1s polling fallback.
            let notify = Notify::new();
            let obs = tokio::select! {
                obs = self.tools.execute(name, args.clone(), cancel) => obs,
                _ = cancel_checker(cancel, &notify) => return "cancelled".to_string(),
            };
            let lower = obs.to_lowercase();
            let retryable = lower.contains("timeout")
                || lower.contains("timed out")
                || lower.contains("connection")
                || lower.contains("temporary")
                || lower.contains("network");
            last = obs;
            if !retryable || attempt + 1 >= max_retries {
                break;
            }
        }
        last
    }
}

/// Classify a tool observation into a structured `(status, error_code)` pair
/// for the additive `tool_status` event.
fn classify_tool_status(observation: &str) -> (&'static str, Option<&'static str>) {
    // Expected non-zero shell results (grep no match, diff changes, test false)
    // are successful tool executions, even though their process exit is 1.
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(observation) {
        if value.get("expected_nonzero").and_then(|v| v.as_bool()) == Some(true) {
            return ("success", None);
        }
    }
    let l = observation.to_lowercase();
    let failed = l.contains("parameter validation failed")
        || l.contains("execution error")
        || l.contains("execution cancelled")
        || l.contains("unknown tool")
        || observation.starts_with('❌')
        || l.contains("\"success\":false")
        || l.contains("\"success\": false")
        || l.contains("command rejected by safety guard")
        || l.contains("traceback");
    if !failed {
        return ("success", None);
    }
    if l.contains("safety guard") || l.contains("dangerous operation") {
        ("error", Some("safety_guard"))
    } else if l.contains("timed out") || l.contains("timeout") {
        ("error", Some("timeout"))
    } else if l.contains("parameter validation")
        || l.contains("parameter parsing")
        || l.contains("missing required")
    {
        ("error", Some("validation"))
    } else if l.contains("unknown tool") || l.contains("not found") {
        ("error", Some("not_found"))
    } else if l.contains("cancel") {
        ("error", Some("cancelled"))
    } else {
        ("error", Some("tool_error"))
    }
}

/// Whether an API error means the request exceeded the provider context limit.
fn is_context_overflow_error(e: &AacodeError) -> bool {
    let s = e.to_string().to_lowercase();
    s.contains("context_length_exceeded")
        || s.contains("context length")
        || s.contains("context window")
        || s.contains("maximum context")
        || s.contains("prompt is too long")
        || s.contains("input is too long")
        || s.contains("too many tokens")
}

/// Extract the provider-reported context limit from an overflow error, if any.
fn context_overflow_limit(e: &AacodeError) -> Option<usize> {
    let s = e.to_string();
    let lower = s.to_lowercase();
    const MARKERS: [&str; 5] = [
        "maximum context length is ",
        "context length is ",
        "maximum context window is ",
        "context window is ",
        "limit is ",
    ];
    for marker in MARKERS {
        if let Some(pos) = lower.find(marker) {
            let rest = &s[pos + marker.len()..];
            let digits: String = rest
                .chars()
                .skip_while(|c| c.is_whitespace())
                .take_while(|c| c.is_ascii_digit() || *c == ',' || *c == '_')
                .filter(|c| c.is_ascii_digit())
                .collect();
            if let Ok(value) = digits.parse::<usize>() {
                if value > 0 {
                    return Some(value);
                }
            }
        }
    }
    None
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

/// Spill oversized tool observations before they enter model-visible history.
/// Small results are returned unchanged; large results keep a bounded preview
/// plus the full-file path (the model can retrieve the rest with run_shell).
fn spill_tool_observation(
    project_path: &std::path::Path,
    session_id: &str,
    call_id: &str,
    observation: String,
    cfg: &crate::config::ContextConfig,
) -> String {
    if observation.is_empty() || cfg.inline_tool_tokens == 0 {
        return observation;
    }
    if crate::session::estimate_tokens(&observation) <= cfg.inline_tool_tokens {
        return observation;
    }
    let safe = |value: &str, fallback: &str| {
        let cleaned: String = value
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
            .collect();
        if cleaned.is_empty() {
            fallback.to_string()
        } else {
            cleaned
        }
    };
    let session = safe(session_id, "session");
    let call = safe(call_id, &uuid::Uuid::new_v4().simple().to_string()[..12]);
    let relative = format!(".aacode/spill/{session}/{call}.txt");
    let path = project_path.join(&relative);
    if let Some(parent) = path.parent() {
        if std::fs::create_dir_all(parent).is_err() {
            return observation;
        }
    }
    if std::fs::write(&path, observation.as_bytes()).is_err() {
        return observation;
    }
    let total = observation.chars().count();
    let head: String = observation.chars().take(cfg.inline_head_chars).collect();
    let tail: String = observation
        .chars()
        .rev()
        .take(cfg.inline_tail_chars)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    format!(
        "{head}\n\n[... tool result spilled: {total} chars; full output at {relative}. Use run_shell head/tail/sed to inspect it.]\n\n{tail}"
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
        let text: String = observation
            .chars()
            .filter(|c| *c != '<' && *c != '>')
            .collect();
        let has_content = text.trim().len() >= 200;
        let e = self.by_domain.entry(domain).or_default();
        e.push(has_content);
        if e.len() > 5 {
            let drain = e.len() - 5;
            e.drain(0..drain);
        }
    }
}

/// Warns when the model repeats an identical tool call too many times in one
/// run. Mirrors the Python repeat guard at a conservative, advisory threshold.
#[derive(Default)]
struct RepeatTracker {
    counts: HashMap<String, usize>,
    warned: std::collections::HashSet<String>,
}

impl RepeatTracker {
    fn observe(&mut self, tool_calls: &[ToolCall]) -> Option<String> {
        let mut warning = None;
        for tc in tool_calls {
            let args = serde_json::to_string(&tc.parsed_args()).unwrap_or_default();
            let sig = format!("{}\u{0}{args}", tc.name);
            let count = self.counts.entry(sig.clone()).or_insert(0);
            *count += 1;
            if *count >= 3 && self.warned.insert(sig) {
                warning = Some(format!(
                    "The tool call `{}` with identical arguments has repeated {} times. Stop repeating it; verify the result, change the approach, or finish with what you have.",
                    tc.name, count
                ));
            }
        }
        warning
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
    #[async_trait::async_trait]
    impl LlmClient for ScriptedLlm {
        async fn chat_stream(
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
        async fn validate(&self) -> Result<()> {
            Ok(())
        }
    }

    // Scripted LLM that also records every request's messages.
    struct RecordingLlm {
        responses: Mutex<Vec<LlmResponse>>,
        seen: Mutex<Vec<Vec<ChatMessage>>>,
    }
    #[async_trait::async_trait]
    impl LlmClient for RecordingLlm {
        async fn chat_stream(
            &self,
            m: &[ChatMessage],
            _t: &[Value],
            emitter: &dyn EventSink,
            _c: &AtomicBool,
        ) -> Result<LlmResponse> {
            self.seen.lock().unwrap().push(m.to_vec());
            let mut q = self.responses.lock().unwrap();
            if q.is_empty() {
                return Err(AacodeError::Api("no more scripted responses".into()));
            }
            let r = q.remove(0);
            emitter.seg("thought", &r.text);
            Ok(r)
        }
        async fn validate(&self) -> Result<()> {
            Ok(())
        }
    }

    struct EchoTool;
    #[async_trait::async_trait]
    impl Tool for EchoTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema::new(
                "echo",
                "echo",
                vec![ToolParameter::new(
                    "text",
                    ParamType::String,
                    true,
                    "t",
                    &[],
                )],
            )
        }
        async fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
            Ok(format!("echoed: {}", args["text"].as_str().unwrap_or("")))
        }
    }

    struct SleepyTool {
        name: &'static str,
        delay_ms: u64,
        lane: Option<&'static str>,
        log: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    }
    #[async_trait::async_trait]
    impl Tool for SleepyTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema::new(self.name, "sleep", vec![])
        }
        fn lane(&self) -> Option<&'static str> {
            self.lane
        }
        async fn call(&self, _args: &Value, _c: &AtomicBool) -> Result<String> {
            self.log
                .lock()
                .unwrap()
                .push(format!("start:{}", self.name));
            tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
            self.log.lock().unwrap().push(format!("end:{}", self.name));
            Ok(format!("{} done", self.name))
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

    fn named_call(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: "{}".into(),
        }
    }

    #[test]
    fn large_tool_observation_is_spilled_with_preview_and_path() {
        let proj = tmp_proj();
        let mut cfg = crate::config::ContextConfig::default();
        cfg.inline_tool_tokens = 10;
        cfg.inline_head_chars = 20;
        cfg.inline_tail_chars = 10;
        let raw = format!("HEAD{}TAIL", "x".repeat(5000));
        let out = spill_tool_observation(&proj, "s1", "call_1", raw.clone(), &cfg);
        assert!(out.len() < raw.len());
        assert!(out.contains(".aacode/spill/s1/call_1.txt"));
        assert!(out.contains("full output at"));
        let path = proj.join(".aacode/spill/s1/call_1.txt");
        assert_eq!(std::fs::read_to_string(path).unwrap(), raw);
    }

    #[test]
    fn repeat_guard_warns_once_at_third_identical_call() {
        let mut g = RepeatTracker::default();
        let call = tc("read_file", r#"{"path":"a"}"#);
        assert!(g.observe(std::slice::from_ref(&call)).is_none());
        assert!(g.observe(std::slice::from_ref(&call)).is_none());
        assert!(g.observe(std::slice::from_ref(&call)).is_some());
        assert!(g.observe(std::slice::from_ref(&call)).is_none());
        let other = tc("read_file", r#"{"path":"b"}"#);
        assert!(g.observe(&[other]).is_none());
    }

    /// Same-lane tools must not overlap (deterministic event log, no timing).
    struct HugeTool;
    #[async_trait::async_trait]
    impl Tool for HugeTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema::new("huge", "return a large result", vec![])
        }
        async fn call(&self, _args: &Value, _c: &AtomicBool) -> Result<String> {
            Ok("HEAD".to_string() + &"x".repeat(5000) + "TAIL")
        }
    }

    #[tokio::test]
    async fn large_tool_result_is_spilled_before_session_persistence() {
        let llm = ScriptedLlm {
            responses: Mutex::new(vec![
                LlmResponse {
                    tool_calls: vec![tc("huge", "{}")],
                    ..Default::default()
                },
                LlmResponse {
                    text: "done".into(),
                    ..Default::default()
                },
            ]),
        };
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(HugeTool));
        let mut cfg = AgentConfig::default();
        cfg.context.inline_tool_tokens = 10;
        cfg.context.inline_head_chars = 12;
        cfg.context.inline_tail_chars = 8;
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);
        let proj = tmp_proj();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let msgs = vec![ChatMessage::system("sys"), ChatMessage::user("go")];
        let res = loop_.run(msgs, &mut sm, &sink, &cancel).await.unwrap();
        assert_eq!(res.status, RunStatus::Completed);
        let tool = sm
            .messages
            .iter()
            .find(|m| m.role == "tool")
            .expect("tool message");
        assert!(tool.content.contains(".aacode/spill/"));
        assert!(tool.content.contains("full output at"));
        let spill_dir = proj.join(".aacode/spill");
        assert!(spill_dir.join("session").is_dir() || spill_dir.exists());
    }

    #[tokio::test]
    async fn lane_batch_runs_sequentially() {
        let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(SleepyTool {
            name: "a",
            delay_ms: 30,
            lane: Some("x"),
            log: log.clone(),
        }));
        reg.register(Box::new(SleepyTool {
            name: "b",
            delay_ms: 30,
            lane: Some("x"),
            log: log.clone(),
        }));
        let llm = ScriptedLlm {
            responses: Mutex::new(vec![
                LlmResponse {
                    tool_calls: vec![named_call("c1", "a"), named_call("c2", "b")],
                    ..Default::default()
                },
                LlmResponse {
                    text: "done".into(),
                    ..Default::default()
                },
            ]),
        };
        let mut cfg = AgentConfig::default();
        cfg.max_iterations = 3;
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);
        let proj = tmp_proj();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        loop_
            .run(
                vec![ChatMessage::system("s"), ChatMessage::user("u")],
                &mut sm,
                &sink,
                &cancel,
            )
            .await
            .unwrap();

        let events = log.lock().unwrap().clone();
        assert_eq!(events, vec!["start:a", "end:a", "start:b", "end:b"]);
        let ids: Vec<String> = sm
            .messages
            .iter()
            .filter(|m| m.role == "tool")
            .map(|m| m.tool_call_id.clone().unwrap_or_default())
            .collect();
        assert_eq!(ids, vec!["c1".to_string(), "c2".to_string()]);
    }

    /// Independent tools run concurrently, but results are still submitted in
    /// the model's original call order.
    #[tokio::test]
    async fn concurrent_batch_submits_results_in_call_order() {
        let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(SleepyTool {
            name: "slow",
            delay_ms: 80,
            lane: None,
            log: log.clone(),
        }));
        reg.register(Box::new(SleepyTool {
            name: "fast",
            delay_ms: 5,
            lane: None,
            log: log.clone(),
        }));
        let llm = ScriptedLlm {
            responses: Mutex::new(vec![
                LlmResponse {
                    tool_calls: vec![named_call("c1", "slow"), named_call("c2", "fast")],
                    ..Default::default()
                },
                LlmResponse {
                    text: "done".into(),
                    ..Default::default()
                },
            ]),
        };
        let mut cfg = AgentConfig::default();
        cfg.max_iterations = 3;
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);
        let proj = tmp_proj();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        loop_
            .run(
                vec![ChatMessage::system("s"), ChatMessage::user("u")],
                &mut sm,
                &sink,
                &cancel,
            )
            .await
            .unwrap();

        let ids: Vec<String> = sm
            .messages
            .iter()
            .filter(|m| m.role == "tool")
            .map(|m| m.tool_call_id.clone().unwrap_or_default())
            .collect();
        assert_eq!(ids, vec!["c1".to_string(), "c2".to_string()]);
        let events = log.lock().unwrap().clone();
        let end_fast = events.iter().position(|e| e == "end:fast");
        let end_slow = events.iter().position(|e| e == "end:slow");
        assert!(
            end_fast < end_slow,
            "independent tools should overlap: {events:?}"
        );
    }

    /// Read-only mode blocks mutating tools before dispatch but allows reads.
    #[tokio::test]
    async fn read_only_mode_blocks_mutating_tools() {
        let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(SleepyTool {
            name: "run_shell",
            delay_ms: 1,
            lane: None,
            log: log.clone(),
        }));
        reg.register(Box::new(SleepyTool {
            name: "get_page_title",
            delay_ms: 1,
            lane: None,
            log: log.clone(),
        }));
        let llm = ScriptedLlm {
            responses: Mutex::new(vec![
                LlmResponse {
                    tool_calls: vec![
                        named_call("c1", "run_shell"),
                        named_call("c2", "get_page_title"),
                    ],
                    ..Default::default()
                },
                LlmResponse {
                    text: "done".into(),
                    ..Default::default()
                },
            ]),
        };
        let mut cfg = AgentConfig::default();
        cfg.max_iterations = 3;
        cfg.safety.permission_mode = crate::config::PermissionMode::ReadOnly;
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);
        let proj = tmp_proj();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        loop_
            .run(
                vec![ChatMessage::system("s"), ChatMessage::user("u")],
                &mut sm,
                &sink,
                &cancel,
            )
            .await
            .unwrap();

        let events = log.lock().unwrap().clone();
        assert!(
            !events.iter().any(|e| e == "start:run_shell"),
            "mutating tool ran in read-only mode: {events:?}"
        );
        assert!(events.iter().any(|e| e == "start:get_page_title"));
        let blocked = sm
            .messages
            .iter()
            .find(|m| m.role == "tool" && m.tool_call_id.as_deref() == Some("c1"))
            .expect("blocked tool result");
        assert!(
            blocked.content.contains("read-only mode"),
            "{}",
            blocked.content
        );
    }

    #[test]
    fn classifies_tool_status() {
        assert_eq!(super::classify_tool_status("all good"), ("success", None));
        assert_eq!(
            super::classify_tool_status(
                "{\"success\":false,\"hint\":\"command timed out — split it\"}"
            ),
            ("error", Some("timeout"))
        );
        assert_eq!(
            super::classify_tool_status("Parameter validation failed\n\nmissing required"),
            ("error", Some("validation"))
        );
        assert_eq!(
            super::classify_tool_status("Execution error: boom"),
            ("error", Some("tool_error"))
        );
        assert_eq!(
            super::classify_tool_status("Error: Unknown tool 'x'"),
            ("error", Some("not_found"))
        );
        assert_eq!(
            super::classify_tool_status("Execution cancelled"),
            ("error", Some("cancelled"))
        );
        assert_eq!(
            super::classify_tool_status(
                "{\"success\":false,\"error\":\"Command rejected by safety guard\"}"
            ),
            ("error", Some("safety_guard"))
        );
    }

    /// End-to-end: the loop must send a compacted view and persist the
    /// checkpoint, leaving the full message list untouched.
    #[tokio::test]
    async fn retrieval_compacts_view_and_persists_checkpoint() {
        let mut cfg = AgentConfig::default();
        cfg.max_iterations = 2;
        cfg.context.max_context_tokens = 2_000_000;
        cfg.context.retrieval_trigger_tokens = 50;
        cfg.context.retrieval_keep_tokens = 40;
        cfg.context.retrieval_keep_min_rounds = 1;
        cfg.context.retrieval_keep_recent_user_messages = 1;

        let llm = RecordingLlm {
            responses: Mutex::new(vec![LlmResponse {
                text: "done".into(),
                ..Default::default()
            }]),
            seen: Mutex::new(vec![]),
        };
        let reg = ToolRegistry::new();
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);

        let proj = tmp_proj();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("seed", None).unwrap();

        let mut msgs = vec![ChatMessage::system("SYS")];
        for i in 0..20 {
            msgs.push(ChatMessage::user(format!("task {i} {}", "x".repeat(120))));
            msgs.push(ChatMessage::assistant("ok"));
        }

        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let res = loop_
            .run(msgs.clone(), &mut sm, &sink, &cancel)
            .await
            .unwrap();
        assert_eq!(res.status, RunStatus::Completed);

        let seen = llm.seen.lock().unwrap();
        let first = &seen[0];
        assert!(first.len() < msgs.len(), "view should be compacted");
        assert_eq!(first[0].role, "system");
        assert!(
            first[1].content.starts_with("<conversation-checkpoint>"),
            "checkpoint stub missing: {}",
            first[1].content
        );
        let cp = sm.context_checkpoint().expect("checkpoint persisted");
        assert!(cp["cutoff_index"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn completes_when_no_tool_calls() {
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
        let res = loop_.run(msgs, &mut sm, &sink, &cancel).await.unwrap();
        assert_eq!(res.status, RunStatus::Completed);
        assert_eq!(res.final_text, "all done");
        assert!(sink.lines().iter().any(|l| l.contains(r#""type":"done""#)));
    }

    #[tokio::test]
    async fn empty_response_is_error_not_completed() {
        let llm = ScriptedLlm {
            responses: Mutex::new(vec![LlmResponse::default()]),
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
        let res = loop_.run(msgs, &mut sm, &sink, &cancel).await.unwrap();
        match res.status {
            RunStatus::Error(msg) => assert!(msg.contains("empty final response")),
            other => panic!("empty response must not be {other:?}"),
        }
        assert!(!sink
            .lines()
            .iter()
            .any(|l| l.contains(r#""status":"completed""#)));
        assert!(!sm
            .messages
            .iter()
            .any(|m| m.role == "assistant" && m.content.is_empty()));
    }

    #[tokio::test]
    async fn executes_tool_then_completes() {
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
        let res = loop_.run(msgs, &mut sm, &sink, &cancel).await.unwrap();
        assert_eq!(res.status, RunStatus::Completed);
        // observation emitted
        assert!(sink.lines().iter().any(|l| l.contains("echoed: hey")));
        // session persisted assistant(tool_calls) + tool + final assistant
        assert!(sm.messages.iter().any(|m| m.tool_calls.is_some()));
        assert!(sm.messages.iter().any(|m| m.role == "tool"));
    }

    #[tokio::test]
    async fn session_usage_aggregates_from_provider_responses() {
        let usage = crate::llm::Usage {
            input_tokens: 1000,
            output_tokens: 10,
            total_tokens: 1010,
            cache_read_tokens: 900,
            cache_write_tokens: 0,
            cache_miss_tokens: 100,
            reasoning_tokens: 2,
        };
        let llm = ScriptedLlm {
            responses: Mutex::new(vec![
                LlmResponse {
                    tool_calls: vec![tc("echo", "{\"text\":\"one\"}")],
                    usage: Some(usage.clone()),
                    ..Default::default()
                },
                LlmResponse {
                    text: "done".into(),
                    usage: Some(usage),
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
        let msgs = vec![ChatMessage::system("sys"), ChatMessage::user("hi")];
        let res = loop_.run(msgs, &mut sm, &sink, &cancel).await.unwrap();
        assert_eq!(res.status, RunStatus::Completed);

        let lines = sink.lines();
        assert!(
            lines.iter().any(|l| {
                l.contains("\"type\":\"usage\"")
                    && l.contains("\"session_usage\"")
                    && l.contains("\"context_trigger_tokens\"")
            }),
            "live usage event missing: {lines:?}"
        );

        let summary = sm.list_sessions().into_iter().next().unwrap();
        assert_eq!(summary.calls, 2);
        assert_eq!(summary.uncached_input_tokens, 200);
        assert_eq!(summary.cache_read_tokens, 1800);
        assert_eq!(summary.output_tokens, 20);
        assert_eq!(summary.reasoning_tokens, 4);
        assert!((summary.cache_hit_rate - 0.9).abs() < 1e-9);
        assert_eq!(
            summary.context_tokens, 1000,
            "display context must use provider prompt"
        );
        assert_eq!(summary.context_trigger_tokens, summary.context_window);
    }

    #[tokio::test]
    async fn repeated_identical_tool_call_gets_one_warning_after_tool_results() {
        let call = || LlmResponse {
            tool_calls: vec![tc("echo", "{\"text\":\"same\"}")],
            ..Default::default()
        };
        let llm = ScriptedLlm {
            responses: Mutex::new(vec![
                call(),
                call(),
                call(),
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
        let msgs = vec![ChatMessage::system("sys"), ChatMessage::user("hi")];
        let _ = loop_.run(msgs, &mut sm, &sink, &cancel).await.unwrap();

        let warnings: Vec<usize> = sm
            .messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role == "system" && m.content.contains("repeated 3 times"))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(warnings.len(), 1, "warning must be emitted exactly once");
        let last_tool = sm
            .messages
            .iter()
            .rposition(|m| m.role == "tool")
            .expect("tool result persisted");
        assert!(
            warnings[0] > last_tool,
            "warning must come after the tool result, not split the tool pair"
        );
    }

    #[tokio::test]
    async fn persists_turn_messages_without_segment_duplication() {
        let llm = ScriptedLlm {
            responses: Mutex::new(vec![
                LlmResponse {
                    text: "let me echo".into(),
                    tool_calls: vec![tc("echo", "{\"text\":\"hey\"}")],
                    reasoning_content: Some("I need to echo".into()),
                    ..Default::default()
                },
                LlmResponse {
                    text: "all done".into(),
                    reasoning_content: Some("summarize".into()),
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
        let res = loop_.run(msgs, &mut sm, &sink, &cancel).await.unwrap();
        assert_eq!(res.status, RunStatus::Completed);

        // The turn is persisted as plain messages only: assistant(tool_calls)
        // carries reasoning + content (~ thinking/thought/action), the tool
        // message carries the observation — no duplicated render segments.
        let tool_asst = sm
            .messages
            .iter()
            .find(|m| m.tool_calls.is_some())
            .expect("assistant with tool_calls present");
        assert_eq!(
            tool_asst.reasoning_content.as_deref(),
            Some("I need to echo")
        );
        assert_eq!(tool_asst.content, "let me echo");
        assert_eq!(tool_asst.tool_calls.as_ref().unwrap()[0].name, "echo");

        let tool_msg = sm
            .messages
            .iter()
            .find(|m| m.role == "tool")
            .expect("tool result present");
        assert!(tool_msg.content.contains("echoed: hey"));

        let final_asst = sm
            .messages
            .iter()
            .rev()
            .find(|m| m.role == "assistant")
            .expect("final assistant present");
        assert_eq!(final_asst.reasoning_content.as_deref(), Some("summarize"));
        assert_eq!(final_asst.content, "all done");
    }

    /// The exact messages handed to the model must contain each tool call and
    /// each tool result exactly once (no duplication), adjacent, with no missing
    /// result and no stray duplicated system/user rows.
    #[tokio::test]
    async fn llm_request_has_exactly_one_copy_of_each_tool_message() {
        let llm = RecordingLlm {
            responses: Mutex::new(vec![
                LlmResponse {
                    text: "let me echo".into(),
                    tool_calls: vec![tc("echo", "{\"text\":\"hey\"}")],
                    reasoning_content: Some("I need to echo".into()),
                    ..Default::default()
                },
                LlmResponse {
                    text: "all done".into(),
                    reasoning_content: Some("summarize".into()),
                    ..Default::default()
                },
            ]),
            seen: Mutex::new(vec![]),
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
        let res = loop_.run(msgs, &mut sm, &sink, &cancel).await.unwrap();
        assert_eq!(res.status, RunStatus::Completed);

        let seen = llm.seen.lock().unwrap();
        assert_eq!(seen.len(), 2, "one request per model turn");
        let second = &seen[1];

        let with_calls: Vec<&ChatMessage> =
            second.iter().filter(|m| m.tool_calls.is_some()).collect();
        assert_eq!(with_calls.len(), 1, "exactly one assistant tool_calls msg");
        let tool_msgs: Vec<&ChatMessage> = second.iter().filter(|m| m.role == "tool").collect();
        assert_eq!(
            tool_msgs.len(),
            1,
            "exactly one tool result (no duplication)"
        );
        assert_eq!(tool_msgs[0].tool_call_id.as_deref(), Some("c1"));
        assert_eq!(tool_msgs[0].content, "echoed: hey");

        assert_eq!(with_calls[0].content, "let me echo");
        assert_eq!(
            with_calls[0].reasoning_content.as_deref(),
            Some("I need to echo")
        );
        let ai = second.iter().position(|m| m.tool_calls.is_some()).unwrap();
        assert_eq!(second[ai + 1].role, "tool", "result adjacent to its call");

        // No duplicated user/system rows slipped in.
        assert_eq!(second.iter().filter(|m| m.role == "user").count(), 1);
        assert_eq!(second.iter().filter(|m| m.role == "system").count(), 1);

        // Persisted history maps 1:1 (no duplicate rows on disk either).
        assert_eq!(sm.history_chat().len(), sm.messages.len());
    }

    #[tokio::test]
    async fn respects_cancel_before_start() {
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
            .await
            .unwrap();
        assert_eq!(res.status, RunStatus::Cancelled);
    }

    #[tokio::test]
    async fn max_iterations_reached() {
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
            .await
            .unwrap();
        assert_eq!(res.status, RunStatus::MaxIterations);
        assert_eq!(res.iterations, 3);
    }

    #[tokio::test]
    async fn domain_parse() {
        assert_eq!(domain_of("https://example.com/path"), "example.com");
        assert_eq!(domain_of("http://a.b.c/x/y"), "a.b.c");
    }

    #[test]
    fn context_overflow_errors_are_detected() {
        assert!(is_context_overflow_error(&AacodeError::Api(
            "context_length_exceeded: maximum context length is 1000000".into()
        )));
        assert!(is_context_overflow_error(&AacodeError::Api(
            "prompt is too long".into()
        )));
        assert!(!is_context_overflow_error(&AacodeError::Api(
            "HTTP 401 unauthorized".into()
        )));
        assert_eq!(
            context_overflow_limit(&AacodeError::Api(
                "context_length_exceeded: maximum context length is 128,000 tokens".into()
            )),
            Some(128_000)
        );
        assert_eq!(
            context_overflow_limit(&AacodeError::Api(
                "input is too long; limit is 1000000".into()
            )),
            Some(1_000_000)
        );
    }

    #[tokio::test]
    async fn classify_errors() {
        assert!(
            classify_api_error(&AacodeError::Api("HTTP 401 no".into())).contains("authentication")
        );
        assert!(classify_api_error(&AacodeError::Api("rate limit".into())).contains("quota"));
        assert!(classify_api_error(&AacodeError::Network("reset".into())).contains("Network"));
        assert!(classify_api_error(&AacodeError::Config("no key".into())).contains("Configuration"));
    }

    // An LLM that fails N times with a retryable error, then succeeds.
    struct FlakyLlm {
        remaining_failures: Mutex<u32>,
    }
    #[async_trait::async_trait]
    impl LlmClient for FlakyLlm {
        async fn chat_stream(
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
        async fn validate(&self) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn retries_transient_llm_errors() {
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
            .await
            .unwrap();
        assert_eq!(res.status, RunStatus::Completed);
        assert_eq!(res.final_text, "recovered");
    }

    #[tokio::test]
    async fn truncated_response_not_completed() {
        // Truncated response with no tool calls → injects "continue" →
        // needs another LLM response. We provide a chain so the loop can
        // exhaust iterations without crashing on empty ScriptedLlm.
        let responses: Vec<LlmResponse> = (0..5)
            .map(|_| LlmResponse {
                text: "partial".into(),
                tool_calls: vec![],
                finish_reason: Some("length".to_string()),
                ..Default::default()
            })
            .collect();
        let llm = ScriptedLlm {
            responses: Mutex::new(responses),
        };
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(EchoTool));
        let mut cfg = AgentConfig::default();
        cfg.max_iterations = 5;
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);
        let proj = tmp_proj();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let res = loop_
            .run(vec![ChatMessage::user("x")], &mut sm, &sink, &cancel)
            .await
            .unwrap();
        // Truncated response with no tool calls should NOT complete — it
        // should loop until max_iterations and then return MaxIterations.
        assert_eq!(
            res.status,
            RunStatus::MaxIterations,
            "truncated response must not be treated as completed"
        );
    }

    #[tokio::test]
    async fn context_overflow_retries_with_shorter_history() {
        let llm = ContextOverflowThenOkLlm {
            calls: Mutex::new(vec![]),
            systems: Mutex::new(vec![]),
        };
        let reg = ToolRegistry::new();
        let cfg = AgentConfig::default();
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);
        let proj = tmp_proj();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        let mut messages = vec![ChatMessage::system("SYS")];
        for i in 0..30 {
            messages.push(ChatMessage::user(format!("task {i} {}", "x".repeat(800))));
            messages.push(ChatMessage::assistant("ok"));
        }
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let res = loop_.run(messages, &mut sm, &sink, &cancel).await.unwrap();
        assert_eq!(res.status, RunStatus::Completed);
        let calls = llm.calls.lock().unwrap();
        assert_eq!(calls.len(), 2, "must retry exactly once after overflow");
        assert!(calls[1] < calls[0], "retry view must be shorter");
        let systems = llm.systems.lock().unwrap();
        assert_eq!(systems.len(), 2);
        assert_eq!(systems[0], "SYS", "original system prompt changed");
        assert_eq!(
            systems[1], "SYS",
            "retry must preserve cached system prefix"
        );
    }

    #[tokio::test]
    async fn max_iterations_returns_last_text() {
        let responses: Vec<LlmResponse> = (0..5)
            .map(|i| LlmResponse {
                text: format!("iter {}", i),
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
        cfg.max_iterations = 2;
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);
        let proj = tmp_proj();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let res = loop_
            .run(vec![ChatMessage::user("x")], &mut sm, &sink, &cancel)
            .await
            .unwrap();
        assert_eq!(res.status, RunStatus::MaxIterations);
        assert!(
            !res.final_text.is_empty(),
            "max_iterations must return last response text"
        );
    }

    // ── LLMs for retry / buffering tests ──────────────────────────────

    /// Fails N times with a retryable error, then succeeds with stream events.
    struct FailThenStreamLlm {
        remaining_failures: Mutex<u32>,
    }
    #[async_trait::async_trait]
    impl LlmClient for FailThenStreamLlm {
        async fn chat_stream(
            &self,
            _m: &[ChatMessage],
            _t: &[Value],
            emitter: &dyn EventSink,
            _c: &AtomicBool,
        ) -> Result<LlmResponse> {
            let mut n = self.remaining_failures.lock().unwrap();
            if *n > 0 {
                *n -= 1;
                return Err(AacodeError::Network("temporary reset".into()));
            }
            emitter.delta("thought", "streaming");
            emitter.seg("thinking", "reasoning done");
            emitter.seg("thought", "complete response");
            Ok(LlmResponse {
                text: "complete response".into(),
                ..Default::default()
            })
        }
        async fn validate(&self) -> Result<()> {
            Ok(())
        }
    }

    struct ContextOverflowThenOkLlm {
        calls: Mutex<Vec<usize>>,
        systems: Mutex<Vec<String>>,
    }
    #[async_trait::async_trait]
    impl LlmClient for ContextOverflowThenOkLlm {
        async fn chat_stream(
            &self,
            m: &[ChatMessage],
            _t: &[Value],
            _e: &dyn EventSink,
            _c: &AtomicBool,
        ) -> Result<LlmResponse> {
            let mut calls = self.calls.lock().unwrap();
            calls.push(m.len());
            self.systems
                .lock()
                .unwrap()
                .push(m.first().map(|x| x.content.clone()).unwrap_or_default());
            if calls.len() == 1 {
                return Err(AacodeError::Api(
                    "context_length_exceeded: maximum context length is 1000000".into(),
                ));
            }
            Ok(LlmResponse {
                text: "recovered".into(),
                ..Default::default()
            })
        }
        async fn validate(&self) -> Result<()> {
            Ok(())
        }
    }

    /// Emits events on attempt 0 then fails, succeeds on attempt 1 with
    /// different content. Simulates a mid-stream disconnect followed by
    /// a successful retry.
    struct EmitThenFailLlm {
        call: Mutex<u32>,
    }
    #[async_trait::async_trait]
    impl LlmClient for EmitThenFailLlm {
        async fn chat_stream(
            &self,
            _m: &[ChatMessage],
            _t: &[Value],
            emitter: &dyn EventSink,
            _c: &AtomicBool,
        ) -> Result<LlmResponse> {
            let mut call = self.call.lock().unwrap();
            *call += 1;
            if *call == 1 {
                // Simulate partial stream output before a mid-stream error.
                emitter.delta("thought", "partial tok");
                emitter.seg("thinking", "partial reasoning");
                emitter.seg("thought", "partial text");
                return Err(AacodeError::Network("mid-stream disconnect".into()));
            }
            // Retry: completely different (correct) content.
            emitter.delta("thought", "complete");
            emitter.seg("thinking", "full reasoning");
            emitter.seg("thought", "final complete text");
            Ok(LlmResponse {
                text: "final complete text".into(),
                ..Default::default()
            })
        }
        async fn validate(&self) -> Result<()> {
            Ok(())
        }
    }

    /// Fails every call (never succeeds). Used for exhausted-retry tests.
    struct AlwaysFailLlm;
    #[async_trait::async_trait]
    impl LlmClient for AlwaysFailLlm {
        async fn chat_stream(
            &self,
            _m: &[ChatMessage],
            _t: &[Value],
            _emitter: &dyn EventSink,
            _c: &AtomicBool,
        ) -> Result<LlmResponse> {
            Err(AacodeError::Network("always fails".into()))
        }
        async fn validate(&self) -> Result<()> {
            Ok(())
        }
    }

    /// A tool that always returns "timeout" observation — triggers retry.
    struct TimeoutTool;
    #[async_trait::async_trait]
    impl Tool for TimeoutTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema::new(
                "timeout_cmd",
                "timeout",
                vec![ToolParameter::new("x", ParamType::String, true, "x", &[])],
            )
        }
        async fn call(&self, _args: &Value, _c: &AtomicBool) -> Result<String> {
            Ok("command timed out after 10s".to_string())
        }
    }

    /// Counts how many times its tool was invoked.
    struct CountingTool {
        count: std::sync::Arc<Mutex<u32>>,
        tool_name: &'static str,
        response: String,
    }
    impl CountingTool {
        fn new(name: &'static str, response: String) -> (Self, std::sync::Arc<Mutex<u32>>) {
            let count = std::sync::Arc::new(Mutex::new(0u32));
            (
                CountingTool {
                    count: count.clone(),
                    tool_name: name,
                    response,
                },
                count,
            )
        }
    }
    #[async_trait::async_trait]
    impl Tool for CountingTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema::new(self.tool_name, self.tool_name, vec![])
        }
        async fn call(&self, _args: &Value, _c: &AtomicBool) -> Result<String> {
            *self.count.lock().unwrap() += 1;
            Ok(self.response.clone())
        }
    }

    #[tokio::test]
    async fn execute_with_retry_retries_on_network() {
        let mut reg = ToolRegistry::new();
        let (tool, counter) = CountingTool::new(
            "understand_image",
            r#"{"success":false,"error":"network error: vision: error sending request for url (https://api.moonshot.cn/v1/chat/completions)"}"#.to_string(),
        );
        reg.register(Box::new(tool));
        let mut cfg = AgentConfig::default();
        cfg.limits.max_retries = 3;
        cfg.max_iterations = 1;
        let responses = vec![LlmResponse {
            text: "try".into(),
            tool_calls: vec![tc(
                "understand_image",
                r#"{"image_path":"a.jpg","prompt":"desc"}"#,
            )],
            ..Default::default()
        }];
        let llm = ScriptedLlm {
            responses: Mutex::new(responses),
        };
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);
        let proj = tmp_proj();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let _ = loop_
            .run(vec![ChatMessage::user("x")], &mut sm, &sink, &cancel)
            .await;
        let invocations = *counter.lock().unwrap();
        assert!(
            invocations >= 2,
            "network error must trigger retry, got {invocations} invocations"
        );
    }

    #[tokio::test]
    async fn execute_with_retry_does_not_retry_on_config_error() {
        let mut reg = ToolRegistry::new();
        let (tool, counter) = CountingTool::new(
            "understand_image",
            r#"{"success":false,"error":"config error: no multimodal model configured"}"#
                .to_string(),
        );
        reg.register(Box::new(tool));
        let mut cfg = AgentConfig::default();
        cfg.limits.max_retries = 3;
        cfg.max_iterations = 1;
        let responses = vec![LlmResponse {
            text: "try".into(),
            tool_calls: vec![tc(
                "understand_image",
                r#"{"image_path":"a.jpg","prompt":"desc"}"#,
            )],
            ..Default::default()
        }];
        let llm = ScriptedLlm {
            responses: Mutex::new(responses),
        };
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);
        let proj = tmp_proj();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let _ = loop_
            .run(vec![ChatMessage::user("x")], &mut sm, &sink, &cancel)
            .await;
        assert_eq!(
            *counter.lock().unwrap(),
            1,
            "config error must not trigger retry"
        );
    }

    #[tokio::test]
    async fn execute_with_retry_timeout_retried() {
        let mut reg = ToolRegistry::new();
        let (tool, counter) =
            CountingTool::new("timeout_cmd", "command timed out after 10s".to_string());
        reg.register(Box::new(tool));
        let mut cfg = AgentConfig::default();
        cfg.limits.max_retries = 3;
        cfg.max_iterations = 1;
        let responses = vec![LlmResponse {
            text: "try".into(),
            tool_calls: vec![tc("timeout_cmd", r#"{"x":"1"}"#)],
            ..Default::default()
        }];
        let llm = ScriptedLlm {
            responses: Mutex::new(responses),
        };
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);
        let proj = tmp_proj();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let _ = loop_
            .run(vec![ChatMessage::user("x")], &mut sm, &sink, &cancel)
            .await;
        assert!(*counter.lock().unwrap() >= 2, "timeout must trigger retry");
    }

    #[tokio::test]
    async fn execute_with_retry_connection_retried() {
        let mut reg = ToolRegistry::new();
        let (tool, counter) = CountingTool::new("fetch_url", "connection reset".to_string());
        reg.register(Box::new(tool));
        let mut cfg = AgentConfig::default();
        cfg.limits.max_retries = 3;
        cfg.max_iterations = 1;
        let responses = vec![LlmResponse {
            text: "try".into(),
            tool_calls: vec![tc("fetch_url", r#"{"url":"http://x"}"#)],
            ..Default::default()
        }];
        let llm = ScriptedLlm {
            responses: Mutex::new(responses),
        };
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);
        let proj = tmp_proj();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let _ = loop_
            .run(vec![ChatMessage::user("x")], &mut sm, &sink, &cancel)
            .await;
        assert!(
            *counter.lock().unwrap() >= 2,
            "connection must trigger retry"
        );
    }

    #[tokio::test]
    async fn execute_with_retry_respects_cancel() {
        let responses = vec![LlmResponse {
            text: "try".into(),
            tool_calls: vec![tc("timeout_cmd", "{\"x\":\"1\"}")],
            ..Default::default()
        }];
        let llm = ScriptedLlm {
            responses: Mutex::new(responses),
        };
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(TimeoutTool));
        let mut cfg = AgentConfig::default();
        cfg.limits.max_retries = 10;
        cfg.max_iterations = 5;
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);
        let proj = tmp_proj();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(true); // pre-set cancel
        let res = loop_
            .run(vec![ChatMessage::user("x")], &mut sm, &sink, &cancel)
            .await
            .unwrap();
        // Must not retry forever — cancel check in first iteration returns
        // Cancelled before tool execution even starts.
        assert_eq!(res.status, RunStatus::Cancelled);
    }

    #[tokio::test]
    async fn max_context_tokens_enforced_and_errors() {
        let llm = ScriptedLlm {
            responses: Mutex::new(vec![LlmResponse {
                text: "done".into(),
                ..Default::default()
            }]),
        };
        let reg = ToolRegistry::new();
        let mut cfg = AgentConfig::default();
        cfg.max_iterations = 5;
        cfg.context.max_context_tokens = 10;
        cfg.context.compact_trigger_tokens = 5;
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);

        let proj = tmp_proj();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);

        let msgs = vec![
            ChatMessage::system("you are a helpful assistant who provides detailed responses"),
            ChatMessage::user("do something"),
        ];
        let res = loop_.run(msgs, &mut sm, &sink, &cancel).await;
        assert!(
            res.is_err(),
            "context over max must error, not silently proceed"
        );
        let msg = format!("{}", res.err().unwrap());
        assert!(
            msg.contains("context too large") || msg.contains("exceeds limit"),
            "error must mention context size: {msg}"
        );
    }

    // ── retry buffering tests ────────────────────────────────────────

    #[tokio::test]
    async fn pipe_mode_retry_buffers_and_replays() {
        // FlakyLlm fails once then returns Ok (no events emitted).
        // With buffering, the emitter must only see output from the
        // retry attempt — nothing leaked from the failed first attempt.
        let llm = FailThenStreamLlm {
            remaining_failures: Mutex::new(1),
        };
        let reg = ToolRegistry::new();
        let mut cfg = AgentConfig::default();
        cfg.limits.max_retries = 3;
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);

        let proj = tmp_proj();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        // Pipe mode (non-TTY) — buffering should engage on retry.
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);

        let res = loop_
            .run(vec![ChatMessage::user("x")], &mut sm, &sink, &cancel)
            .await
            .unwrap();
        assert_eq!(res.status, RunStatus::Completed);
        assert_eq!(res.final_text, "complete response");

        let lines = sink.lines();
        // The retry emitted delta + seg events. Verify they are replayed.
        assert!(
            lines
                .iter()
                .any(|l| l.contains(r#""type":"delta""#) && l.contains("streaming")),
            "delta event must be replayed to real sink"
        );
        assert!(
            lines
                .iter()
                .any(|l| l.contains(r#""type":"seg_content""#) && l.contains("complete response")),
            "seg_content event must be replayed to real sink"
        );
    }

    #[tokio::test]
    async fn tty_mode_never_buffers_retries() {
        // In TTY mode, attempts > 0 still use the real emitter directly.
        let llm = FailThenStreamLlm {
            remaining_failures: Mutex::new(1),
        };
        let reg = ToolRegistry::new();
        let mut cfg = AgentConfig::default();
        cfg.limits.max_retries = 3;
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);

        let proj = tmp_proj();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        // TTY mode — buffering should NOT engage.
        let sink = CollectingSink::new(true);
        let cancel = AtomicBool::new(false);

        let res = loop_
            .run(vec![ChatMessage::user("x")], &mut sm, &sink, &cancel)
            .await
            .unwrap();
        assert_eq!(res.status, RunStatus::Completed);

        // The delta("thought", ...) goes through token_line → emit_line in
        // TTY mode. The seg("thought", ...) default impl emits JSONL
        // regardless of TTY flag on CollectingSink. So we should find
        // both raw text and JSONL in the output.
        let lines = sink.lines();
        // In TTY mode, delta emits raw text via token_line.
        assert!(
            lines.iter().any(|l| l == "streaming"),
            "TTY delta must emit raw text: {lines:?}"
        );
    }

    #[tokio::test]
    async fn multiple_retries_only_last_success_replayed() {
        // Fail twice, succeed on third. Each failure must be discarded;
        // only the successful attempt's events reach the real emitter.
        let llm = FailThenStreamLlm {
            remaining_failures: Mutex::new(2),
        };
        let reg = ToolRegistry::new();
        let mut cfg = AgentConfig::default();
        cfg.limits.max_retries = 4;
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);

        let proj = tmp_proj();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);

        let res = loop_
            .run(vec![ChatMessage::user("x")], &mut sm, &sink, &cancel)
            .await
            .unwrap();
        assert_eq!(res.status, RunStatus::Completed);

        let lines = sink.lines();
        // Each retry's buffer was discarded on failure. Only the final
        // success should appear. The exact content comes from attempt 2
        // (index 2, the third call).
        let seg_count = lines
            .iter()
            .filter(|l| l.contains(r#""type":"seg_content""#) && l.contains("complete response"))
            .count();
        assert_eq!(
            seg_count, 1,
            "only one seg_content for the final thought; found {seg_count}. lines: {lines:?}"
        );
    }

    #[tokio::test]
    async fn first_attempt_streams_directly_leak_is_allowed() {
        // Attempt 0 emits events DIRECTLY to real emitter then fails.
        // Attempt 1 (buffered) retries. The real emitter will contain
        // BOTH the leaked attempt-0 events AND the replayed attempt-1
        // events. This is by design: real-time streaming on first
        // attempt trades perfect isolation for live UI feedback.
        let llm = EmitThenFailLlm {
            call: Mutex::new(0),
        };
        let reg = ToolRegistry::new();
        let mut cfg = AgentConfig::default();
        cfg.limits.max_retries = 3;
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);

        let proj = tmp_proj();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);

        let res = loop_
            .run(vec![ChatMessage::user("x")], &mut sm, &sink, &cancel)
            .await
            .unwrap();
        assert_eq!(res.status, RunStatus::Completed);
        assert_eq!(res.final_text, "final complete text");

        let lines = sink.lines();
        // Attempt 0 leaked "partial text" directly to the real emitter.
        assert!(
            lines.iter().any(|l| l.contains("partial text")),
            "first attempt leaked: {lines:?}"
        );
        // Attempt 1 replayed "final complete text" to the real emitter.
        assert!(
            lines.iter().any(|l| l.contains("final complete text")),
            "retry must replay: {lines:?}"
        );
        // Both sets appear, but retry content is the authoritative final
        // seg_content that the iOS side should use.
    }

    #[tokio::test]
    async fn buffered_retry_replays_all_event_types() {
        // Verify that the buffered CollectingSink captures and replays
        // all event types: delta, seg_content (thinking + thought).
        let llm = FailThenStreamLlm {
            remaining_failures: Mutex::new(1),
        };
        let reg = ToolRegistry::new();
        let mut cfg = AgentConfig::default();
        cfg.limits.max_retries = 3;
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);

        let proj = tmp_proj();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);

        let res = loop_
            .run(vec![ChatMessage::user("x")], &mut sm, &sink, &cancel)
            .await
            .unwrap();
        assert_eq!(res.status, RunStatus::Completed);

        let lines = sink.lines();
        // Delta event replayed.
        assert!(
            lines
                .iter()
                .any(|l| l.contains(r#""type":"delta""#) && l.contains("streaming")),
            "delta missing: {lines:?}"
        );
        // Thinking seg_content replayed.
        assert!(
            lines.iter().any(|l| l.contains(r#""type":"seg_content""#) && l.contains(r#""seg":"thinking""#)),
            "thinking seg missing: {lines:?}"
        );
        // Thought seg_content replayed.
        assert!(
            lines
                .iter()
                .any(|l| l.contains(r#""type":"seg_content""#) && l.contains(r#""seg":"thought""#)),
            "thought seg missing: {lines:?}"
        );
        // done event from the react_loop (emitted after chat_with_retry
        // returns, not from the LLM).
        assert!(
            lines.iter().any(|l| l.contains(r#""type":"done""#)),
            "done event missing: {lines:?}"
        );
    }

    #[tokio::test]
    async fn buffered_retry_respects_cancel() {
        // Pre-set cancel: the cancel check at the top of each iteration
        // (in ReactLoop::run) catches it before chat_with_retry is called.
        let llm = FailThenStreamLlm {
            remaining_failures: Mutex::new(0),
        };
        let reg = ToolRegistry::new();
        let mut cfg = AgentConfig::default();
        cfg.limits.max_retries = 3;
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);

        let proj = tmp_proj();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(true);

        let res = loop_
            .run(vec![ChatMessage::user("x")], &mut sm, &sink, &cancel)
            .await
            .unwrap();
        assert_eq!(res.status, RunStatus::Cancelled);
    }

    #[tokio::test]
    async fn cancel_during_retry_backoff_stops_cleanly() {
        use std::sync::Arc;
        let llm = FailThenStreamLlm {
            remaining_failures: Mutex::new(2),
        };
        let reg = ToolRegistry::new();
        let mut cfg = AgentConfig::default();
        cfg.limits.max_retries = 5;
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);

        let proj = tmp_proj();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        let sink = CollectingSink::new(false);
        let cancel = Arc::new(AtomicBool::new(false));

        let cancel2 = Arc::clone(&cancel);
        std::thread::spawn(move || {
            // Wait for the first failure + backoff sleep to start.
            std::thread::sleep(std::time::Duration::from_millis(500));
            cancel2.store(true, Ordering::SeqCst);
        });

        let res = loop_
            .run(
                vec![ChatMessage::user("x")],
                &mut sm,
                &sink,
                cancel.as_ref(),
            )
            .await
            .unwrap();
        assert_eq!(
            res.status,
            RunStatus::Cancelled,
            "cancel during retry backoff must return Cancelled"
        );
    }

    #[tokio::test]
    async fn exhausted_retries_buffers_discarded_no_events() {
        // When all retries fail, no buffered content should reach the
        // real emitter (each buffer was discarded on failure).
        let llm = AlwaysFailLlm;
        let reg = ToolRegistry::new();
        let mut cfg = AgentConfig::default();
        cfg.limits.max_retries = 3;
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);

        let proj = tmp_proj();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);

        let res = loop_
            .run(vec![ChatMessage::user("x")], &mut sm, &sink, &cancel)
            .await
            .unwrap();
        assert!(matches!(res.status, RunStatus::Error(_)));
        // The react_loop emits an error event on the real emitter.
        let lines = sink.lines();
        assert!(
            lines.iter().any(|l| l.contains(r#""type":"error""#)),
            "error event must be emitted: {lines:?}"
        );
        // But there should be no delta/seg_content from retries.
        let deltas = lines
            .iter()
            .filter(|l| l.contains(r#""type":"delta""#))
            .count();
        let segs = lines
            .iter()
            .filter(|l| l.contains(r#""type":"seg_content""#))
            .count();
        assert_eq!(deltas, 0, "no delta events from failed retries: {lines:?}");
        assert_eq!(
            segs, 0,
            "no seg_content events from failed retries: {lines:?}"
        );
    }

    /// A tool that never returns (like a long blocking `run_shell`).
    struct BlockingTool;
    #[async_trait::async_trait]
    impl Tool for BlockingTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema::new("blocking_tool", "never returns", vec![])
        }
        async fn call(&self, _args: &Value, _c: &AtomicBool) -> Result<String> {
            futures::future::pending::<()>().await;
            Ok(String::new())
        }
    }

    /// Cancel must interrupt a blocking tool promptly (the loop must not wait
    /// for it), which also lets the session guard be released.
    #[tokio::test]
    async fn cancel_interrupts_a_blocking_tool() {
        let llm = ScriptedLlm {
            responses: Mutex::new(vec![LlmResponse {
                text: "work".into(),
                tool_calls: vec![tc("blocking_tool", "{}")],
                ..Default::default()
            }]),
        };
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(BlockingTool));
        let cfg = AgentConfig::default();
        let loop_ = ReactLoop::new(&llm, &reg, &cfg, vec![]);
        let proj = tmp_proj();
        let mut sm = SessionManager::new(&proj);
        sm.create_session("t", None).unwrap();
        let sink = CollectingSink::new(false);
        let cancel = Arc::new(AtomicBool::new(false));
        let c2 = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(200));
            c2.store(true, Ordering::SeqCst);
        });
        let start = std::time::Instant::now();
        let res = loop_
            .run(
                vec![ChatMessage::user("x")],
                &mut sm,
                &sink,
                cancel.as_ref(),
            )
            .await
            .unwrap();
        assert!(
            matches!(res.status, RunStatus::Cancelled),
            "expected Cancelled, got {:?}",
            res.status
        );
        assert!(
            start.elapsed() < std::time::Duration::from_secs(3),
            "cancel should be prompt, took {:?}",
            start.elapsed()
        );
    }
}

/// Returns a future that completes when the cancel flag is set.
/// Uses Notify for immediate wakeup, with 1s polling fallback.
async fn cancel_checker(cancel: &AtomicBool, notify: &Notify) {
    use std::time::Duration;
    loop {
        tokio::select! {
            _ = notify.notified() => {},
            _ = tokio::time::sleep(Duration::from_secs(1)) => {},
        }
        if cancel.load(Ordering::SeqCst) {
            return;
        }
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
