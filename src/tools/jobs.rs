// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Background shell jobs.
//!
//! `run_shell` waits for completion (bounded by its timeout). These tools let a
//! long command (build, test suite, dev server) run in the background while the
//! agent keeps reasoning:
//!
//!   job_start → job_status → job_collect → job_cancel / job_list
//!
//! Execution uses the existing [`ShellBackend`] (native shell on desktop,
//! fastshell sandbox on mobile), so no new execution domain is introduced.
//! Jobs are **process-local**: a restart loses them (status is reported as
//! unknown), and cancellation is best-effort (the backend has no kill handle);
//! completed results are still collected exactly once per call (idempotent —
//! collecting never re-runs the command).

use super::backend::ShellBackend;
use super::registry::Tool;
use super::schema::{ParamType, ToolParameter, ToolSchema};
use crate::error::{AacodeError, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Collect cap (chars per stream) unless `full_output` is set.
pub const DEFAULT_COLLECT_CHARS: usize = 20_000;
/// Maximum number of retained jobs (running + recently finished).
pub const MAX_JOBS: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl JobStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            JobStatus::Running => "running",
            JobStatus::Completed => "completed",
            JobStatus::Failed => "failed",
            JobStatus::Cancelled => "cancelled",
        }
    }
}

struct JobRecord {
    status: JobStatus,
    stdout: String,
    stderr: String,
    exit_code: i32,
    started: Instant,
    finished: Option<Instant>,
    cancel: Arc<AtomicBool>,
}

#[derive(Debug, Clone)]
pub struct JobSnapshot {
    pub id: String,
    pub status: &'static str,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub elapsed_ms: u128,
    pub cancel_requested: bool,
}

#[derive(Default)]
pub struct JobStore {
    inner: Arc<Mutex<HashMap<String, JobRecord>>>,
    max_jobs: usize,
}

impl JobStore {
    pub fn new(max_jobs: usize) -> Self {
        JobStore {
            inner: Arc::new(Mutex::new(HashMap::new())),
            max_jobs: max_jobs.max(1),
        }
    }

    /// Spawn a background command. Returns the job id.
    pub fn start(
        &self,
        backend: Arc<dyn ShellBackend>,
        command: String,
        cwd: PathBuf,
        timeout_secs: u64,
        idle_timeout_secs: u64,
    ) -> std::result::Result<String, String> {
        let id = format!("job_{}", &uuid::Uuid::new_v4().simple().to_string()[..8]);
        let cancel = Arc::new(AtomicBool::new(false));
        {
            let mut jobs = self.inner.lock().unwrap();
            // Keep the store bounded: forget finished jobs older than 10 min.
            let now = Instant::now();
            jobs.retain(|_, r| {
                r.status == JobStatus::Running
                    || r.finished
                        .map(|f| now.duration_since(f).as_secs() < 600)
                        .unwrap_or(true)
            });
            if jobs.len() >= self.max_jobs {
                return Err(format!(
                    "too many jobs ({} max); collect or wait for one to finish",
                    self.max_jobs
                ));
            }
            jobs.insert(
                id.clone(),
                JobRecord {
                    status: JobStatus::Running,
                    stdout: String::new(),
                    stderr: String::new(),
                    exit_code: i32::MIN,
                    started: Instant::now(),
                    finished: None,
                    cancel: cancel.clone(),
                },
            );
        }

        let store = self.inner.clone();
        let id2 = id.clone();
        tokio::spawn(async move {
            let cmd2 = command.clone();
            let cwd2 = cwd.clone();
            let res = tokio::task::spawn_blocking(move || {
                backend.run(&cmd2, None, timeout_secs, idle_timeout_secs, &cwd2)
            })
            .await;
            let mut jobs = store.lock().unwrap();
            if let Some(record) = jobs.get_mut(&id2) {
                match res {
                    Ok(out) => {
                        record.stdout = out.stdout;
                        record.stderr = out.stderr;
                        record.exit_code = out.exit_code;
                        record.status = if cancel.load(Ordering::SeqCst) {
                            JobStatus::Cancelled
                        } else if out.exit_code == 0 {
                            JobStatus::Completed
                        } else {
                            JobStatus::Failed
                        };
                    }
                    Err(e) => {
                        record.stderr = format!("job join error: {e}");
                        record.status = if cancel.load(Ordering::SeqCst) {
                            JobStatus::Cancelled
                        } else {
                            JobStatus::Failed
                        };
                        record.exit_code = -1;
                    }
                }
                record.finished = Some(Instant::now());
            }
        });
        Ok(id)
    }

    pub fn status(&self, id: &str) -> Option<JobSnapshot> {
        let jobs = self.inner.lock().unwrap();
        jobs.get(id).map(|r| self.snapshot(id, r))
    }

    pub fn cancel(&self, id: &str) -> bool {
        let jobs = self.inner.lock().unwrap();
        match jobs.get(id) {
            Some(r) => {
                r.cancel.store(true, Ordering::SeqCst);
                true
            }
            None => false,
        }
    }

    pub fn list(&self) -> Vec<JobSnapshot> {
        let jobs = self.inner.lock().unwrap();
        let mut out: Vec<JobSnapshot> = jobs.iter().map(|(id, r)| self.snapshot(id, r)).collect();
        out.sort_by(|a, b| a.id.cmp(&b.id));
        out
    }

    fn snapshot(&self, id: &str, r: &JobRecord) -> JobSnapshot {
        JobSnapshot {
            id: id.to_string(),
            status: r.status.as_str(),
            stdout: r.stdout.clone(),
            stderr: r.stderr.clone(),
            exit_code: r.exit_code,
            elapsed_ms: r
                .finished
                .unwrap_or_else(Instant::now)
                .duration_since(r.started)
                .as_millis(),
            cancel_requested: r.cancel.load(Ordering::SeqCst),
        }
    }
}

fn truncate_chars(s: &str, max: usize) -> (String, bool, usize) {
    let total = s.chars().count();
    if max == 0 || total <= max {
        return (s.to_string(), false, total);
    }
    (s.chars().take(max).collect(), true, total)
}

// ── tools ──────────────────────────────────────────────────────────────────

pub struct JobStartTool {
    pub backend: Arc<dyn ShellBackend>,
    pub project_path: PathBuf,
    pub timeout_secs: u64,
    pub idle_timeout_secs: u64,
    pub store: Arc<JobStore>,
}

#[async_trait::async_trait]
impl Tool for JobStartTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "job_start",
            "Start a shell command in the background (build, test suite, server). \
             Returns a job_id immediately; poll job_status/job_collect. The job runs \
             through the same sandbox/shell backend as run_shell. Process-local: a \
             restart loses the job.",
            vec![
                ToolParameter::new(
                    "command",
                    ParamType::String,
                    true,
                    "Shell command to run in the background.",
                    &["cmd"],
                ),
                ToolParameter::new(
                    "timeout",
                    ParamType::Integer,
                    false,
                    "Total timeout in seconds (default: the configured shell timeout).",
                    &["time_limit"],
                ),
            ],
        )
    }

    fn lane(&self) -> Option<&'static str> {
        Some("shell")
    }

    async fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let command = args
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if command.is_empty() {
            return Err(AacodeError::Tool("job_start requires 'command'".into()));
        }
        let timeout = args
            .get("timeout")
            .and_then(|v| v.as_u64())
            .unwrap_or(self.timeout_secs);
        let id = self
            .store
            .start(
                self.backend.clone(),
                command,
                self.project_path.clone(),
                timeout,
                self.idle_timeout_secs,
            )
            .map_err(AacodeError::Tool)?;
        Ok(json!({"job_id": id, "status": "running"}).to_string())
    }
}

pub struct JobStatusTool {
    pub store: Arc<JobStore>,
}

#[async_trait::async_trait]
impl Tool for JobStatusTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "job_status",
            "Return a background job's status (running|completed|failed|cancelled), \
             exit code, elapsed time and cancel flag.",
            vec![ToolParameter::new(
                "job_id",
                ParamType::String,
                true,
                "Job id returned by job_start.",
                &["id", "job"],
            )],
        )
    }

    async fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let id = args.get("job_id").and_then(|v| v.as_str()).unwrap_or("");
        match self.store.status(id) {
            Some(s) => Ok(json!({
                "job_id": s.id,
                "status": s.status,
                "exit_code": if s.exit_code == i32::MIN { Value::Null } else { json!(s.exit_code) },
                "elapsed_ms": s.elapsed_ms,
                "cancel_requested": s.cancel_requested,
            })
            .to_string()),
            None => Err(AacodeError::Tool(format!("unknown job '{id}'"))),
        }
    }
}

pub struct JobCollectTool {
    pub store: Arc<JobStore>,
    pub project_path: PathBuf,
}

#[async_trait::async_trait]
impl Tool for JobCollectTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "job_collect",
            "Collect a background job's output. Running jobs return status only; finished \
             jobs return bounded stdout/stderr plus available/truncated flags. Full output is \
             archived under .aacode/extracts. Repeated collection never re-runs the command.",
            vec![
                ToolParameter::new(
                    "job_id",
                    ParamType::String,
                    true,
                    "Job id returned by job_start.",
                    &["id", "job"],
                ),
                ToolParameter::new(
                    "max_chars",
                    ParamType::Integer,
                    false,
                    "Max chars per stream (default 20000).",
                    &["max_output"],
                ),
                ToolParameter::new(
                    "full_output",
                    ParamType::Boolean,
                    false,
                    "Return the full stdout/stderr (no cap).",
                    &["full", "unlimited"],
                ),
            ],
        )
    }

    async fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let id = args.get("job_id").and_then(|v| v.as_str()).unwrap_or("");
        let full = args
            .get("full_output")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let max = args
            .get("max_chars")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(DEFAULT_COLLECT_CHARS);
        let Some(s) = self.store.status(id) else {
            return Err(AacodeError::Tool(format!("unknown job '{id}'")));
        };
        if s.status == "running" {
            return Ok(json!({
                "job_id": s.id,
                "status": s.status,
                "stdout": "",
                "stderr": "",
                "available": false,
            })
            .to_string());
        }
        let (stdout, out_trunc, out_total) = if full {
            (s.stdout.clone(), false, s.stdout.chars().count())
        } else {
            truncate_chars(&s.stdout, max)
        };
        let (stderr, err_trunc, err_total) = if full {
            (s.stderr.clone(), false, s.stderr.chars().count())
        } else {
            truncate_chars(&s.stderr, max)
        };
        let mut out = json!({
            "job_id": s.id,
            "status": s.status,
            "exit_code": if s.exit_code == i32::MIN { Value::Null } else { json!(s.exit_code) },
            "elapsed_ms": s.elapsed_ms,
            "stdout": stdout,
            "stderr": stderr,
            "available": true,
            "stdout_truncated": out_trunc,
            "stderr_truncated": err_trunc,
            "stdout_chars": out_total,
            "stderr_chars": err_total,
        });
        if out_trunc || err_trunc {
            // Archive the full streams for later inspection.
            let dir = self.project_path.join(".aacode").join("extracts");
            if std::fs::create_dir_all(&dir).is_ok() {
                let out_path = dir.join(format!("{}_stdout.txt", s.id));
                let err_path = dir.join(format!("{}_stderr.txt", s.id));
                if std::fs::write(&out_path, &s.stdout).is_ok() {
                    out["stdout_archive"] = json!(out_path.to_string_lossy());
                }
                if !s.stderr.is_empty() && std::fs::write(&err_path, &s.stderr).is_ok() {
                    out["stderr_archive"] = json!(err_path.to_string_lossy());
                }
            }
        }
        Ok(out.to_string())
    }
}

pub struct JobCancelTool {
    pub store: Arc<JobStore>,
}

#[async_trait::async_trait]
impl Tool for JobCancelTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "job_cancel",
            "Request cancellation of a background job. This records intent; the process \
             may keep running until it exits (no forced kill), so effects are not rolled back.",
            vec![ToolParameter::new(
                "job_id",
                ParamType::String,
                true,
                "Job id returned by job_start.",
                &["id", "job"],
            )],
        )
    }

    async fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let id = args.get("job_id").and_then(|v| v.as_str()).unwrap_or("");
        if self.store.cancel(id) {
            Ok(json!({"job_id": id, "cancel_requested": true}).to_string())
        } else {
            Err(AacodeError::Tool(format!("unknown job '{id}'")))
        }
    }
}

pub struct JobListTool {
    pub store: Arc<JobStore>,
}

#[async_trait::async_trait]
impl Tool for JobListTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "job_list",
            "List background jobs with status, elapsed time and cancel flag.",
            vec![],
        )
    }

    async fn call(&self, _args: &Value, _c: &AtomicBool) -> Result<String> {
        let jobs: Vec<Value> = self
            .store
            .list()
            .into_iter()
            .map(|s| {
                json!({
                    "job_id": s.id,
                    "status": s.status,
                    "exit_code": if s.exit_code == i32::MIN { Value::Null } else { json!(s.exit_code) },
                    "elapsed_ms": s.elapsed_ms,
                    "cancel_requested": s.cancel_requested,
                })
            })
            .collect();
        Ok(json!({"jobs": jobs, "count": jobs.len()}).to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::backend::NativeShell;

    fn tmp() -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "aacode_jobs_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    async fn wait_done(store: &JobStore, id: &str) -> JobSnapshot {
        for _ in 0..200 {
            let s = store.status(id).unwrap();
            if s.status != "running" {
                return s;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        panic!("job did not finish");
    }

    #[tokio::test]
    async fn start_status_collect_roundtrip() {
        let dir = tmp();
        let store = Arc::new(JobStore::new(MAX_JOBS));
        let tool = JobStartTool {
            backend: Arc::new(NativeShell::new()),
            project_path: dir.clone(),
            timeout_secs: 30,
            idle_timeout_secs: 30,
            store: store.clone(),
        };
        let out = tool
            .call(
                &json!({"command": "printf 'job-ok'"}),
                &AtomicBool::new(false),
            )
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        let id = v["job_id"].as_str().unwrap().to_string();
        let done = wait_done(&store, &id).await;
        assert_eq!(done.status, "completed", "{:?}", done);
        assert_eq!(done.exit_code, 0);

        let collect = JobCollectTool {
            store: store.clone(),
            project_path: dir,
        };
        let out = collect
            .call(&json!({"job_id": id}), &AtomicBool::new(false))
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["status"], "completed");
        assert!(v["stdout"].as_str().unwrap().contains("job-ok"));
        assert_eq!(v["available"], true);
    }

    #[tokio::test]
    async fn failed_job_reports_exit_code() {
        let dir = tmp();
        let store = Arc::new(JobStore::new(MAX_JOBS));
        let tool = JobStartTool {
            backend: Arc::new(NativeShell::new()),
            project_path: dir,
            timeout_secs: 30,
            idle_timeout_secs: 30,
            store: store.clone(),
        };
        let out = tool
            .call(&json!({"command": "exit 7"}), &AtomicBool::new(false))
            .await
            .unwrap();
        let id: Value = serde_json::from_str::<Value>(&out).unwrap()["job_id"].clone();
        let id = id.as_str().unwrap().to_string();
        let done = wait_done(&store, &id).await;
        assert_eq!(done.status, "failed");
        assert_eq!(done.exit_code, 7);
    }

    #[tokio::test]
    async fn cancel_records_intent() {
        let dir = tmp();
        let store = Arc::new(JobStore::new(MAX_JOBS));
        let tool = JobStartTool {
            backend: Arc::new(NativeShell::new()),
            project_path: dir,
            timeout_secs: 30,
            idle_timeout_secs: 30,
            store: store.clone(),
        };
        let out = tool
            .call(&json!({"command": "sleep 0.2"}), &AtomicBool::new(false))
            .await
            .unwrap();
        let id: Value = serde_json::from_str::<Value>(&out).unwrap()["job_id"].clone();
        let id = id.as_str().unwrap().to_string();
        let cancel = JobCancelTool {
            store: store.clone(),
        };
        let out = cancel
            .call(&json!({"job_id": id}), &AtomicBool::new(false))
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["cancel_requested"], true);
        let done = wait_done(&store, &id).await;
        assert!(done.cancel_requested);
    }

    #[tokio::test]
    async fn max_jobs_is_enforced() {
        let dir = tmp();
        let store = Arc::new(JobStore::new(1));
        let tool = JobStartTool {
            backend: Arc::new(NativeShell::new()),
            project_path: dir,
            timeout_secs: 30,
            idle_timeout_secs: 30,
            store: store.clone(),
        };
        let _ = tool
            .call(&json!({"command": "sleep 0.3"}), &AtomicBool::new(false))
            .await
            .unwrap();
        let second = tool
            .call(&json!({"command": "printf x"}), &AtomicBool::new(false))
            .await;
        assert!(second.is_err(), "second job should be rejected");
    }
}
