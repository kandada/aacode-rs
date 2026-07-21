// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Top-level orchestration: choose a shell backend, build the agent, run a task.
//! Shared by the CLI (`bin/aacode`) and the FFI layer (`ffi.rs`).
//!
//! Backend selection (`config.shell_backend`):
//!   * `Auto` (default) — native OS shell on desktop, fastshell sandbox on mobile.
//!   * `Native` — force the real OS shell (`sh -c` / `cmd /C`).
//!   * `Fastshell` — force the sandbox engine (VFS jail + embedded CPython).

use crate::agent::{MainAgent, RunResult, RunStatus};
use crate::config::{AgentConfig, ShellBackendChoice};
use crate::error::{AacodeError, Result};
use crate::llm::build_client;
use crate::session::SessionManager;
use crate::stream::EventSink;
use crate::tools::backend::{BackendKind, FastshellBackend, NativeShell, ShellBackend};
use fastshell::{Config, Fastshell};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

/// A prepared agent runtime bound to a project directory + shell backend.
pub struct AgentRuntime {
    pub config: AgentConfig,
    pub project_path: PathBuf,
    pub backend: Arc<dyn ShellBackend>,
}

/// Resolve the effective backend kind from config + platform.
fn resolve_backend_kind(choice: ShellBackendChoice) -> BackendKind {
    match choice {
        ShellBackendChoice::Native => BackendKind::Native,
        ShellBackendChoice::Fastshell => BackendKind::Fastshell,
        ShellBackendChoice::Auto => BackendKind::platform_default(),
    }
}

impl AgentRuntime {
    /// Initialize the runtime for `project_path`, building the appropriate
    /// shell backend. The native backend needs no setup; the fastshell backend
    /// initializes a VFS sandbox + embedded CPython.
    pub fn init(config: AgentConfig, project_path: PathBuf) -> Result<Self> {
    // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0
        // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0
        std::fs::create_dir_all(&project_path)
            .map_err(|e| AacodeError::Io(format!("create project dir: {e}")))?;

        let kind = resolve_backend_kind(config.shell_backend);
        let backend: Arc<dyn ShellBackend> = match kind {
            BackendKind::Native => Arc::new(NativeShell::new()),
            BackendKind::Fastshell => {
                let mut fs = Fastshell::new();
                let mut fscfg = Config::default();
                fscfg.sandbox_path = project_path.to_string_lossy().to_string();
                // Python enabled so run_shell can execute `python3 ...` for testing.
                fscfg.python_enabled = true;
                fscfg.command_timeout_ms = config.timeouts.shell_command * 1000;
                fs.init(fscfg).map_err(AacodeError::Other)?;
                Arc::new(FastshellBackend::new(Arc::new(Mutex::new(fs))))
            }
        };

        Ok(AgentRuntime {
            config,
            project_path,
            backend,
        })
    }

    /// Initialize reusing an already-constructed fastshell handle (FFI path
    /// where the host already called fastshell_init on the same sandbox).
    /// Forces the fastshell backend regardless of config.
    pub fn with_fastshell(
        config: AgentConfig,
        project_path: PathBuf,
        fs: Arc<Mutex<Fastshell>>,
    ) -> Self {
        AgentRuntime {
            config,
            project_path,
            backend: Arc::new(FastshellBackend::new(fs)),
        }
    }

    /// Which backend is active (for diagnostics).
    pub fn backend_kind(&self) -> &'static str {
        self.backend.kind()
    }

    /// Validate the configured API key with a lightweight probe.
    pub fn validate_api_key(&self) -> Result<()> {
        let client = build_client(&self.config.model);
        client.validate()
    }

    /// Run a task to completion, streaming events to `emitter`.
    /// `session_id` optionally continues an existing session.
    pub fn run_task(
        &self,
        task: &str,
        session_id: Option<&str>,
        emitter: &dyn EventSink,
        cancel: &AtomicBool,
    ) -> Result<RunResult> {
        // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0
        // Validate config early for a clear error.
        let errs = self.config.validate();
        if !errs.is_empty() {
            emitter.error(&errs.join("; "));
            return Ok(RunResult {
                status: RunStatus::Error(errs.join("; ")),
                iterations: 0,
                final_text: String::new(),
            });
        }

        let mut session = SessionManager::new(&self.project_path);
        if let Some(id) = session_id {
            let _ = session.switch_session(id);
        }

        let agent = MainAgent::new(
            self.config.clone(),
            self.project_path.clone(),
            self.backend.clone(),
        );
        agent.execute(task, &mut session, emitter, cancel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::CollectingSink;

    fn tmp() -> PathBuf {
        std::env::temp_dir().join(format!(
            "aacode_rt_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ))
    }

    #[test]
    fn desktop_defaults_to_native_backend() {
        let d = tmp();
        let mut cfg = AgentConfig::default();
        cfg.model.api_key = Some("x".into());
        let rt = AgentRuntime::init(cfg, d.clone()).unwrap();
        // On desktop, Auto → native.
        #[cfg(not(any(target_os = "android", target_os = "ios")))]
        assert_eq!(rt.backend_kind(), "native");
    }

    #[test]
    fn native_backend_uses_real_cwd() {
        let d = tmp();
        let mut cfg = AgentConfig::default();
        cfg.model.api_key = Some("x".into());
        cfg.shell_backend = ShellBackendChoice::Native;
        let rt = AgentRuntime::init(cfg, d.clone()).unwrap();
        let out = rt.backend.run("echo hi > f.txt && cat f.txt", None, 10, &d);
        assert!(out.stdout.contains("hi"));
        // File created at the real project dir (no VFS jail nesting).
        assert!(d.join("f.txt").exists());
    }

    #[test]
    fn force_fastshell_backend() {
        let d = tmp();
        let mut cfg = AgentConfig::default();
        cfg.model.api_key = Some("x".into());
        cfg.shell_backend = ShellBackendChoice::Fastshell;
        let rt = AgentRuntime::init(cfg, d).unwrap();
        assert_eq!(rt.backend_kind(), "fastshell");
        let out = rt.backend.run("echo hi", None, 10, std::path::Path::new("."));
        assert!(out.stdout.contains("hi"));
    }

    #[test]
    fn run_task_missing_key_errors() {
        let d = tmp();
        let cfg = AgentConfig::default(); // no api key
        let rt = AgentRuntime::init(cfg, d).unwrap();
        let sink = CollectingSink::new(false);
        let cancel = AtomicBool::new(false);
        let res = rt.run_task("do x", None, &sink, &cancel).unwrap();
        assert!(matches!(res.status, RunStatus::Error(_)));
        assert!(sink.lines().iter().any(|l| l.contains("error")));
    }
}
