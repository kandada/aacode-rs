// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Read-only Mobile Use tools (`ui_*`).
//!
//! These are the observation half of Mobile Use: the model can inspect the
//! device UI but **cannot** click/type/gesture. Mutating `ui_act` /
//! `ui_gesture` / `ui_set_progress` stay host-only until a per-call approval
//! flow exists.
//!
//! The host (Android/iOS) implements `DevicePlugin::ui_*`; fastshell forwards
//! through `Fastshell::ui_call`, and `FastshellBackend` exposes it here.

use super::backend::ShellBackend;
use super::registry::Tool;
use super::schema::{ParamType, ToolParameter, ToolSchema};
use crate::error::Result;
use serde_json::{json, Value};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

fn call_ui(backend: &Arc<dyn ShellBackend>, method: &str, args: Value) -> Result<String> {
    match backend.ui_call(method, &args) {
        None => Ok(json!({
            "success": false,
            "error": "mobile use unavailable: no host backend registered"
        })
        .to_string()),
        Some(Ok(v)) => Ok(v.to_string()),
        Some(Err(e)) => Ok(json!({"success": false, "error": e}).to_string()),
    }
}

fn target_arg(args: &Value) -> Value {
    args.get("target").cloned().unwrap_or_else(|| json!({}))
}

pub struct UiBackendsTool {
    pub backend: Arc<dyn ShellBackend>,
}

#[async_trait::async_trait]
impl Tool for UiBackendsTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "ui_backends",
            "List the ready Mobile Use backends (e.g. accessibility / shizuku / root). \
             Read-only. The actual backend is chosen by the host's user settings.",
            vec![],
        )
    }
    async fn call(&self, _args: &Value, _c: &AtomicBool) -> Result<String> {
        call_ui(&self.backend, "ui.backends", json!({}))
    }
}

pub struct UiListWindowsTool {
    pub backend: Arc<dyn ShellBackend>,
}

#[async_trait::async_trait]
impl Tool for UiListWindowsTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "ui_list_windows",
            "List visible device windows with package/window id/bounds/focus. Read-only.",
            vec![],
        )
    }
    async fn call(&self, _args: &Value, _c: &AtomicBool) -> Result<String> {
        call_ui(&self.backend, "ui.list_windows", json!({}))
    }
}

pub struct UiSnapshotTool {
    pub backend: Arc<dyn ShellBackend>,
}

#[async_trait::async_trait]
impl Tool for UiSnapshotTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "ui_snapshot",
            "Read the device UI tree (role/text/view id/bounds/states) with a stable \
             node_ref and fingerprint per node. Read-only. Password fields are redacted. \
             Bounded by max_depth/max_nodes/max_text_len; `truncated` is explicit.",
            vec![
                ToolParameter::new(
                    "target",
                    ParamType::Object,
                    false,
                    "Optional {package, window_id}.",
                    &[],
                ),
                ToolParameter::new(
                    "max_depth",
                    ParamType::Integer,
                    false,
                    "Max depth (default 64).",
                    &[],
                ),
                ToolParameter::new(
                    "max_nodes",
                    ParamType::Integer,
                    false,
                    "Max nodes (default 300).",
                    &[],
                ),
                ToolParameter::new(
                    "max_text_len",
                    ParamType::Integer,
                    false,
                    "Max chars per text/description (0 = unlimited).",
                    &[],
                ),
            ],
        )
    }
    async fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let limits = json!({
            "max_depth": args.get("max_depth").and_then(|v| v.as_u64()).unwrap_or(64),
            "max_nodes": args.get("max_nodes").and_then(|v| v.as_u64()).unwrap_or(300),
            "max_text_len": args.get("max_text_len").and_then(|v| v.as_u64()).unwrap_or(0),
        });
        call_ui(
            &self.backend,
            "ui.snapshot",
            json!({"target": target_arg(args), "limits": limits}),
        )
    }
}

pub struct UiFindTool {
    pub backend: Arc<dyn ShellBackend>,
}

#[async_trait::async_trait]
impl Tool for UiFindTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "ui_find",
            "Find UI nodes by selector: text= / role= / role=button[name=\"X\"] / \
             id= (view id). Returns bounded matches with node_ref + fingerprint. Read-only.",
            vec![
                ToolParameter::new(
                    "selector",
                    ParamType::String,
                    true,
                    "text= / role= / id= (view id) selector.",
                    &["query"],
                ),
                ToolParameter::new(
                    "limit",
                    ParamType::Integer,
                    false,
                    "Max matches (default 20).",
                    &[],
                ),
                ToolParameter::new(
                    "target",
                    ParamType::Object,
                    false,
                    "Optional {package, window_id}.",
                    &[],
                ),
            ],
        )
    }
    async fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let selector = args.get("selector").and_then(|v| v.as_str()).unwrap_or("");
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(20);
        call_ui(
            &self.backend,
            "ui.find",
            json!({"target": target_arg(args), "selector": selector, "limit": limit}),
        )
    }
}

pub struct UiWaitTool {
    pub backend: Arc<dyn ShellBackend>,
}

#[async_trait::async_trait]
impl Tool for UiWaitTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "ui_wait",
            "Wait (bounded, ≤10s) until a selector is present or gone. Read-only: it \
             only observes, never acts.",
            vec![
                ToolParameter::new(
                    "selector",
                    ParamType::String,
                    true,
                    "text= / role= / id= selector.",
                    &["query"],
                ),
                ToolParameter::new(
                    "present",
                    ParamType::Boolean,
                    false,
                    "true = wait until present (default), false = until gone.",
                    &[],
                ),
                ToolParameter::new(
                    "timeout_ms",
                    ParamType::Integer,
                    false,
                    "Timeout (default 5000, max 10000).",
                    &[],
                ),
                ToolParameter::new(
                    "target",
                    ParamType::Object,
                    false,
                    "Optional {package, window_id}.",
                    &[],
                ),
            ],
        )
    }
    async fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let selector = args.get("selector").and_then(|v| v.as_str()).unwrap_or("");
        let present = args
            .get("present")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let timeout = args
            .get("timeout_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(5000)
            .min(10_000);
        call_ui(
            &self.backend,
            "ui.wait",
            json!({
                "target": target_arg(args),
                "selector": selector,
                "present": present,
                "timeout_ms": timeout,
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::backend::CmdOutput;

    struct MockUiBackend;
    impl ShellBackend for MockUiBackend {
        fn run(
            &self,
            _command: &str,
            _stdin: Option<&str>,
            _timeout: u64,
            _idle: u64,
            _cwd: &std::path::Path,
        ) -> CmdOutput {
            CmdOutput {
                stdout: String::new(),
                stderr: String::new(),
                exit_code: 0,
            }
        }
        fn kind(&self) -> &'static str {
            "mock"
        }
        fn ui_call(
            &self,
            method: &str,
            _args: &Value,
        ) -> Option<std::result::Result<Value, String>> {
            Some(Ok(match method {
                "ui.backends" => json!(["accessibility"]),
                "ui.list_windows" => json!({"windows": [{"package": "com.x", "window_id": 1}]}),
                "ui.snapshot" => {
                    json!({"backend": "accessibility", "nodes": [{"node_ref": "n1", "text": "Go"}]})
                }
                "ui.find" => json!({"nodes": [{"node_ref": "n1", "text": "Go"}]}),
                "ui.wait" => json!({"present": true}),
                _ => json!({"ok": false}),
            }))
        }
    }

    fn backend() -> Arc<dyn ShellBackend> {
        Arc::new(MockUiBackend)
    }

    fn parse(s: String) -> Value {
        serde_json::from_str(&s).unwrap()
    }

    #[tokio::test]
    async fn read_only_tools_roundtrip() {
        let b = backend();
        let cancel = AtomicBool::new(false);

        let v = parse(
            UiBackendsTool { backend: b.clone() }
                .call(&json!({}), &cancel)
                .await
                .unwrap(),
        );
        assert_eq!(v[0], "accessibility");

        let v = parse(
            UiListWindowsTool { backend: b.clone() }
                .call(&json!({}), &cancel)
                .await
                .unwrap(),
        );
        assert_eq!(v["windows"][0]["package"], "com.x");

        let v = parse(
            UiSnapshotTool { backend: b.clone() }
                .call(&json!({"max_nodes": 10}), &cancel)
                .await
                .unwrap(),
        );
        assert_eq!(v["nodes"][0]["text"], "Go");

        let v = parse(
            UiFindTool { backend: b.clone() }
                .call(&json!({"selector": "text=Go"}), &cancel)
                .await
                .unwrap(),
        );
        assert_eq!(v["nodes"][0]["node_ref"], "n1");

        let v = parse(
            UiWaitTool { backend: b }
                .call(&json!({"selector": "text=Go", "present": true}), &cancel)
                .await
                .unwrap(),
        );
        assert_eq!(v["present"], true);
    }

    struct CapturingBackend {
        seen: std::sync::Mutex<Vec<(String, Value)>>,
    }

    impl ShellBackend for CapturingBackend {
        fn run(
            &self,
            _command: &str,
            _stdin: Option<&str>,
            _timeout: u64,
            _idle: u64,
            _cwd: &std::path::Path,
        ) -> CmdOutput {
            CmdOutput {
                stdout: String::new(),
                stderr: String::new(),
                exit_code: 0,
            }
        }
        fn kind(&self) -> &'static str {
            "capture"
        }
        fn ui_call(
            &self,
            method: &str,
            args: &Value,
        ) -> Option<std::result::Result<Value, String>> {
            self.seen
                .lock()
                .unwrap()
                .push((method.to_string(), args.clone()));
            Some(Ok(json!({"ok": true})))
        }
    }

    #[tokio::test]
    async fn read_only_tools_forward_bounded_args() {
        let b = Arc::new(CapturingBackend {
            seen: std::sync::Mutex::new(Vec::new()),
        });
        let cancel = AtomicBool::new(false);

        UiSnapshotTool { backend: b.clone() }
            .call(&json!({"max_nodes": 7, "max_depth": 3}), &cancel)
            .await
            .unwrap();
        UiWaitTool { backend: b.clone() }
            .call(
                &json!({"selector": "text=Go", "timeout_ms": 999_999}),
                &cancel,
            )
            .await
            .unwrap();
        UiFindTool { backend: b.clone() }
            .call(&json!({"selector": "id=x", "limit": 500}), &cancel)
            .await
            .unwrap();

        let seen = b.seen.lock().unwrap();
        assert_eq!(seen[0].0, "ui.snapshot");
        assert_eq!(seen[0].1["limits"]["max_nodes"], 7);
        assert_eq!(seen[0].1["limits"]["max_depth"], 3);
        assert_eq!(seen[1].0, "ui.wait");
        assert_eq!(seen[1].1["timeout_ms"], 10_000);
        assert_eq!(seen[2].0, "ui.find");
        assert_eq!(seen[2].1["selector"], "id=x");
    }

    #[test]
    fn default_registry_exposes_read_only_ui_tools() {
        use crate::config::AgentConfig;
        use crate::tools::backend::NativeShell;
        use crate::tools::registry::is_read_only;

        let cfg = AgentConfig::default();
        let reg = crate::tools::build_default_registry(
            Arc::new(NativeShell::new()),
            std::env::temp_dir(),
            &cfg,
        );
        for name in [
            "ui_backends",
            "ui_list_windows",
            "ui_snapshot",
            "ui_find",
            "ui_wait",
        ] {
            assert!(reg.contains(name), "{name} missing from registry");
            assert!(is_read_only(name), "{name} must be read-only");
        }
        for name in ["ui_act", "ui_gesture", "ui_set_progress"] {
            assert!(
                !reg.contains(name),
                "{name} must not be exposed to the model"
            );
        }
    }

    #[tokio::test]
    async fn no_host_backend_is_reported() {
        // NativeShell has no plugin → ui_call returns None.
        let b: Arc<dyn ShellBackend> = Arc::new(crate::tools::backend::NativeShell::new());
        let cancel = AtomicBool::new(false);
        let v = parse(
            UiSnapshotTool { backend: b }
                .call(&json!({}), &cancel)
                .await
                .unwrap(),
        );
        assert_eq!(v["success"], false);
        assert!(v["error"].as_str().unwrap().contains("unavailable"));
    }
}
