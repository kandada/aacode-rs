// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! MCP (Model Context Protocol) client manager.
//!
//! Supports stdio transport (spawn a server process and speak JSON-RPC 2.0
//! over stdin/stdout, framed with Content-Length headers) and a minimal SSE/HTTP
//! transport. Ported from Python `utils/mcp_manager.py` + `sandbox/mcp_client.py`.
//!
//! Configuration lives in the agent config's environment or a JSON file; here
//! we accept explicit server specs. On mobile, stdio servers require the
//! ability to spawn processes (fastshell subprocess policy) — otherwise the
//! manager simply reports zero servers.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

/// Transport type for an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum McpTransport {
    Stdio,
    Sse,
}

/// One configured MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerSpec {
    pub name: String,
    pub transport: McpTransport,
    /// For stdio: the command to run. For sse: unused.
    #[serde(default)]
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    /// For sse: base URL.
    #[serde(default)]
    pub url: String,
}

/// A running stdio server connection.
struct StdioConn {
    child: Child,
    next_id: i64,
}

impl StdioConn {
    fn spawn(spec: &McpServerSpec) -> crate::error::Result<Self> {
        let mut cmd = Command::new(&spec.command);
        cmd.args(&spec.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        let child = cmd
            .spawn()
            .map_err(|e| crate::error::AacodeError::Io(format!("spawn mcp server: {e}")))?;
        let mut conn = StdioConn { child, next_id: 1 };
        // Initialize handshake.
        let _ = conn.request(
            "initialize",
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "aacode-rs", "version": "0.1.0"}
            }),
        );
        // notifications/initialized (no response expected)
        let _ = conn.notify("notifications/initialized", json!({}));
        Ok(conn)
    }

    fn write_message(&mut self, msg: &Value) -> crate::error::Result<()> {
        let body = serde_json::to_string(msg)?;
        let stdin = self
            .child
            .stdin
            .as_mut()
            .ok_or_else(|| crate::error::AacodeError::Io("no stdin".into()))?;
        write!(stdin, "Content-Length: {}\r\n\r\n{}", body.len(), body)?;
        stdin.flush()?;
        Ok(())
    }

    fn read_message(&mut self) -> crate::error::Result<Value> {
        let stdout = self
            .child
            .stdout
            .as_mut()
            .ok_or_else(|| crate::error::AacodeError::Io("no stdout".into()))?;
        let mut reader = BufReader::new(stdout);
        // Read headers.
        let mut content_length = 0usize;
        loop {
            let mut line = String::new();
            let n = reader.read_line(&mut line)?;
            if n == 0 {
                return Err(crate::error::AacodeError::Io("eof".into()));
            }
            let t = line.trim_end();
            if t.is_empty() {
                break;
            }
            if let Some(rest) = t.to_lowercase().strip_prefix("content-length:") {
                content_length = rest.trim().parse().unwrap_or(0);
            }
        }
        let mut buf = vec![0u8; content_length];
        reader.read_exact(&mut buf)?;
        let v: Value = serde_json::from_slice(&buf)?;
        Ok(v)
    }

    fn request(&mut self, method: &str, params: Value) -> crate::error::Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        let msg = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        self.write_message(&msg)?;
        // Read until we get the matching id (skip notifications).
        for _ in 0..50 {
            let resp = self.read_message()?;
            if resp.get("id").and_then(|v| v.as_i64()) == Some(id) {
                if let Some(err) = resp.get("error") {
                    return Err(crate::error::AacodeError::Api(format!("mcp error: {err}")));
                }
                return Ok(resp.get("result").cloned().unwrap_or(json!({})));
            }
        }
        Err(crate::error::AacodeError::Api("no matching mcp response".into()))
    }

    fn notify(&mut self, method: &str, params: Value) -> crate::error::Result<()> {
        let msg = json!({"jsonrpc": "2.0", "method": method, "params": params});
        self.write_message(&msg)
    }
}

impl Drop for StdioConn {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The MCP manager holding server specs and lazily-opened connections.
pub struct McpManager {
    specs: Vec<McpServerSpec>,
    conns: Mutex<std::collections::HashMap<String, StdioConn>>,
    timeout_secs: u64,
}

impl McpManager {
    pub fn new(specs: Vec<McpServerSpec>, timeout_secs: u64) -> Self {
        McpManager {
            specs,
            conns: Mutex::new(std::collections::HashMap::new()),
            timeout_secs,
        }
    }

    pub fn server_names(&self) -> Vec<String> {
        self.specs.iter().map(|s| s.name.clone()).collect()
    }

    fn spec(&self, name: &str) -> Option<&McpServerSpec> {
        self.specs.iter().find(|s| s.name == name)
    }

    /// List tools from all servers. Returns a map name -> Vec<tool json>.
    pub fn list_tools(&self) -> Value {
        let mut all = serde_json::Map::new();
        let mut count = 0usize;
        let mut connected = Vec::new();
        for spec in &self.specs {
            match self.list_server_tools(spec) {
                Ok(tools) => {
                    count += tools.as_array().map(|a| a.len()).unwrap_or(0);
                    connected.push(spec.name.clone());
                    all.insert(spec.name.clone(), tools);
                }
                Err(_) => {
                    all.insert(spec.name.clone(), json!([]));
                }
            }
        }
        json!({
            "success": true,
            "tools": all,
            "count": count,
            "connected_servers": connected,
        })
    }

    fn list_server_tools(&self, spec: &McpServerSpec) -> crate::error::Result<Value> {
        match spec.transport {
            McpTransport::Stdio => {
                let mut conns = self.conns.lock().unwrap();
                if !conns.contains_key(&spec.name) {
                    let conn = StdioConn::spawn(spec)?;
                    conns.insert(spec.name.clone(), conn);
                }
                let conn = conns.get_mut(&spec.name).unwrap();
                let res = conn.request("tools/list", json!({}))?;
                Ok(res.get("tools").cloned().unwrap_or(json!([])))
            }
            McpTransport::Sse => self.sse_request(spec, "tools/list", json!({}))
                .map(|r| r.get("tools").cloned().unwrap_or(json!([]))),
        }
    }

    /// Call a tool: `tool_name` may be "server.tool" or just "tool".
    pub fn call_tool(&self, tool_name: &str, arguments: Value) -> Value {
        let (server, tool) = match tool_name.split_once('.') {
            Some((s, t)) => (Some(s.to_string()), t.to_string()),
            None => (None, tool_name.to_string()),
        };
        let target_spec = match &server {
            Some(s) => self.spec(s).cloned(),
            None => self.specs.first().cloned(),
        };
        let spec = match target_spec {
            Some(s) => s,
            None => return json!({"success": false, "error": "no mcp server configured"}),
        };
        let params = json!({"name": tool, "arguments": arguments});
        let res = match spec.transport {
            McpTransport::Stdio => {
                let mut conns = self.conns.lock().unwrap();
                if !conns.contains_key(&spec.name) {
                    match StdioConn::spawn(&spec) {
                        Ok(c) => {
                            conns.insert(spec.name.clone(), c);
                        }
                        Err(e) => return json!({"success": false, "error": e.to_string()}),
                    }
                }
                let conn = conns.get_mut(&spec.name).unwrap();
                conn.request("tools/call", params)
            }
            McpTransport::Sse => self.sse_request(&spec, "tools/call", params),
        };
        match res {
            Ok(v) => json!({"success": true, "result": v}),
            Err(e) => json!({"success": false, "error": e.to_string()}),
        }
    }

    /// HTTP JSON-RPC for SSE-style servers (single POST, JSON response).
    fn sse_request(
        &self,
        spec: &McpServerSpec,
        method: &str,
        params: Value,
    ) -> crate::error::Result<Value> {
        let body = json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params});
        let resp = ureq::post(&spec.url)
            .timeout(std::time::Duration::from_secs(self.timeout_secs))
            .set("Content-Type", "application/json")
            .send_json(body);
        match resp {
            Ok(r) => {
                let s = r.into_string().unwrap_or_default();
                let v: Value = serde_json::from_str(&s)?;
                if let Some(err) = v.get("error") {
                    return Err(crate::error::AacodeError::Api(format!("mcp error: {err}")));
                }
                Ok(v.get("result").cloned().unwrap_or(json!({})))
            }
            Err(e) => Err(crate::error::AacodeError::Network(e.to_string())),
        }
    }

    /// Status report.
    pub fn status(&self) -> Value {
        let conns = self.conns.lock().unwrap();
        let servers: Vec<Value> = self
            .specs
            .iter()
            .map(|s| {
                json!({
                    "name": s.name,
                    "transport": match s.transport { McpTransport::Stdio => "stdio", McpTransport::Sse => "sse" },
                    "connected": conns.contains_key(&s.name),
                })
            })
            .collect();
        json!({
            "success": true,
            "servers": servers,
            "total_count": self.specs.len(),
            "connected_count": conns.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_manager_status() {
        let m = McpManager::new(vec![], 10);
        let s = m.status();
        assert_eq!(s["total_count"], 0);
        assert_eq!(s["connected_count"], 0);
    }

    #[test]
    fn list_tools_no_servers() {
        let m = McpManager::new(vec![], 10);
        let v = m.list_tools();
        assert_eq!(v["count"], 0);
    }

    #[test]
    fn call_tool_no_server() {
        let m = McpManager::new(vec![], 10);
        let v = m.call_tool("x.y", json!({}));
        assert_eq!(v["success"], false);
    }

    #[test]
    fn tool_name_parsing_via_call() {
        // With a bogus stdio server, call fails gracefully (spawn error).
        let spec = McpServerSpec {
            name: "srv".into(),
            transport: McpTransport::Stdio,
            command: "/nonexistent/bin/mcp".into(),
            args: vec![],
            url: String::new(),
        };
        let m = McpManager::new(vec![spec], 5);
        let v = m.call_tool("srv.dothing", json!({}));
        assert_eq!(v["success"], false);
    }
}
