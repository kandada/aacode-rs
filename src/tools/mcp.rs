// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! MCP tools — list_mcp_tools / call_mcp_tool / get_mcp_status.

use super::registry::Tool;
use super::schema::{ParamType, ToolParameter, ToolSchema};
use crate::error::Result;
use crate::mcp::McpManager;
use serde_json::Value;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

pub struct ListMcpToolsTool {
    pub mgr: Arc<McpManager>,
}
impl Tool for ListMcpToolsTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "list_mcp_tools",
            "List all available MCP tools from configured servers.",
            vec![],
        )
    }
    fn call(&self, _args: &Value, _c: &AtomicBool) -> Result<String> {
        Ok(self.mgr.list_tools().to_string())
    }
}

pub struct CallMcpToolTool {
    pub mgr: Arc<McpManager>,
}
impl Tool for CallMcpToolTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "call_mcp_tool",
            "Call an MCP tool. tool_name is 'server.tool' or just 'tool'.",
            vec![
                ToolParameter::new("tool_name", ParamType::String, true, "server.tool or tool", &["tool", "name", "function"]),
                ToolParameter::new("arguments", ParamType::Object, false, "Tool arguments", &["args", "params", "input"]),
            ],
        )
    }
    fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let name = args.get("tool_name").and_then(|v| v.as_str()).unwrap_or("");
        let arguments = args.get("arguments").cloned().unwrap_or(serde_json::json!({}));
        Ok(self.mgr.call_tool(name, arguments).to_string())
    }
}

pub struct McpStatusTool {
    pub mgr: Arc<McpManager>,
}
impl Tool for McpStatusTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("get_mcp_status", "Get MCP server status.", vec![])
    }
    fn call(&self, _args: &Value, _c: &AtomicBool) -> Result<String> {
        Ok(self.mgr.status().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn status_tool_reports_empty() {
        let mgr = Arc::new(McpManager::new(vec![], 5));
        let t = McpStatusTool { mgr };
        let cancel = AtomicBool::new(false);
        let out = t.call(&json!({}), &cancel).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["total_count"], 0);
    }

    #[test]
    fn list_tool_empty() {
        let mgr = Arc::new(McpManager::new(vec![], 5));
        let t = ListMcpToolsTool { mgr };
        let cancel = AtomicBool::new(false);
        let out = t.call(&json!({}), &cancel).unwrap();
        assert!(out.contains("\"count\":0"));
    }
}
