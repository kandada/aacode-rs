// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Tool schema definitions + conversion to OpenAI/Anthropic native tools.
//!
//! Ported from Python `utils/tool_schemas.py` + `utils/tool_adapter.py`.

use serde_json::{json, Value};

/// JSON-schema primitive type for a parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamType {
    String,
    Integer,
    Number,
    Boolean,
    Array,
    Object,
}

impl ParamType {
    pub fn json_name(&self) -> &'static str {
        match self {
            ParamType::String => "string",
            ParamType::Integer => "integer",
            ParamType::Number => "number",
            ParamType::Boolean => "boolean",
            ParamType::Array => "array",
            ParamType::Object => "object",
        }
    }

    /// Whether a serde_json Value matches this declared type (used for validation).
    pub fn matches(&self, v: &Value) -> bool {
        match self {
            ParamType::String => v.is_string(),
            ParamType::Integer => v.is_i64() || v.is_u64(),
            ParamType::Number => v.is_number(),
            ParamType::Boolean => v.is_boolean(),
            ParamType::Array => v.is_array(),
            ParamType::Object => v.is_object(),
        }
    }
}

/// A single tool parameter definition.
#[derive(Debug, Clone)]
pub struct ToolParameter {
    pub name: &'static str,
    pub ty: ParamType,
    pub required: bool,
    pub description: &'static str,
    /// Alternative names that get normalized to `name`.
    pub aliases: &'static [&'static str],
}

impl ToolParameter {
    pub const fn new(
        name: &'static str,
        ty: ParamType,
        required: bool,
        description: &'static str,
        aliases: &'static [&'static str],
    ) -> Self {
        ToolParameter {
            name,
            ty,
            required,
            description,
            aliases,
        }
    }
}

/// A tool's schema: name, description, parameters.
#[derive(Debug, Clone)]
pub struct ToolSchema {
    pub name: &'static str,
    pub description: &'static str,
    pub parameters: Vec<ToolParameter>,
}

impl ToolSchema {
    pub fn new(
        name: &'static str,
        description: &'static str,
        parameters: Vec<ToolParameter>,
    ) -> Self {
        ToolSchema {
            name,
            description,
            parameters,
        }
    }

    /// Build the JSON-schema `properties` object and `required` array.
    fn json_schema(&self) -> (Value, Vec<Value>) {
        let mut props = serde_json::Map::new();
        let mut required = Vec::new();
        for p in &self.parameters {
            props.insert(
                p.name.to_string(),
                json!({"type": p.ty.json_name(), "description": p.description}),
            );
            if p.required {
                required.push(Value::String(p.name.to_string()));
            }
        }
        (Value::Object(props), required)
    }

    /// OpenAI native tool object.
    pub fn to_openai_tool(&self) -> Value {
        let (props, required) = self.json_schema();
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                }
            }
        })
    }

    /// Anthropic native tool object.
    pub fn to_anthropic_tool(&self) -> Value {
        let (props, required) = self.json_schema();
        json!({
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": props,
                "required": required,
            }
        })
    }

    /// Normalize input params: rewrite known aliases to canonical names.
    pub fn normalize_params(&self, input: &Value) -> Value {
        let obj = match input.as_object() {
            Some(o) => o,
            None => return input.clone(),
        };
        // alias -> canonical
        let mut alias_map: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
        for p in &self.parameters {
            alias_map.insert(p.name, p.name);
            for a in p.aliases {
                alias_map.insert(a, p.name);
            }
        }
        let mut out = serde_json::Map::new();
        for (k, v) in obj {
            let canonical = alias_map.get(k.as_str()).copied().unwrap_or(k.as_str());
            out.insert(canonical.to_string(), v.clone());
        }
        Value::Object(out)
    }

    /// Validate (after normalization): required present + type match.
    /// Returns Ok(()) or Err(human message).
    pub fn validate(&self, normalized: &Value) -> std::result::Result<(), String> {
        let obj = normalized.as_object();
        // missing required
        let mut missing = Vec::new();
        for p in &self.parameters {
            if p.required {
                let present = obj.map(|o| o.contains_key(p.name)).unwrap_or(false);
                if !present {
                    missing.push(p.name);
                }
            }
        }
        if !missing.is_empty() {
            let details: Vec<String> = missing
                .iter()
                .map(|m| {
                    let p = self.parameters.iter().find(|p| &p.name == m).unwrap();
                    format!("  • {} ({}): {}", p.name, p.ty.json_name(), p.description)
                })
                .collect();
            return Err(format!(
                "❌ Missing required parameters: {}\n\n📋 Parameter details:\n{}",
                missing.join(", "),
                details.join("\n")
            ));
        }
        // type check present params
        if let Some(o) = obj {
            for p in &self.parameters {
                if let Some(v) = o.get(p.name) {
                    if !v.is_null() && !p.ty.matches(v) {
                        return Err(format!(
                            "Parameter '{}' expected type {}, got a different type",
                            p.name,
                            p.ty.json_name()
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Human-readable documentation for the tool (used in not-found/error hints).
    pub fn documentation(&self) -> String {
        let mut doc = format!("## {}\n\n{}\n\n### Parameters\n", self.name, self.description);
        for p in &self.parameters {
            let req = if p.required { "required" } else { "optional" };
            let aliases = if p.aliases.is_empty() {
                String::new()
            } else {
                format!(" (aliases: {})", p.aliases.join(", "))
            };
            doc.push_str(&format!(
                "- {}{} ({}, {}): {}\n",
                p.name,
                aliases,
                p.ty.json_name(),
                req,
                p.description
            ));
        }
        doc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ToolSchema {
        ToolSchema::new(
            "run_shell",
            "run a shell command",
            vec![
                ToolParameter::new(
                    "command",
                    ParamType::String,
                    true,
                    "the command",
                    &["cmd", "shell", "script", "exec"],
                ),
                ToolParameter::new("timeout", ParamType::Integer, false, "seconds", &["time_limit"]),
            ],
        )
    }

    #[test]
    fn openai_tool_shape() {
        let t = sample().to_openai_tool();
        assert_eq!(t["type"], "function");
        assert_eq!(t["function"]["name"], "run_shell");
        assert_eq!(t["function"]["parameters"]["properties"]["command"]["type"], "string");
        assert_eq!(t["function"]["parameters"]["required"][0], "command");
    }

    #[test]
    fn anthropic_tool_shape() {
        let t = sample().to_anthropic_tool();
        assert_eq!(t["name"], "run_shell");
        assert_eq!(t["input_schema"]["properties"]["timeout"]["type"], "integer");
    }

    #[test]
    fn alias_normalization() {
        let s = sample();
        let n = s.normalize_params(&json!({"cmd": "ls", "time_limit": 5}));
        assert_eq!(n["command"], "ls");
        assert_eq!(n["timeout"], 5);
    }

    #[test]
    fn validate_missing_required() {
        let s = sample();
        let n = s.normalize_params(&json!({"timeout": 5}));
        let err = s.validate(&n).unwrap_err();
        assert!(err.contains("command"));
    }

    #[test]
    fn validate_type_error() {
        let s = sample();
        let n = s.normalize_params(&json!({"command": 123}));
        assert!(s.validate(&n).is_err());
    }

    #[test]
    fn validate_ok() {
        let s = sample();
        let n = s.normalize_params(&json!({"command": "ls", "timeout": 5}));
        assert!(s.validate(&n).is_ok());
    }
}
