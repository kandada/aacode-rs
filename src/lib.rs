// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! # aacode-rs
//!
//! A Rust port of the AACode ReAct coding agent, built on top of the
//! `fastshell` sandbox runtime for mobile (Android/iOS) AI agents.
//!
//! Module layout mirrors `design.md` §4.

pub mod config;
pub mod error;
pub mod stream;

pub mod llm;
pub mod tools;
pub mod session;
pub mod context;
pub mod agent;
pub mod mcp;

pub mod runtime;
pub mod ffi;

pub use config::AgentConfig;
pub use error::{AacodeError, Result};
pub use runtime::AgentRuntime;
