// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Agent core: ReAct loop, main agent, prompts, sub-agents, compaction.

pub mod compact;
pub mod main_agent;
pub mod prompts;
pub mod react_loop;
pub mod sanitize;
pub mod sub_agent;

pub use main_agent::MainAgent;
pub use react_loop::{ReactLoop, RunResult, RunStatus};
