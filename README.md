[中文](README.zh.md) |

# aacode-rs — CLI Programming Agent

[![Rust](https://img.shields.io/badge/Rust-1.80+-orange.svg)](https://www.rust-lang.org/)
[![Crates.io](https://img.shields.io/crates/v/aacode-rs.svg)](https://crates.io/crates/aacode-rs)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](LICENSE)

> **AI Programming CLI Agent in pure Rust** — a lightweight ReAct architecture, 100% Rust, no Python dependency.

## Design Principles

* Shell as universal adapter — all file, code, and system operations go through `run_shell`
* File-based context — dynamic discovery, Markdown files as primary storage
* Context management — smart compaction when token budget is exhausted
* Document-based Skills — SKILL.md instruction guides, no embedded scripts
* Layered tool system — atomic tools, management tools, Skills three-layer architecture
* Safety guardrails — path confinement, dangerous command rejection, network permissions
* Cross-platform — compiles to macOS, Linux, Windows, Android, iOS from one codebase
* No LLM SDK — async HTTP streaming (tokio + reqwest) with hand-rolled SSE/JSON parsing for all LLM APIs

## Quick Start

### Operating System

This project is developed and tested on macOS and Linux. It is recommended to use macOS or Linux. Windows is also supported.

### Build

```bash
git clone https://github.com/kandada/fastshell.git
cd fastshell/aacode-rs
cp .env.example .env   # edit with your API key
cargo build --release

# The binary is at target/release/aacode
```

### Getting Started

```bash
# Run a task
cargo run --release -- -p examples/my_project "Your task description"

# Or manually
export LLM_API_KEY="your-api-key"
export LLM_API_URL="your-api-url"
export LLM_MODEL_NAME="your-model-name"
./target/release/aacode -p examples/my_project "Your task description"

# Advanced modes
## Plan-first mode
cargo run --release -- -p examples/my_project "Complex task" --plan-first

## Interactive continuous conversation
cargo run --release -- -p examples/my_project "Initial task" --interactive

## Specify session
cargo run --release -- --session session_20250128_123456_0 "Continue task"
```

### Or Install via cargo (Recommended)

After `cargo install`, you can use the `aacode` command. The default workspace is the **current directory** — no need to specify `-p` unless you want a different location.

```bash
# Install
cargo install aacode-rs

# Enter interactive session mode (no task required)
aacode

# Run a single task in current directory
aacode "your task"

# Or explicitly with aacode run
aacode run "your task"

# Specify a different project directory
aacode run -p /your/project/path "your task"
```

## Configuration

### Large Language Model

Supports DeepSeek, OpenAI, Anthropic, Kimi, MiniMax, and all OpenAI/Anthropic-compatible endpoints.

```bash
# OpenAI
export LLM_API_KEY="your-openai-key"
export LLM_API_URL="https://api.openai.com/v1"
export LLM_MODEL_NAME="gpt-4"
export LLM_GATEWAY="openai"
export LLM_MULTIMODAL="false"

# OpenAI-compatible models (DeepSeek, etc.)
export LLM_API_KEY="your-api-key"
export LLM_API_URL="https://your-api-endpoint/v1"
export LLM_MODEL_NAME="your-model-name"
export LLM_GATEWAY="openai"

# Anthropic-compatible models (Claude, Kimi, MiniMax, etc.)
export LLM_API_KEY="your-api-key"
export LLM_API_URL="https://your-api-endpoint/v1"
export LLM_MODEL_NAME="your-model-name"
export LLM_GATEWAY="anthropic"
```

### Shell Backend

```bash
# Native OS shell (default on desktop, no dependencies)
export AACODE_SHELL_BACKEND="native"

# fastshell sandbox (180+ built-in commands, VFS isolation, embedded Python)
export AACODE_SHELL_BACKEND="fastshell"
```

### Skills Directory

```bash
# Enable builtin + user-dir skills mode (optional)
export AACODE_SKILLS_DIR="/path/to/skills"
```

### Multimodal Models

Supports multimodal models (Kimi K2.5, MiniMax M2.5, etc.) for `understand_image` / `understand_ui_design` tools. Configure in aacode_config.yaml:

```yaml
multimodal:
  name: "kimi-k2.5"
  api_key: "your-kimi-key"
  api_url: "https://api.moonshot.cn/v1"
  gateway: "anthropic"
```

### Search Engine

Supports SearXNG. Users need to deploy their own and configure via environment variable `SEARCHXNG_URL`.

### MCP

Configure MCP resources (stdio and sse) in aacode_config.yaml.

### Skills

Skills are document-type: `run_skills` returns the SKILL.md instructions, which the agent follows using `run_shell` and other tools. There are two discovery modes:

| Mode | When | Source |
|---|---|---|
| **Project mode** (legacy, desktop CLI) | `AACODE_SKILLS_DIR` not set | Scans `<project>/skills/` and `<project>/.aacode/skills/` |
| **User-dir mode** (mobile hosts) | `AACODE_SKILLS_DIR` is set | Builtin skills (compiled into binary) + `<skills_dir>/*/SKILL.md` |

In project mode, **no builtin skills are injected** — you must place SKILL.md files manually in the project's `skills/` directory.

In user-dir mode, builtin skills are always available (compiled into the binary for zero file dependencies):

| Builtin Skill | Always injected | Gated by `extra_builtins` | Description |
|---|---|---|---|
| `skill_creator` | Yes | No | Meta-skill for creating and updating skills |
| `book_writer` | Yes | No | Multi-phase book writing (outline → storyline → chapters → review) |
| `agent_cron` | No | Yes | Meta-skill for scheduling cron tasks (Android only) |

To enable `agent_cron` on a mobile host, declare it in your config:
```json
{ "skills": { "extra_builtins": ["agent_cron"] } }
```

A user skill with the same name as a builtin overrides it.

#### Directory Structure

```
<skills_dir>/<skill_name>/
└── SKILL.md    # Skill description and instruction guide
```

#### SKILL.md Format

```markdown
## Description
What this skill does — keep to one line, shown in every system prompt.

## Parameters
- param1: description of param1
- param2: description of param2

## Example
run_skills("skill_name", {"param1": "value1", "param2": "value2"})
```

## Architecture

```
┌──────────────────────────────────────────────┐
│                  MainAgent                     │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │ ReActLoop │  │  Prompt   │  │  Context  │  │
│  │ think→act │  │  Builder  │  │  Manager  │  │
│  └──────────┘  └──────────┘  └───────────┘  │
├──────────────────────────────────────────────┤
│                 Tool Registry                  │
│  Shell · Web · Code · Skills · Todo · Session │
│  Delegation · Multimodal · MCP                │
├──────────────────────────────────────────────┤
│              Shell Backend                     │
│  Native OS shell  |  fastshell sandbox        │
└──────────────────────────────────────────────┘
```

## Mobile Embedding (C ABI)

aacode-rs can be embedded into Android/iOS apps via a **handle-based C ABI**
(`src/ffi.rs`), compiled as a static library (`libaacode_rs.a`). The API is
platform-agnostic; each host provides a thin glue layer (Android: `jni_glue.c`,
iOS: a Swift bridge).

```c
typedef void (*aacode_event_fn)(const char *line, void *userdata);

void*  aacode_task_start(const char *task_json, aacode_event_fn cb, void *userdata); // non-blocking
char*  aacode_task_wait(void *handle);   // block until done, returns terminal JSON
void   aacode_task_cancel(void *handle); // non-blocking, per-handle
void   aacode_task_free(void *handle);
char*  aacode_validate_api_key(const char *config_json);
char*  aacode_list_sessions(const char *project_path);
char*  aacode_get_session_messages(const char *project_path, const char *session_id);
void   aacode_free_string(char *ptr);
```

* Each task is an opaque **handle** — `start` is non-blocking, `wait` blocks for
  the terminal result. Cancellation targets a single handle (no global state),
  so concurrent tasks (e.g. cron + chat) are naturally isolated.
* Events stream as JSONL through `cb(line, userdata)`; the callback context is
  per-task (`userdata`), so hosts need no thread-local or global trampoline.
* Early failures (bad JSON / missing task / session busy) emit an `error` event
  and a terminal result — they are never silent.
* The terminal event is an enriched `done`:
  `{"type":"done","session_id":...,"status":...,"iterations":...,"final_text":...}`,
  where `status` ∈ `completed | max_iterations | cancelled | error`.

See [ANDROID_INTEGRATION.md](../ANDROID_INTEGRATION.md) and
[IOS_INTEGRATION.md](../IOS_INTEGRATION.md) for host-specific integration.

## Core Capabilities

* **Shell Execution** — Safely execute any shell command as the universal adapter (all file I/O, code, system ops)
* **File Operations** — Read, write, and modify files in the project workspace via `run_shell`
* **Web Search & Fetch** — Search the web (SearXNG, Brave, Google CSE, Bing) and fetch URL content
* **Code Tools** — `execute_python` (system python3 / embedded RustPython), `run_tests`, `debug_code`, `analyze_code`
* **Task Management** — Todo lists with add/mark/update/summary, historical tracking
* **Session Management** — Create, switch, continue, list, and delete conversation sessions
* **Sub-Agent Delegation** — Delegate tasks to sub-agents with their own ReAct loop
* **Multimodal Understanding** — Analyze images, videos, and UI design drafts
* **MCP Protocol** — Connect to external MCP servers for extended tool capabilities
* **Extensible Skills** — Builtin skills + user skills via SKILL.md; add custom skills in the skills directory
* **LLM Compatibility** — OpenAI (GPT, DeepSeek, MiniMax, etc.) and Anthropic (Claude, Kimi, etc.)

## Usage Examples

### Example 1: Create Hello World

```bash
cargo run --release -- -p examples/hello_demo "Create a hello.py file with content print('Hello, World!')"
```

### Example 2: Develop Calculator

```bash
cargo run --release -- -p examples/calculator "Create a calculator program supporting addition, subtraction, multiplication, and division with test cases"
```

### Example 3: Web Application Development

```bash
cargo run --release -- -p examples/web_app "Create a simple web application with home and about pages"
```

### Example 4: Data Processing

```bash
cargo run --release -- -p examples/data_analysis "Create a data analysis script that reads CSV files in the project directory and generates statistical charts"
```

## Best Practices

### 1. Clear Task Descriptions

✅ **Good description**:
```
"Create a Python program that uses the requests library to fetch weather API data
and saves the results to a weather.json file"
```

❌ **Poor description**:
```
"Make a weather program"
```

### 2. Execute Complex Tasks in Steps

```bash
# Step 1: Create basic structure
cargo run --release -- -p examples/app "Create application basic structure"

# Step 2: Add features
cargo run --release -- -p examples/app "Add user authentication features"

# Step 3: Test
cargo run --release -- -p examples/app "Write tests for all features"
```

### 3. Use Project Guidelines

Edit an `init.md` file in your task directory and add project-specific rules:

```markdown
# Project Guidelines

## Code Style
- Use PEP 8 standards
- Function names use snake_case
- Class names use PascalCase

## Testing Requirements
- Every feature must have unit tests
- Test coverage must be at least 80%

## Documentation Requirements
- All public functions must have docstrings
- README must include usage examples
```

### 4. Rely on the Agent's Thinking

* The agent will analyze the project structure, read existing code, and adapt to conventions automatically.
* It can find solutions via `search_web` / `search_code`, install dependencies via `run_shell` in interactive mode.
* For complex tasks, ask step by step — the agent builds incrementally.
* With Plans toggle enabled, the agent presents a plan first, then executes.

## Security Features

* **Path confinment** — Restrict file access within project directory
* **Command safety** — Dangerous system command patterns are rejected
* **Sanbox isolation** — All operations run in a sandbox environment (fastshell backend)
* **Network permission system** — External network access requires explicit permission (mobile)

## Documentation

* [USAGE.md](USAGE.md) — Detailed usage guide with all CLI options and environment variables
* [design.md](design.md) — Architecture decisions and design rationale
* [LLM_CLIENT.md](LLM_CLIENT.md) — LLM protocol compatibility and streaming details
* [DEPS.md](DEPS.md) — Full dependency audit

## License

Copyright (c) 2024-2026 xiefujin <490021684@qq.com>. All rights reserved.

This project is initiated and developed by xiefujin (github: [kandada](https://github.com/kandada), email: 490021684@qq.com), licensed under **GPL-3.0**. All derivative works must also be open source under GPL. See [LICENSE](LICENSE).

## Contact

* Official Website: [https://aacode-ai.com](https://aacode-ai.com)
* Project Home: [xiefujin](https://github.com/kandada/aacode)
* Issue Reporting: [Issues](https://github.com/kandada/aacode/issues)
* Feature Suggestions: [Discussions](https://github.com/kandada/aacode/discussions)

---

<div align="center">

**Start your AI programming journey today!**

Made with ❤️ by [xiefujin](https://github.com/kandada)

</div>
