[English](README.md) |

# aacode-rs — CLI 编程 Agent

[![Rust](https://img.shields.io/badge/Rust-1.80+-orange.svg)](https://www.rust-lang.org/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](LICENSE)

> **纯 Rust 实现的 AI 编程 CLI Agent** — 轻量化 ReAct 架构，100% Rust，无 Python 依赖。

## 设计原则

* Shell 作为万能适配器 — 所有文件、代码、系统操作均通过 `run_shell` 完成
* 文件化上下文 — 动态发现，Markdown 文件作为主要存储
* 上下文管理 — token 预算耗尽时的智能压缩
* 文档型 Skills — SKILL.md 指令指南，无需内嵌脚本
* 分层工具系统 — 原子工具、管理工具、Skills 三层架构
* 安全护栏 — 路径限制、危险命令拒绝、网络权限控制
* 跨平台 — 同一代码库编译到 macOS、Linux、Windows、Android、iOS
* 零 LLM SDK — 异步 HTTP 流式（tokio + reqwest）+ 手写 SSE/JSON 解析，直接调用所有 LLM API

## 快速开始

### 操作系统

该项目主要在 macOS 和 Linux 上开发和测试，建议使用 macOS 或 Linux。Windows 也可使用。

### 构建

```bash
git clone https://github.com/kandada/fastshell.git
cd fastshell/aacode-rs
cp .env.example .env   # 编辑填入你的 API Key
cargo build --release

# 二进制文件在 target/release/aacode
```

### 开始使用

**特别说明**：启动任务之前，你可以在任务目录中建一个 `init.md` 文件，作为任务详细描述文件，尽可能详细描述你的设计思路，能得到更好的结果。

```bash
# 执行任务
cargo run --release -- -p examples/my_project "你的任务描述"

# 或手动运行
export LLM_API_KEY="your-api-key"
export LLM_API_URL="your-api-url"
export LLM_MODEL_NAME="your-model-name"
./target/release/aacode -p examples/my_project "你的任务描述"

# 高级模式
## 规划优先模式
cargo run --release -- -p examples/my_project "复杂任务" --plan-first

## 交互式连续对话
cargo run --release -- -p examples/my_project "初始任务" --interactive

## 指定会话
cargo run --release -- --session session_20250128_123456_0 "继续任务"
```

### 或通过 cargo 安装（推荐）

cargo install 后可使用 `aacode` 命令，默认工作目录为**当前目录**，无需指定 `-p` 即可直接使用。

```bash
# 安装
cargo install aacode-rs

# 进入交互会话模式（无需任务）
aacode

# 在当前目录执行单个任务
aacode "你的任务"

# 或明确使用 aacode run
aacode run "你的任务"

# 指定其他项目目录
aacode run -p /your/project/path "你的任务"
```

## 配置说明

### 大语言模型

支持 DeepSeek、OpenAI、Anthropic、Kimi、MiniMax 等所有兼容 OpenAI/Anthropic API 的模型。

```bash
# OpenAI
export LLM_API_KEY="your-openai-key"
export LLM_API_URL="https://api.openai.com/v1"
export LLM_MODEL_NAME="gpt-4"
export LLM_GATEWAY="openai"
export LLM_MULTIMODAL="false"

# 兼容 OpenAI API 的模型（DeepSeek 等）
export LLM_API_KEY="your-api-key"
export LLM_API_URL="https://your-api-endpoint/v1"
export LLM_MODEL_NAME="your-model-name"
export LLM_GATEWAY="openai"

# 兼容 Anthropic API 的模型（Claude、Kimi、MiniMax 等）
export LLM_API_KEY="your-api-key"
export LLM_API_URL="https://your-api-endpoint/v1"
export LLM_MODEL_NAME="your-model-name"
export LLM_GATEWAY="anthropic"
```

### Shell 后端

```bash
# 原生 OS shell（桌面默认，无需依赖）
export AACODE_SHELL_BACKEND="native"

# fastshell 沙箱（180+ 内置命令、VFS 隔离、内嵌 Python）
export AACODE_SHELL_BACKEND="fastshell"
```

### Skills 目录

```bash
# 启用内置 + 用户目录 Skills 模式（可选）
export AACODE_SKILLS_DIR="/path/to/skills"
```

### 多模态模型

支持多模态模型（Kimi K2.5、MiniMax M2.5 等），用于 `understand_image` / `understand_ui_design` 等工具。在 aacode_config.yaml 中配置：

```yaml
multimodal:
  name: "kimi-k2.5"
  api_key: "your-kimi-key"
  api_url: "https://api.moonshot.cn/v1"
  gateway: "anthropic"
```

### 搜索引擎

支持 SearXNG，需要用户自行部署并通过环境变量 `SEARCHXNG_URL` 配置。

### MCP

在 aacode_config.yaml 中配置 MCP 资源（支持 stdio 和 sse）。

### Skills（技能）

Skills 为文档型：`run_skills` 返回 SKILL.md 的指令内容，Agent 借助 `run_shell` 及其他工具执行。有两种发现模式：

| 模式 | 触发条件 | 来源 |
|---|---|---|
| **项目模式**（传统，桌面 CLI） | 未设置 `AACODE_SKILLS_DIR` | 扫描 `<project>/skills/` 和 `<project>/.aacode/skills/` |
| **用户目录模式**（移动端宿主） | 设置了 `AACODE_SKILLS_DIR` | 内置技能（编译进二进制） + `<skills_dir>/*/SKILL.md` |

在项目模式下，**不会注入任何内置技能** —— 你需要手动在项目的 `skills/` 目录下放置 SKILL.md。

在用户目录模式下，内置技能始终可用（编译进二进制，零文件依赖）：

| 内置技能 | 始终注入 | 受 `extra_builtins` 门控 | 说明 |
|---|---|---|---|
| `skill_creator` | 是 | 否 | 创建和更新 Skill 的元技能 |
| `book_writer` | 是 | 否 | 多阶段写书（大纲 → 故事线 → 逐章写作 → 审查） |
| `agent_cron` | 否 | 是 | 定时任务调度的元技能（仅 Android） |

如需在移动端启用 `agent_cron`，在配置中声明：
```json
{ "skills": { "extra_builtins": ["agent_cron"] } }
```

同名用户技能会覆盖内置技能。

#### 目录结构

```
<skills_dir>/<skill名>/
└── SKILL.md    # 技能描述和指令指南
```

#### SKILL.md 格式

```markdown
## Description
技能描述 —— 保持一行，会出现在每次的 system prompt 中。

## Parameters
- param1: 参数1描述
- param2: 参数2描述

## Example
run_skills("skill_name", {"param1": "value1", "param2": "value2"})
```

## 架构设计

```
┌──────────────────────────────────────────────┐
│                  MainAgent                     │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │ ReAct循环 │  │  提示词  │  │ 上下文管理│  │
│  │ 思考→动作 │  │  构建器  │  │  管理器   │  │
│  └──────────┘  └──────────┘  └───────────┘  │
├──────────────────────────────────────────────┤
│                 工具注册中心                    │
│  Shell · 网络 · 代码 · Skills · Todo · Session │
│  委托 · 多模态 · MCP                          │
├──────────────────────────────────────────────┤
│              Shell 后端                         │
│  原生 OS shell  |  fastshell 沙箱              │
└──────────────────────────────────────────────┘
```

## 移动端嵌入（C ABI）

aacode-rs 可通过**句柄式 C ABI**（`src/ffi.rs`）嵌入 Android/iOS 应用，编译为
静态库 `libaacode_rs.a`。API 与平台无关；各宿主只提供一层薄胶水
（Android：`jni_glue.c`；iOS：Swift 桥接）。

```c
typedef void (*aacode_event_fn)(const char *line, void *userdata);

void*  aacode_task_start(const char *task_json, aacode_event_fn cb, void *userdata); // 非阻塞
char*  aacode_task_wait(void *handle);   // 阻塞到结束，返回终态 JSON
void   aacode_task_cancel(void *handle); // 非阻塞，按句柄取消
void   aacode_task_free(void *handle);
char*  aacode_validate_api_key(const char *config_json);
char*  aacode_list_sessions(const char *project_path);
char*  aacode_get_session_messages(const char *project_path, const char *session_id);
void   aacode_free_string(char *ptr);
```

* 每个任务是**不透明句柄**——`start` 非阻塞，`wait` 阻塞取终态。取消只针对单个句柄
  （无全局态），因此并发任务（如 cron + chat）天然隔离。
* 事件以 JSONL 形式经 `cb(line, userdata)` 流式回调；回调上下文是每任务独立的
  （`userdata`），宿主无需线程局部存储或全局 trampoline。
* 早退失败（bad JSON / 缺 task / 会话忙）会发出 `error` 事件并返回终态——绝不静默。
* 终态事件为富化的 `done`：
  `{"type":"done","session_id":...,"status":...,"iterations":...,"final_text":...}`，
  其中 `status` ∈ `completed | max_iterations | cancelled | error`。

宿主侧集成见 [ANDROID_INTEGRATION.md](../ANDROID_INTEGRATION.md) 与
[IOS_INTEGRATION.md](../IOS_INTEGRATION.md)。

## 核心能力

* **Shell 执行** — 安全执行任意 shell 命令，作为万能适配器（处理所有文件 I/O、代码、系统操作）
* **文件操作** — 在项目工作区内通过 `run_shell` 读写和修改文件
* **网络搜索与抓取** — 搜索网络（SearXNG、Brave、Google CSE、Bing）并获取 URL 内容
* **代码工具** — `execute_python`（系统 python3 / 内嵌 RustPython）、`run_tests`、`debug_code`、`analyze_code`
* **任务管理** — 待办列表，支持添加/标记/更新/摘要，历史追踪
* **会话管理** — 创建、切换、继续、列出和删除对话会话
* **子代理委托** — 将任务委托给专业化子代理，拥有独立的 ReAct 循环
* **多模态理解** — 分析图片、视频和 UI 设计稿
* **MCP 协议** — 连接外部 MCP 服务器以扩展工具能力
* **可扩展 Skills** — 内置 Skills + 用户自定义 SKILL.md，在 Skills 目录中添加即可
* **LLM 兼容性** — OpenAI 系列（GPT、DeepSeek、MiniMax 等）和 Anthropic 系列（Claude、Kimi 等）

## 使用示例

### 示例1：创建 Hello World

```bash
cargo run --release -- -p examples/hello_demo "创建一个hello.py文件，内容为print('Hello, World!')"
```

### 示例2：开发计算器

```bash
cargo run --release -- -p examples/calculator "创建一个支持加减乘除的计算器程序，包含测试用例"
```

### 示例3：Web应用开发

```bash
cargo run --release -- -p examples/web_app "创建一个简单的Web应用，包含首页和关于页面"
```

### 示例4：数据处理

```bash
cargo run --release -- -p examples/data_analysis "创建一个数据分析脚本，读取项目目录中的CSV文件并生成统计图表"
```

## 最佳实践

### 1. 任务描述要清晰

✅ **好的描述**：
```
"创建一个Python程序，使用requests库获取天气API数据，
并将结果保存到weather.json文件"
```

❌ **不好的描述**：
```
"做个天气程序"
```

### 2. 分步骤执行复杂任务

对于复杂项目，分多次执行：

```bash
# 第一步：创建基础结构
cargo run --release -- -p examples/app "创建应用基础结构"

# 第二步：添加功能
cargo run --release -- -p examples/app "添加用户认证功能"

# 第三步：测试
cargo run --release -- -p examples/app "为所有功能编写测试"
```

### 3. 利用项目指导原则

在任务目录中编辑 `init.md` 文件，添加项目特定的规则和设计思路：

```markdown
# 项目指导原则

## 代码风格
- 使用PEP 8规范
- 函数名使用snake_case
- 类名使用PascalCase

## 测试要求
- 每个功能必须有单元测试
- 测试覆盖率不低于80%

## 文档要求
- 所有公共函数必须有docstring
- README必须包含使用示例
```

### 4. 相信 Agent 的自主思考能力

* Agent 会自动分析项目结构，阅读已有代码，适配项目风格。
* 需要外部信息时，它会通过 `search_web` / `search_code` 自行搜索；需要安装工具时，交互模式下会用 `run_shell` 完成。
* 复杂任务建议分步进行，Agent 会增量构建，每一步都验证后再继续。
* 开启 Plans 后 Agent 会先列出计划再执行，便于把控方向。

## 安全特性

* **路径限制** — 限制文件访问在项目目录内
* **命令安全** — 阻止危险系统命令执行
* **沙箱隔离** — 所有操作在沙箱环境中进行（fastshell 后端）
* **网络权限控制** — 外网访问需显式授权（移动端）

## 文档

* [USAGE.md](USAGE.md) — 详细使用指南，含全部 CLI 选项和环境变量说明
* [design.md](design.md) — 架构决策与设计考量
* [LLM_CLIENT.md](LLM_CLIENT.md) — LLM 协议兼容性与流式解析说明
* [DEPS.md](DEPS.md) — 完整依赖审计

## 许可证

Copyright (c) 2024-2026 xiefujin <490021684@qq.com>. All rights reserved.

本项目为 xiefujin（github：[kandada](https://github.com/kandada)，邮箱：490021684@qq.com）发起并开发，采用 **GPL-3.0** 许可证。所有衍生作品必须同样以 GPL 开源。详见 [LICENSE](LICENSE)。

## 联系方式

* 官方网站：[https://aacode-ai.com](https://aacode-ai.com)
* 项目主页：[xiefujin](https://github.com/kandada/aacode)
* 问题反馈：[Issues](https://github.com/kandada/aacode/issues)
* 功能建议：[Discussions](https://github.com/kandada/aacode/discussions)

---

<div align="center">

**立即开始你的 AI 编程之旅！**

Made with ❤️ by [xiefujin](https://github.com/kandada)

</div>
