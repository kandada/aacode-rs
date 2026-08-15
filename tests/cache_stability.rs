// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Cache-stability integration tests: verify that the message prefix is
//! byte-stable across execute() calls so the provider KV/prefix cache stays hot.
//!
//! Key invariants:
//!   1. messages[0] (system prompt) is fully static — init.md, skills, etc.
//!      are all baked in once and never change between execute() calls.
//!   2. No dynamic system messages are appended after the user task — every
//!      execute() call produces an identical messages[0] prefix.
//!   3. Compact view prefix is stable within a single run.

use aacode_rs::config::AgentConfig;
use aacode_rs::llm::types::ChatMessage;
use aacode_rs::runtime::AgentRuntime;
use aacode_rs::session::SessionManager;
use aacode_rs::stream::CollectingSink;
use std::sync::atomic::AtomicBool;
use std::thread;

// ───────── helpers ─────────

fn tmp_project() -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!(
        "aacode_cs_{}_{}",
        std::process::id(),
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn sse_content(text: &str, finish: &str) -> String {
    format!(
        "data: {{\"choices\":[{{\"delta\":{{\"content\":{text:?}}}}}]}}\n\n\
         data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":{finish:?}}}]}}\n\n\
         data: [DONE]\n\n"
    )
}

/// A simple mock LLM server returning scripted SSE responses.
struct MockLlm {
    addr: String,
    _handle: thread::JoinHandle<()>,
}

impl MockLlm {
    fn start(responses: Vec<String>) -> Self {
        let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let addr = format!("http://{}", server.server_addr());
        let handle = thread::spawn(move || {
            let mut queue = responses.into_iter();
            for mut request in server.incoming_requests() {
                let mut buf = String::new();
                let _ = request.as_reader().read_to_string(&mut buf);
                let body = queue.next().unwrap_or_else(|| "data: [DONE]\n\n".to_string());
                let header =
                    tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"text/event-stream"[..])
                        .unwrap();
                let response = tiny_http::Response::from_string(body).with_header(header);
                let _ = request.respond(response);
                if queue.len() == 0 {
                    break;
                }
            }
        });
        MockLlm {
            addr,
            _handle: handle,
        }
    }
}

fn config_for(addr: &str) -> AgentConfig {
    let mut cfg = AgentConfig::default();
    cfg.model.name = "mock-model".into();
    cfg.model.api_key = Some("sk-test".into());
    cfg.model.base_url = Some(format!("{addr}/v1"));
    cfg.model.gateway = aacode_rs::config::Gateway::Openai;
    cfg
}

fn extract_session_id(sink: &CollectingSink) -> String {
    let lines = sink.lines();
    let sc: Vec<_> = lines
        .iter()
        .filter(|l| l.contains(r#""type":"session_created""#))
        .collect();
    assert!(!sc.is_empty(), "session_created event missing");
    let line = sc[0];
    let start = line.find(r#""session_id":""#).unwrap() + r#""session_id":""#.len();
    let end = line[start..].find('"').unwrap();
    line[start..start + end].to_string()
}

// ───────── tests ─────────

#[tokio::test]
async fn execute_produces_user_and_assistant_only() {
    let mock = MockLlm::start(vec![sse_content("Task completed.", "stop")]);
    let cfg = config_for(&mock.addr);
    let proj = tmp_project();
    let rt = AgentRuntime::init(cfg, proj.clone()).unwrap();
    let sink = CollectingSink::new(false);
    let cancel = AtomicBool::new(false);

    let res = rt.run_task("analyze the project", None, &sink, &cancel).await.unwrap();
    assert_eq!(format!("{:?}", res.status), "Completed");

    let sid = extract_session_id(&sink);
    let sm = SessionManager::new(&proj);
    let msgs = sm.read_session_messages(&sid);
    assert!(!msgs.is_empty());

    // Session contains only user + assistant (no extra system messages).
    // init.md is now in messages[0] (not persisted), project analysis is removed.
    assert!(
        msgs.iter().any(|m| m.role == "user" && m.content.contains("analyze the project"))
    );
    assert!(msgs.iter().any(|m| m.role == "assistant"));

    // No dynamic system messages should be persisted.
    let system_count = msgs.iter().filter(|m| m.role == "system").count();
    assert_eq!(system_count, 0, "no system messages should be persisted (found {system_count})");
}

#[tokio::test]
async fn consecutive_tasks_preserve_history() {
    let mock = MockLlm::start(vec![
        sse_content("First task done.", "stop"),
        sse_content("Second task done.", "stop"),
    ]);
    let cfg = config_for(&mock.addr);
    let proj = tmp_project();
    let rt = AgentRuntime::init(cfg, proj.clone()).unwrap();
    let cancel = AtomicBool::new(false);

    // Task 1
    let sink1 = CollectingSink::new(false);
    let res1 = rt.run_task("task one", None, &sink1, &cancel).await.unwrap();
    assert_eq!(format!("{:?}", res1.status), "Completed");
    let sid = extract_session_id(&sink1);

    // Task 2 — continues the same session
    let sink2 = CollectingSink::new(false);
    let res2 = rt.run_task("task two", Some(&sid), &sink2, &cancel).await.unwrap();
    assert_eq!(format!("{:?}", res2.status), "Completed");

    // Read final session state.
    let sm = SessionManager::new(&proj);
    let msgs = sm.read_session_messages(&sid);

    // Session should have messages from both tasks.
    let user_count = msgs.iter().filter(|m| m.role == "user").count();
    assert!(user_count >= 2, "session should have >=2 user messages, got {user_count}");

    // Each task should produce an assistant response.
    let assistant_count = msgs.iter().filter(|m| m.role == "assistant").count();
    assert!(assistant_count >= 2, "session should have >=2 assistant messages, got {assistant_count}");

    // No dynamic system messages should be injected into the session.
    let system_count = msgs.iter().filter(|m| m.role == "system").count();
    assert_eq!(system_count, 0, "no system messages should be persisted (found {system_count})");

    // Task 1 messages must appear before task 2's user message.
    let last_task1_user = msgs
        .iter()
        .position(|m| m.role == "user" && m.content.contains("task one"))
        .unwrap();
    let last_task2_user = msgs
        .iter()
        .rposition(|m| m.role == "user" && m.content.contains("task two"))
        .unwrap();
    assert!(
        last_task1_user < last_task2_user,
        "task 1 messages must precede task 2 messages"
    );
}

#[test]
fn compact_view_prefix_stable_within_call() {
    use aacode_rs::agent::compact::{build_compact_view_cached, CompactCache};
    use aacode_rs::config::ContextConfig;
    use aacode_rs::llm::types::ToolCall;

    fn convo(rounds: usize) -> Vec<ChatMessage> {
        let mut v = vec![ChatMessage::system("STATIC_SYSTEM_PROMPT")];
        for i in 0..rounds {
            v.push(ChatMessage::user(format!("ask {i} {}", "x".repeat(200))));
            v.push(ChatMessage::assistant_with_tools(
                "",
                vec![ToolCall {
                    id: format!("c{i}"),
                    name: "run_shell".into(),
                    arguments: "{}".into(),
                }],
            ));
            v.push(ChatMessage::tool_result(format!("c{i}"), "y".repeat(200)));
        }
        v
    }

    fn view_prefix(view: &[ChatMessage]) -> Vec<String> {
        view.iter()
            .map(|m| {
                format!(
                    "{}|{}|{:?}",
                    m.role,
                    m.content.chars().take(50).collect::<String>(),
                    m.tool_calls.as_ref().map(|t| t.len())
                )
            })
            .collect()
    }

    let cfg = ContextConfig {
        compact_trigger_tokens: 10,
        protect_first_rounds: 1,
        keep_last_rounds: 2,
        protect_last_user_rounds: 1,
        ..Default::default()
    };

    let mut cache: Option<CompactCache> = None;
    let mut msgs = convo(12);

    let (v1, c1, _) = build_compact_view_cached(&msgs, &cfg, &mut cache);
    let v1 = v1.into_owned();
    assert!(c1, "should trigger compaction");
    assert!(cache.is_some(), "cache must be populated");

    // Simulate another agent turn (append-only growth).
    msgs.push(ChatMessage::user("follow-up question"));
    let (v2, _, _) = build_compact_view_cached(&msgs, &cfg, &mut cache);
    let v2 = v2.into_owned();
    assert!(v2.len() > v1.len(), "view should grow");

    let p1 = view_prefix(&v1);
    let p2 = view_prefix(&v2);
    assert_eq!(
        p1,
        p2[..p1.len()],
        "compacted view prefix must be stable across iterations"
    );

    // Another iteration — still stable.
    msgs.push(ChatMessage::user("another follow-up"));
    let (v3, _, _) = build_compact_view_cached(&msgs, &cfg, &mut cache);
    let v3 = v3.into_owned();
    let p3 = view_prefix(&v3);
    assert_eq!(
        p2,
        p3[..p2.len()],
        "second append must also extend prefix stably"
    );
}

#[test]
fn system_prompt_includes_init_md_not_analysis() {
    use aacode_rs::agent::MainAgent;
    use aacode_rs::tools::ShellBackend;
    use aacode_rs::tools::backend::NativeShell;
    use std::sync::Arc;

    let proj = tmp_project();
    let mut cfg = AgentConfig::default();
    cfg.model.api_key = Some("sk-test".into());
    let backend: Arc<dyn ShellBackend> = Arc::new(NativeShell::new());
    let agent = MainAgent::new(cfg, proj.clone(), backend);
    let _reg = agent.build_registry();

    // We can't call build_system_prompt directly (it's private), but we
    // verify through ContextManager that init.md exists and project analysis
    // is available (just no longer injected per-call).
    let ctx = aacode_rs::context::ContextManager::new(&proj);
    let init = ctx.load_init_instructions();
    assert!(!init.is_empty(), "init.md should be available");

    // Project analysis still exists as a method but is no longer called
    // by execute().
    let analysis = ctx.analyze_project_structure();
    assert!(!analysis.is_empty(), "analyze_project_structure should still work");

    // Verified by main_agent.rs unit test:
    //   system_prompt_has_static_content — init IS in build_system_prompt()
    //   project analysis is NOT in build_system_prompt() or execute()
}
