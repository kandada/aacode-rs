// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Concurrency integration tests: multiple agent tasks running in parallel
//! through the real C ABI (`aacode_run_task_with_cb`), verifying stream
//! isolation, per-task cancellation, and same-session rejection.

use aacode_rs::ffi::{aacode_cancel_task, aacode_run_task_with_cb};
use serde_json::json;
use std::ffi::{c_char, CStr, CString};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

// ───────────────────────── mock LLM server ─────────────────────────

struct MockLlm {
    addr: String,
    _handle: thread::JoinHandle<()>,
}

impl MockLlm {
    fn start(responses: Vec<String>) -> MockLlm {
        let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let addr = format!("http://{}", server.server_addr());
        let handle = thread::spawn(move || {
            let mut queue = responses.into_iter();
            for mut request in server.incoming_requests() {
                let mut buf = String::new();
                let _ = request.as_reader().read_to_string(&mut buf);
                let body = queue
                    .next()
                    .unwrap_or_else(|| "data: [DONE]\n\n".to_string());
                let header = tiny_http::Header::from_bytes(
                    &b"Content-Type"[..],
                    &b"text/event-stream"[..],
                )
                .unwrap();
                let response = tiny_http::Response::from_string(body).with_header(header);
                let _ = request.respond(response);
            }
        });
        MockLlm {
            addr,
            _handle: handle,
        }
    }
}

fn sse_content(text: &str, finish: &str) -> String {
    format!(
        "data: {{\"choices\":[{{\"delta\":{{\"content\":{text:?}}}}}]}}\n\n\
         data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":{finish:?}}}]}}\n\n\
         data: [DONE]\n\n"
    )
}

fn sse_tool_call(id: &str, name: &str, args: &str) -> String {
    let args_json = serde_json::to_string(args).unwrap();
    format!(
        "data: {{\"choices\":[{{\"delta\":{{\"tool_calls\":[{{\"index\":0,\"id\":{id:?},\"function\":{{\"name\":{name:?},\"arguments\":{args_json}}}}}]}}}}]}}\n\n\
         data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"tool_calls\"}}]}}\n\n\
         data: [DONE]\n\n"
    )
}

fn tmp_project(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!(
        "aacode_cc_{}_{}_{}",
        tag,
        std::process::id(),
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn task_json(addr: &str, project: &std::path::Path, task: &str, task_id: &str, session_id: Option<&str>) -> CString {
    let mut v = json!({
        "task": task,
        "project_path": project.to_string_lossy(),
        "client_task_id": task_id,
        "model": {
            "name": "mock-model",
            "api_key": "sk-test",
            "base_url": format!("{addr}/v1"),
            "gateway": "openai",
        }
    });
    if let Some(sid) = session_id {
        v["session_id"] = json!(sid);
    }
    CString::new(v.to_string()).unwrap()
}

fn run_ffi(task: &CString, cb: extern "C" fn(*const c_char)) -> String {
    let ptr = aacode_run_task_with_cb(task.as_ptr(), cb);
    let out = unsafe { CStr::from_ptr(ptr).to_str().unwrap().to_string() };
    aacode_rs::ffi::aacode_free_string(ptr);
    out
}

// Per-test static collectors (extern "C" fns can't capture closures).
macro_rules! collector {
    ($buf:ident, $cb:ident) => {
        static $buf: Mutex<Vec<String>> = Mutex::new(Vec::new());
        extern "C" fn $cb(line: *const c_char) {
            if line.is_null() {
                return;
            }
            let s = unsafe { CStr::from_ptr(line) }.to_string_lossy().to_string();
            $buf.lock().unwrap().push(s);
        }
    };
}

// ─────────────────────────── tests ───────────────────────────

collector!(BUF_ALPHA, cb_alpha);
collector!(BUF_BRAVO, cb_bravo);

#[test]
fn parallel_tasks_have_isolated_streams() {
    // Two tasks with distinct mock LLMs and distinct callbacks run at the
    // same time. Each stream must contain only its own content.
    let mock_a = MockLlm::start(vec![sse_content("MARKER_ALPHA done", "stop")]);
    let mock_b = MockLlm::start(vec![sse_content("MARKER_BRAVO done", "stop")]);
    let proj_a = tmp_project("a");
    let proj_b = tmp_project("b");

    let ta = task_json(&mock_a.addr, &proj_a, "task alpha", "cc_alpha", None);
    let tb = task_json(&mock_b.addr, &proj_b, "task bravo", "cc_bravo", None);

    let ha = thread::spawn(move || run_ffi(&ta, cb_alpha));
    let hb = thread::spawn(move || run_ffi(&tb, cb_bravo));
    let ra = ha.join().unwrap();
    let rb = hb.join().unwrap();

    assert!(ra.contains("completed"), "alpha result: {ra}");
    assert!(rb.contains("completed"), "bravo result: {rb}");

    let a_lines = BUF_ALPHA.lock().unwrap().join("\n");
    let b_lines = BUF_BRAVO.lock().unwrap().join("\n");
    assert!(a_lines.contains("MARKER_ALPHA"), "alpha stream: {a_lines}");
    assert!(b_lines.contains("MARKER_BRAVO"), "bravo stream: {b_lines}");
    // The crucial isolation property:
    assert!(!a_lines.contains("MARKER_BRAVO"), "alpha stream polluted by bravo");
    assert!(!b_lines.contains("MARKER_ALPHA"), "bravo stream polluted by alpha");
}

collector!(BUF_SLOW, cb_slow);
collector!(BUF_FAST, cb_fast);

#[test]
fn cancel_one_task_does_not_affect_the_other() {
    // Task SLOW executes `sleep 3` via run_shell; we cancel it mid-flight.
    // Task FAST runs concurrently and must complete untouched.
    let mock_slow = MockLlm::start(vec![
        sse_tool_call("c1", "run_shell", "{\"command\":\"sleep 3\"}"),
        sse_content("slow finished (should not reach)", "stop"),
    ]);
    let mock_fast = MockLlm::start(vec![sse_content("fast done", "stop")]);
    let proj_slow = tmp_project("slow");
    let proj_fast = tmp_project("fast");

    let ts = task_json(&mock_slow.addr, &proj_slow, "slow task", "cc_slow", None);
    let tf = task_json(&mock_fast.addr, &proj_fast, "fast task", "cc_fast", None);

    let hs = thread::spawn(move || run_ffi(&ts, cb_slow));
    // Give SLOW a moment to enter the sleep, then cancel ONLY it.
    thread::sleep(Duration::from_millis(600));
    let id = CString::new("cc_slow").unwrap();
    aacode_cancel_task(id.as_ptr());

    // FAST starts after the cancel of SLOW — a global-cancel bug would have
    // wiped it (the old code also reset flags on new task start; both broken
    // behaviors are covered here).
    let hf = thread::spawn(move || run_ffi(&tf, cb_fast));

    let rs = hs.join().unwrap();
    let rf = hf.join().unwrap();

    assert!(
        rs.contains("cancelled"),
        "slow task must be cancelled, got: {rs}"
    );
    assert!(
        rf.contains("completed"),
        "fast task must complete despite slow's cancel, got: {rf}"
    );
}

collector!(BUF_S1, cb_s1);
collector!(BUF_S2, cb_s2);

#[test]
fn same_session_second_task_rejected() {
    // Task 1 holds session "sess_x" (sleeping in a shell command); task 2 on
    // the SAME project+session must be rejected immediately.
    let mock1 = MockLlm::start(vec![
        sse_tool_call("c1", "run_shell", "{\"command\":\"sleep 2\"}"),
        sse_content("t1 done", "stop"),
    ]);
    let mock2 = MockLlm::start(vec![sse_content("t2 done", "stop")]);
    let proj = tmp_project("sess");

    // Pre-create the session so both tasks address the same id.
    let sid = {
        let mut sm = aacode_rs::session::SessionManager::new(&proj);
        sm.create_session("seed", None).unwrap()
    };

    let t1 = task_json(&mock1.addr, &proj, "task one", "cc_s1", Some(&sid));
    let t2 = task_json(&mock2.addr, &proj, "task two", "cc_s2", Some(&sid));

    let h1 = thread::spawn(move || run_ffi(&t1, cb_s1));
    thread::sleep(Duration::from_millis(500));
    let start = std::time::Instant::now();
    let r2 = run_ffi(&t2, cb_s2);
    let elapsed = start.elapsed();

    assert!(
        r2.contains("already has a task running"),
        "second task must be rejected, got: {r2}"
    );
    assert!(elapsed.as_millis() < 500, "rejection must be immediate");

    let r1 = h1.join().unwrap();
    assert!(r1.contains("completed"), "first task unaffected: {r1}");

    // After t1 finishes, the session is free again.
    let mock3 = MockLlm::start(vec![sse_content("t3 done", "stop")]);
    let t3 = task_json(&mock3.addr, &proj, "task three", "cc_s3", Some(&sid));
    let r3 = run_ffi(&t3, cb_s2);
    assert!(r3.contains("completed"), "session must be reusable: {r3}");
}
