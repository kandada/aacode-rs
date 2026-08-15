// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Concurrency integration tests: multiple agent tasks running in parallel
//! through the real C ABI (`aacode_task_start` / `aacode_task_wait` /
//! `aacode_task_cancel`), verifying stream isolation, per-task cancellation,
//! and same-session rejection.

use aacode_rs::ffi::{aacode_free_string, aacode_task_cancel, aacode_task_start, aacode_task_wait};
use serde_json::json;
use std::ffi::{c_char, c_void, CStr, CString};
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

fn task_json(addr: &str, project: &std::path::Path, task: &str, session_id: Option<&str>) -> CString {
    let mut v = json!({
        "task": task,
        "project_path": project.to_string_lossy(),
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

fn wait_and_free(h: *mut aacode_rs::ffi::AacodeTask) -> String {
    let ptr = aacode_task_wait(h);
    let out = unsafe { CStr::from_ptr(ptr).to_str().unwrap().to_string() };
    aacode_free_string(ptr);
    aacode_rs::ffi::aacode_task_free(h);
    out
}

fn run_ffi(task: &CString, cb: extern "C" fn(*const c_char, *mut c_void)) -> String {
    let h = aacode_task_start(task.as_ptr(), Some(cb), std::ptr::null_mut());
    wait_and_free(h)
}

// Per-test static collectors (extern "C" fns can't capture closures).
macro_rules! collector {
    ($buf:ident, $cb:ident) => {
        static $buf: Mutex<Vec<String>> = Mutex::new(Vec::new());
        extern "C" fn $cb(line: *const c_char, _userdata: *mut c_void) {
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

    let ta = task_json(&mock_a.addr, &proj_a, "task alpha", None);
    let tb = task_json(&mock_b.addr, &proj_b, "task bravo", None);

    // Start both concurrently, then wait.
    let h_a = aacode_task_start(ta.as_ptr(), Some(cb_alpha), std::ptr::null_mut());
    let h_b = aacode_task_start(tb.as_ptr(), Some(cb_bravo), std::ptr::null_mut());
    let ra = wait_and_free(h_a);
    let rb = wait_and_free(h_b);

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

    let ts = task_json(&mock_slow.addr, &proj_slow, "slow task", None);
    let tf = task_json(&mock_fast.addr, &proj_fast, "fast task", None);

    let h_slow = aacode_task_start(ts.as_ptr(), Some(cb_slow), std::ptr::null_mut());
    // Give SLOW a moment to enter the sleep, then cancel ONLY it (by handle).
    thread::sleep(Duration::from_millis(600));
    aacode_task_cancel(h_slow);

    // FAST starts after the cancel of SLOW — cancellation must be per-handle,
    // never global.
    let h_fast = aacode_task_start(tf.as_ptr(), Some(cb_fast), std::ptr::null_mut());

    let rs = wait_and_free(h_slow);
    let rf = wait_and_free(h_fast);

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

collector!(BUF_DONE, cb_done);

#[test]
fn terminal_done_event_carries_status_and_final_text() {
    // The enriched `done` event (not just the return value) must carry the
    // terminal outcome: status, iterations, final_text.
    let mock = MockLlm::start(vec![sse_content("DONE_MARKER", "stop")]);
    let proj = tmp_project("done_enrich");
    let t = task_json(&mock.addr, &proj, "enrich task", None);
    let r = run_ffi(&t, cb_done);
    assert!(r.contains("completed"), "result: {r}");

    let lines = BUF_DONE.lock().unwrap();
    let done_line = lines
        .iter()
        .find(|l| l.contains(r#""type":"done""#))
        .cloned()
        .unwrap_or_default();
    assert!(done_line.contains(r#""status":"completed""#), "done line: {done_line}");
    assert!(done_line.contains("DONE_MARKER"), "final_text missing: {done_line}");
}

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

    let t1 = task_json(&mock1.addr, &proj, "task one", Some(&sid));
    let t2 = task_json(&mock2.addr, &proj, "task two", Some(&sid));

    let h1 = aacode_task_start(t1.as_ptr(), Some(cb_s1), std::ptr::null_mut());
    thread::sleep(Duration::from_millis(500));
    let start = std::time::Instant::now();
    let r2 = run_ffi(&t2, cb_s2);
    let elapsed = start.elapsed();

    assert!(
        r2.contains("already has a task running"),
        "second task must be rejected, got: {r2}"
    );
    assert!(elapsed.as_millis() < 500, "rejection must be immediate");

    let r1 = wait_and_free(h1);
    assert!(r1.contains("completed"), "first task unaffected: {r1}");

    // After t1 finishes, the session is free again.
    let mock3 = MockLlm::start(vec![sse_content("t3 done", "stop")]);
    let t3 = task_json(&mock3.addr, &proj, "task three", Some(&sid));
    let r3 = run_ffi(&t3, cb_s2);
    assert!(r3.contains("completed"), "session must be reusable: {r3}");
}
