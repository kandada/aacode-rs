// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Web tools — search_web, fetch_url, search_code.
//!
//! Ported from Python `tools/web_tools.py`. Multi-backend search with
//! automatic fallback: SearXNG → Brave → Google CSE → Bing → SerpAPI →
//! HTML scrape (Bing + Sogou, raced in parallel).
//!
//! Robustness features (mobile-first):
//!   * Separate connect timeout (2s) so a dead/unreachable SearXNG host is
//!     detected quickly instead of eating the whole request timeout.
//!   * Per-engine circuit breaker: after a transport failure the engine is
//!     skipped for a cool-down window instead of being retried on every call.
//!   * A global time budget (deadline) shared by all engines, so the total
//!     latency is bounded no matter how many backends are configured.
//!   * The HTML fallback scrapers run in parallel; first non-empty wins.

use super::registry::Tool;
use super::schema::{ParamType, ToolParameter, ToolSchema};
use crate::config::SearchConfig;
use crate::error::{AacodeError, Result};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

/// Max time (secs) to establish a TCP+TLS connection before declaring the
/// engine unreachable. Keeps dead LAN SearXNG hosts from stalling searches.
const CONNECT_TIMEOUT_SECS: u64 = 2;
/// Cap for any single engine attempt, regardless of the overall budget.
const PER_ENGINE_CAP_SECS: f64 = 4.0;
/// How long a failing engine's circuit stays open (attempts skipped).
const CIRCUIT_OPEN_SECS: f64 = 120.0;
/// Consecutive transport failures before the circuit opens.
const FAILURES_TO_OPEN: u32 = 1;
/// Minimum window always granted to the final HTML-scrape fallback.
const SCRAPE_MIN_SECS: f64 = 3.0;

static HTTP_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(CONNECT_TIMEOUT_SECS))
        .build()
        .expect("reqwest client")
});

// ──────────────────────────── search_web ────────────────────────────────

/// Mutable health/rate state for one engine (guarded by the global Mutex).
#[derive(Default, Clone)]
struct EngineHealth {
    /// Consecutive transport failures.
    failures: u32,
    /// Engine is skipped until this epoch-seconds timestamp.
    open_until: f64,
    /// Last request timestamp (rate limiting).
    last_call: f64,
}

/// Process-wide engine health map, shared by ALL SearchWebTool instances.
/// With concurrent agent tasks (multi-session), one task discovering that
/// SearXNG is down immediately benefits every other task.
static ENGINE_HEALTH: std::sync::OnceLock<Mutex<std::collections::HashMap<String, EngineHealth>>> =
    std::sync::OnceLock::new();

fn engine_health() -> &'static Mutex<std::collections::HashMap<String, EngineHealth>> {
    ENGINE_HEALTH.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

pub struct SearchWebTool {
    pub cfg: SearchConfig,
    pub timeout_secs: u64,
}

/// Outcome of a single engine attempt (for diagnostics + breaker updates).
enum Attempt {
    Ok(Vec<Value>),
    Empty,
    TransportError(String),
    Skipped(&'static str),
}

impl SearchWebTool {
    pub fn new(cfg: SearchConfig, timeout_secs: u64) -> Self {
        SearchWebTool { cfg, timeout_secs }
    }

    /// Pick the best available engine from config.
    fn choose_best_engine(&self) -> &'static str {
        let det = detect_engine_type(self.cfg.searxng_url.as_deref().unwrap_or(""));
        match det {
            "brave" if self.cfg.brave_api_key.is_some() => "brave",
            "google_cse" if self.cfg.google_cse_key.is_some() => "google_cse",
            "bing" if self.cfg.bing_api_key.is_some() => "bing",
            "serpapi" if self.cfg.serpapi_key.is_some() => "serpapi",
            _ => {
                if self.cfg.searxng_url.is_some() {
                    "searxng"
                } else {
                    ""
                }
            }
        }
    }

    /// True if the engine's circuit is currently open (recently failed).
    fn circuit_open(&self, engine: &str) -> bool {
        let guard = engine_health().lock().unwrap_or_else(|e| e.into_inner());
        guard
            .get(engine)
            .map(|h| current_time() < h.open_until)
            .unwrap_or(false)
    }

    /// Record the outcome of an engine attempt for the circuit breaker.
    fn record_outcome(&self, engine: &str, ok: bool) {
        let mut guard = engine_health().lock().unwrap_or_else(|e| e.into_inner());
        let h = guard.entry(engine.to_string()).or_default();
        if ok {
            h.failures = 0;
            h.open_until = 0.0;
        } else {
            h.failures += 1;
            if h.failures >= FAILURES_TO_OPEN {
                h.open_until = current_time() + CIRCUIT_OPEN_SECS;
            }
        }
    }

    /// Rate limit: sleep (bounded by the remaining budget) until the engine's
    /// minimum interval has elapsed since its previous call.
    async fn enforce_rate_limit(&self, engine: &str, remaining: f64) {
        let do_wait = {
            let mut guard = engine_health().lock().unwrap_or_else(|e| e.into_inner());
            let rate = match engine {
                "searxng" => 0.5,
                _ => 1.0,
            };
            let now = current_time();
            let h = guard.entry(engine.to_string()).or_default();
            if h.last_call > 0.0 {
                let wait = rate - (now - h.last_call);
                if wait > 0.0 {
                    let wait = wait.min(remaining.max(0.0));
                    Some(wait)
                } else {
                    h.last_call = current_time();
                    None
                }
            } else {
                h.last_call = current_time();
                None
            }
        };
        if let Some(wait) = do_wait {
            tokio::time::sleep(Duration::from_secs_f64(wait)).await;
            let mut guard = engine_health().lock().unwrap_or_else(|e| e.into_inner());
            guard.entry(engine.to_string()).or_default().last_call = current_time();
        }
    }

    /// Try all configured backends in priority order, with fallback.
    /// Returns (success, engine, results, engines_tried diagnostics).
    async fn search_with_fallback(
        &self,
        query: &str,
        max_results: usize,
        timeout: u64,
        cancel: &AtomicBool,
    ) -> (bool, String, Vec<Value>, Vec<String>) {
        let deadline = current_time() + timeout as f64;
        let mut tried: Vec<String> = Vec::new();

        let mut engine = self.choose_best_engine();
        if engine.is_empty() {
            engine = "searxng"; // default if nothing configured
        }

        // Engine priority: detected engine first, then the others.
        let mut order: Vec<&str> = vec![engine];
        for fb in ["searxng", "brave", "google_cse", "bing", "serpapi"] {
            if fb != engine {
                order.push(fb);
            }
        }

        for eng in order {
            if cancel.load(Ordering::Relaxed) {
                tried.push(format!("{eng}:cancelled"));
                return (false, eng.to_string(), vec![], tried);
            }
            let remaining = deadline - current_time();
            if remaining <= 0.5 {
                tried.push(format!("{eng}:skipped(budget)"));
                continue;
            }
            match self.try_engine(eng, query, max_results, remaining).await {
                Attempt::Ok(results) => {
                    tried.push(format!("{eng}:ok"));
                    self.record_outcome(eng, true);
                    return (true, eng.to_string(), results, tried);
                }
                Attempt::Empty => {
                    tried.push(format!("{eng}:empty"));
                    self.record_outcome(eng, true); // reachable, just no hits
                }
                Attempt::TransportError(e) => {
                    tried.push(format!("{eng}:error({})", brief_err(&e)));
                    self.record_outcome(eng, false);
                }
                Attempt::Skipped(why) => {
                    tried.push(format!("{eng}:skipped({why})"));
                }
            }
        }

        // HTML fallback scrape (Bing + Sogou raced in parallel). Always grant
        // it a minimum window even if the engines used up the budget.
        if !cancel.load(Ordering::Relaxed) {
            let remaining = (deadline - current_time()).max(SCRAPE_MIN_SECS);
            if let Some((name, results)) = fallback_scrape(query, max_results, remaining).await {
                if !results.is_empty() {
                    tried.push(format!("{name}:ok"));
                    return (true, name, results, tried);
                }
            }
            tried.push("fallback_scrape:empty".to_string());
        }

        (false, engine.to_string(), vec![], tried)
    }

    async fn try_engine(
        &self,
        engine: &str,
        query: &str,
        max_results: usize,
        remaining: f64,
    ) -> Attempt {
        // Unconfigured engines are skipped outright (no key / no URL).
        let configured = match engine {
            "searxng" => self.cfg.searxng_url.is_some(),
            "brave" => self.cfg.brave_api_key.is_some(),
            "google_cse" => self.cfg.google_cse_key.is_some() && self.cfg.google_cse_cx.is_some(),
            "bing" => self.cfg.bing_api_key.is_some(),
            "serpapi" => self.cfg.serpapi_key.is_some(),
            _ => false,
        };
        if !configured {
            return Attempt::Skipped("not-configured");
        }
        if self.circuit_open(engine) {
            return Attempt::Skipped("circuit-open");
        }
        self.enforce_rate_limit(engine, remaining).await;
        let budget = remaining.min(PER_ENGINE_CAP_SECS);
        let r = match engine {
            "searxng" => searxng_search(
                self.cfg.searxng_url.as_deref().unwrap_or("http://localhost:8080"),
                query,
                max_results,
                budget,
            ).await,
            "brave" => brave_search(
                self.cfg.brave_api_key.as_deref().unwrap_or(""),
                query,
                max_results,
                budget,
            ).await,
            "google_cse" => google_cse_search(
                self.cfg.google_cse_key.as_deref().unwrap_or(""),
                self.cfg.google_cse_cx.as_deref().unwrap_or(""),
                query,
                max_results,
                budget,
            ).await,
            "bing" => bing_search(
                self.cfg.bing_api_key.as_deref().unwrap_or(""),
                query,
                max_results,
                budget,
            ).await,
            "serpapi" => serpapi_search(
                self.cfg.serpapi_key.as_deref().unwrap_or(""),
                query,
                max_results,
                budget,
            ).await,
            _ => return Attempt::Skipped("unknown-engine"),
        };
        match r {
            Ok(v) if !v.is_empty() => Attempt::Ok(v),
            Ok(_) => Attempt::Empty,
            Err(e) => Attempt::TransportError(e.to_string()),
        }
    }
}

/// Shorten a transport error message for the diagnostics list.
fn brief_err(e: &str) -> String {
    let first = e.lines().next().unwrap_or(e);
    let mut s: String = first.chars().take(80).collect();
    if first.chars().count() > 80 {
        s.push('…');
    }
    s
}

#[async_trait::async_trait]
impl Tool for SearchWebTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "search_web",
            "Search the web. Auto-detects backend from configured URL/key. Supports SearXNG, Brave, Google CSE, Bing, SerpAPI. Falls back to HTML scraping (Bing/Sogou) when all backends fail.",
            vec![
                ToolParameter::new("query", ParamType::String, true, "Search keywords", &["search", "keyword", "q", "term"]),
                ToolParameter::new("max_results", ParamType::Integer, false, "Max results (default 5)", &["limit", "count", "num", "num_results"]),
                ToolParameter::new("timeout", ParamType::Integer, false, "Timeout seconds (default 8)", &["timeout_seconds", "time_limit"]),
            ],
        )
    }

    async fn call(&self, args: &Value, cancel: &AtomicBool) -> Result<String> {
        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
        let max_results = args.get("max_results").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let timeout = args.get("timeout").and_then(|v| v.as_u64()).unwrap_or(self.timeout_secs);

        if query.is_empty() {
            return Ok(json!({"success": false, "error": "empty query"}).to_string());
        }

        let (success, engine, results, tried) =
            self.search_with_fallback(query, max_results, timeout, cancel).await;

        Ok(json!({
            "success": success,
            "query": query,
            "engine": engine,
            "results": results,
            "total_results": results.len(),
            "engines_tried": tried,
        })
        .to_string())
    }
}

// ──────────────────── Engine-specific search helpers ────────────────────

async fn searxng_search(base: &str, query: &str, max_results: usize, budget: f64) -> Result<Vec<Value>> {
    let url = format!("{}/search", base.trim_end_matches('/'));
    let resp = HTTP_CLIENT
        .get(&url)
        .query(&[("q", query), ("format", "json")])
        .timeout(Duration::from_secs_f64(budget))
        .send().await
        .map_err(|e| AacodeError::Network(e.to_string()))?;
    let body = resp.text().await.map_err(|e| AacodeError::Network(e.to_string()))?;
    let v: Value = serde_json::from_str(&body)?;
    Ok(extract_results(&v, max_results, "title", "url", "content"))
}

async fn brave_search(api_key: &str, query: &str, max_results: usize, budget: f64) -> Result<Vec<Value>> {
    let resp = HTTP_CLIENT
        .get("https://api.search.brave.com/res/v1/web/search")
        .header("Accept", "application/json")
        .header("Accept-Encoding", "gzip")
        .header("X-Subscription-Token", api_key)
        .query(&[("q", query), ("count", &max_results.to_string())])
        .timeout(Duration::from_secs_f64(budget))
        .send().await
        .map_err(|e| AacodeError::Network(e.to_string()))?;
    let body = resp.text().await.map_err(|e| AacodeError::Network(e.to_string()))?;
    let v: Value = serde_json::from_str(&body)?;
    let mut out = Vec::new();
    if let Some(arr) = v.get("web").and_then(|r| r.get("results")).and_then(|r| r.as_array()) {
        for item in arr.iter().take(max_results) {
            out.push(json!({
                "title": item.get("title").and_then(|x| x.as_str()).unwrap_or(""),
                "url": item.get("url").and_then(|x| x.as_str()).unwrap_or(""),
                "content": item.get("description").and_then(|x| x.as_str()).unwrap_or(""),
            }));
        }
    }
    Ok(out)
}

async fn google_cse_search(api_key: &str, cx: &str, query: &str, max_results: usize, budget: f64) -> Result<Vec<Value>> {
    let resp = HTTP_CLIENT
        .get("https://www.googleapis.com/customsearch/v1")
        .query(&[("key", api_key), ("cx", cx), ("q", query), ("num", &max_results.to_string())])
        .timeout(Duration::from_secs_f64(budget))
        .send().await
        .map_err(|e| AacodeError::Network(e.to_string()))?;
    let body = resp.text().await.map_err(|e| AacodeError::Network(e.to_string()))?;
    let v: Value = serde_json::from_str(&body)?;
    Ok(extract_results(&v, max_results, "title", "link", "snippet"))
}

async fn bing_search(_api_key: &str, query: &str, max_results: usize, budget: f64) -> Result<Vec<Value>> {
    // Bing v7 API requires Ocp-Apim-Subscription-Key.
    let resp = HTTP_CLIENT
        .get("https://api.bing.microsoft.com/v7.0/search")
        .header("Ocp-Apim-Subscription-Key", _api_key)
        .query(&[("q", query), ("count", &max_results.to_string()), ("mkt", "en-US")])
        .timeout(Duration::from_secs_f64(budget))
        .send().await
        .map_err(|e| AacodeError::Network(e.to_string()))?;
    let body = resp.text().await.map_err(|e| AacodeError::Network(e.to_string()))?;
    let v: Value = serde_json::from_str(&body)?;
    let mut out = Vec::new();
    if let Some(arr) = v.get("webPages").and_then(|r| r.get("value")).and_then(|r| r.as_array()) {
        for item in arr.iter().take(max_results) {
            out.push(json!({
                "title": item.get("name").and_then(|x| x.as_str()).unwrap_or(""),
                "url": item.get("url").and_then(|x| x.as_str()).unwrap_or(""),
                "content": item.get("snippet").and_then(|x| x.as_str()).unwrap_or(""),
            }));
        }
    }
    Ok(out)
}

async fn serpapi_search(api_key: &str, query: &str, max_results: usize, budget: f64) -> Result<Vec<Value>> {
    let resp = HTTP_CLIENT
        .get("https://serpapi.com/search")
        .query(&[("api_key", api_key), ("q", query), ("engine", "google"), ("num", &max_results.to_string())])
        .timeout(Duration::from_secs_f64(budget))
        .send().await
        .map_err(|e| AacodeError::Network(e.to_string()))?;
    let body = resp.text().await.map_err(|e| AacodeError::Network(e.to_string()))?;
    let v: Value = serde_json::from_str(&body)?;
    Ok(extract_results(&v, max_results, "title", "link", "snippet"))
}

/// Extract {title, url, content} from a JSON search response's results array.
fn extract_results(v: &Value, max_results: usize, title_k: &str, url_k: &str, snippet_k: &str) -> Vec<Value> {
    let arr = match v.get("results").or_else(|| v.get("items")).and_then(|r| r.as_array()) {
        Some(a) => a,
        None => return Vec::new(),
    };
    arr.iter()
        .take(max_results)
        .map(|item| {
            json!({
                "title": item.get(title_k).and_then(|x| x.as_str()).unwrap_or(""),
                "url": item.get(url_k).and_then(|x| x.as_str()).unwrap_or(""),
                "content": item.get(snippet_k).and_then(|x| x.as_str()).unwrap_or(""),
            })
        })
        .collect()
}

// ──────────────────────── HTML fallback scrape ──────────────────────────

async fn fallback_scrape(query: &str, max_results: usize, budget: f64) -> Option<(String, Vec<Value>)> {
    if budget <= 0.5 {
        return None;
    }
    // (name, url, result_regex, clean_regex, user_agent)
    // All raced in parallel; first NON-EMPTY (after quality filtering) wins.
    let scrapers: [(&str, &str, &str, &str, &str); 3] = [
        (
            "ddg_scrape",
            "https://html.duckduckgo.com/html/",
            r#"(?s)<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</"#,
            r#"<[^>]+>"#,
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ),
        (
            "bing_scrape",
            "https://www.bing.com/search",
            // Anchor on the <h2> title link — the first <a> inside b_algo is
            // often the cite/breadcrumb link, not the result title.
            r#"(?s)<li\s+class="b_algo"[^>]*>.*?<h2[^>]*>\s*<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>.*?<p[^>]*>(.*?)</p>"#,
            r#"<[^>]+>"#,
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ),
        (
            "sogou_scrape",
            "https://www.sogou.com/web",
            r#"(?s)<div[^>]*class="[^"]*vrwrap[^"]*"[^>]*>.*?<a[^>]*href="(https?://[^"]+)"[^>]*>\s*(.*?)\s*</a>.*?<p[^>]*>(.*?)</p>"#,
            r#"<[^>]+>"#,
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ),
    ];

    let (tx, mut rx) = tokio::sync::mpsc::channel::<(String, Vec<Value>)>(3);
    for (name, base_url, result_re, clean_re, ua) in &scrapers {
        let q = query.to_string();
        let n = name.to_string();
        let bu = base_url.to_string();
        let re = result_re.to_string();
        let cr = clean_re.to_string();
        let ua = ua.to_string();
        let tx = tx.clone();
        let budget = budget * 0.9; // all run in parallel with (nearly) full budget
        tokio::spawn(async move {
            let scraped = scrape_one(&n, &bu, &q, max_results, budget, &re, &cr, &ua).await;
            if let Some(v) = scraped {
                let _ = tx.send((n, v)).await;
            }
        });
    }
    drop(tx);

    tokio::time::timeout(Duration::from_secs_f64(budget), rx.recv()).await.ok().flatten()
}

async fn scrape_one(
    name: &str,
    base_url: &str,
    query: &str,
    max_results: usize,
    budget: f64,
    result_re: &str,
    clean_re: &str,
    ua: &str,
) -> Option<Vec<Value>> {
    let resp = HTTP_CLIENT
        .get(base_url)
        .header("User-Agent", ua)
        .header("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8")
        .query(&[("q", query)])
        .timeout(Duration::from_secs_f64(budget))
        .send().await;
    let resp = match resp {
        Ok(r) => r,
        Err(_) => return None,
    };
    let html = match resp.text().await {
        Ok(s) => s,
        Err(_) => return None,
    };
    let re = regex::Regex::new(result_re).ok()?;
    let clean = regex::Regex::new(clean_re).ok()?;
    let mut results = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for caps in re.captures_iter(&html) {
        let raw_url = caps.get(1).map(|m| m.as_str().trim().to_string()).unwrap_or_default();
        let url = normalize_result_url(&raw_url);
        let title = html_unescape(
            clean
                .replace_all(caps.get(2).map(|m| m.as_str()).unwrap_or(""), "")
                .trim(),
        );
        let snippet = html_unescape(
            clean
                .replace_all(caps.get(3).map(|m| m.as_str()).unwrap_or(""), "")
                .trim(),
        );
        if !is_quality_result(&url, &title) || seen.contains(&url) {
            continue;
        }
        seen.insert(url.clone());
        results.push(json!({
            "title": title,
            "url": url,
            "content": snippet,
            "engine": name,
        }));
        if results.len() >= max_results {
            break;
        }
    }
    if results.is_empty() {
        None
    } else {
        Some(results)
    }
}

/// DuckDuckGo html results use redirect links like
/// `//duckduckgo.com/l/?uddg=<percent-encoded-url>&rut=...` — unwrap them.
fn normalize_result_url(url: &str) -> String {
    if let Some(pos) = url.find("uddg=") {
        let rest = &url[pos + 5..];
        let enc = rest.split('&').next().unwrap_or(rest);
        return percent_decode(enc);
    }
    if let Some(stripped) = url.strip_prefix("//") {
        return format!("https://{stripped}");
    }
    url.to_string()
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(b) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(b);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

fn html_unescape(s: &str) -> String {
    s.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&#x27;", "'")
}

/// Drop junk entries: breadcrumb/cite pseudo-titles ("site.com › path"),
/// empty titles, ad/redirect links.
fn is_quality_result(url: &str, title: &str) -> bool {
    if url.is_empty() || !url.starts_with("http") {
        return false;
    }
    if title.is_empty() || title.contains('›') {
        return false;
    }
    // A "title" that is just a URL/domain (no spaces, looks like a host).
    if !title.contains(' ') && (title.contains("http") || title.contains(".com") || title.contains(".org")) {
        return false;
    }
    // Bing ad redirects.
    if url.contains("bing.com/aclick") || url.contains("duckduckgo.com/y.js") {
        return false;
    }
    true
}

// ─────────────────────────── detect engine type ─────────────────────────

fn detect_engine_type(url: &str) -> &'static str {
    let lower = url.to_lowercase();
    if lower.contains("brave.com") {
        "brave"
    } else if lower.contains("googleapis.com/customsearch") {
        "google_cse"
    } else if lower.contains("bing.microsoft.com") {
        "bing"
    } else if lower.contains("serpapi.com") {
        "serpapi"
    } else {
        "searxng"
    }
}

fn current_time() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// ─────────────────────────── fetch_url ──────────────────────────────────

pub struct FetchUrlTool {
    pub project_path: PathBuf,
    pub timeout_secs: u64,
}

#[async_trait::async_trait]
impl Tool for FetchUrlTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "fetch_url",
            "Fetch the content of a URL (cleaned text). Also available via run_shell + curl.",
            vec![
                ToolParameter::new("url", ParamType::String, true, "URL to fetch", &["link", "uri", "address"]),
                ToolParameter::new("timeout", ParamType::Integer, false, "Timeout seconds", &["time_limit", "max_time", "wait"]),
                ToolParameter::new("max_content_length", ParamType::Integer, false, "Max cleaned chars", &["max_length", "max_chars"]),
            ],
        )
    }

    async fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let url = args.get("url").and_then(|v| v.as_str()).unwrap_or("");
        if url.is_empty() {
            return Ok(json!({"success": false, "error": "empty url"}).to_string());
        }
        let timeout = args.get("timeout").and_then(|v| v.as_u64()).unwrap_or(self.timeout_secs);
        let max_len = args.get("max_content_length").and_then(|v| v.as_u64()).unwrap_or(5000) as usize;

        let resp = HTTP_CLIENT
            .get(url)
            .header("User-Agent", "aacode-rs/0.1")
            .timeout(Duration::from_secs_f64(timeout as f64))
            .send().await;
        let (status, body) = match resp {
            Ok(r) => {
                let status = r.status().as_u16();
                (status, r.text().await.unwrap_or_default())
            }
            Err(e) => {
                return Ok(json!({"success": false, "url": url, "error": e.to_string()}).to_string());
            }
        };

        let raw_len = body.len();
        let cleaned = clean_html(&body);
        let content: String = cleaned.chars().take(max_len).collect();
        let saved = save_extract(&self.project_path, &cleaned);

        Ok(json!({
            "success": true,
            "url": url,
            "status_code": status,
            "raw_length": raw_len,
            "content_length": cleaned.chars().count(),
            "content": content,
            "extract_file": saved,
        })
        .to_string())
    }
}

fn save_extract(project_path: &std::path::Path, content: &str) -> Option<String> {
    let dir = project_path.join(".aacode").join("context");
    std::fs::create_dir_all(&dir).ok()?;
    // Unique filename so concurrent tasks (multi-session) don't clobber each
    // other's extracts.
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let path = dir.join(format!("web_fetch_{stamp}_{}.txt", std::process::id()));
    std::fs::write(&path, content).ok()?;
    // Keep only the most recent 20 extracts to bound disk usage.
    if let Ok(entries) = std::fs::read_dir(&dir) {
        let mut files: Vec<_> = entries
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("web_fetch_"))
            .collect();
        if files.len() > 20 {
            files.sort_by_key(|e| e.metadata().and_then(|m| m.modified()).ok());
            for old in files.iter().take(files.len() - 20) {
                let _ = std::fs::remove_file(old.path());
            }
        }
    }
    Some(path.to_string_lossy().to_string())
}

// ─────────────────────────── HTML cleaning ──────────────────────────────

/// Strip HTML tags + script/style and collapse whitespace.
pub fn clean_html(html: &str) -> String {
    let stripped = remove_blocks(html);
    let mut out = String::with_capacity(stripped.len() / 2);
    let mut in_tag = false;
    for c in stripped.chars() {
        match c {
            '<' => in_tag = true,
            '>' => { in_tag = false; out.push(' '); }
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    let decoded = out
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"");
    decoded.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn remove_blocks(html: &str) -> String {
    let lower = html.to_lowercase();
    let chars: Vec<char> = html.chars().collect();
    let low: Vec<char> = lower.chars().collect();
    let mut out = String::with_capacity(html.len());
    let mut i = 0;
    let n = chars.len();
    while i < n {
        let rest: String = low[i..].iter().take(8).collect();
        if rest.starts_with("<script") || rest.starts_with("<style") {
            let close = if rest.starts_with("<script") { "</script>" } else { "</style>" };
            let low_rest: String = low[i..].iter().collect();
            if let Some(pos) = low_rest.find(close) {
                i += pos + close.len();
                continue;
            } else {
                break;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

// ─────────────────────────── search_code ────────────────────────────────

pub struct SearchCodeTool {
    pub cfg: SearchConfig,
    pub timeout_secs: u64,
}

#[async_trait::async_trait]
impl Tool for SearchCodeTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "search_code",
            "Search code examples. If SearXNG is configured, uses it with categories=it to find code. Falls back to GitHub repository search.",
            vec![
                ToolParameter::new("query", ParamType::String, true, "Search keywords", &["q", "keyword", "search"]),
                ToolParameter::new("max_results", ParamType::Integer, false, "Max results", &["limit", "count"]),
            ],
        )
    }
    async fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
        if query.is_empty() {
            return Ok(json!({"success": false, "error": "empty query"}).to_string());
        }
        let max = args.get("max_results").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

        // Prefer SearXNG with IT categories (fast connect timeout so a dead
        // host degrades to the GitHub fallback quickly).
        if let Some(base) = &self.cfg.searxng_url {
            let url = format!("{}/search", base.trim_end_matches('/'));
            match HTTP_CLIENT
                .get(&url)
                .query(&[("q", query), ("format", "json"), ("categories", "it")])
                .timeout(Duration::from_secs_f64(PER_ENGINE_CAP_SECS))
                .send().await
            {
                Ok(r) => {
                    let body = r.text().await.unwrap_or_default();
                    if let Ok(v) = serde_json::from_str::<Value>(&body) {
                        let results = extract_results(&v, max, "title", "url", "content");
                        if !results.is_empty() {
                            return Ok(json!({"success": true, "query": query, "results": results}).to_string());
                        }
                    }
                }
                Err(_) => {}
            }
        }

        // Fallback: GitHub repository search (no key needed).
        let url = format!(
            "https://api.github.com/search/repositories?q={}&per_page={}",
            urlencode(query),
            max
        );
        let resp = HTTP_CLIENT
            .get(&url)
            .header("User-Agent", "aacode-rs")
            .header("Accept", "application/vnd.github+json")
            .timeout(Duration::from_secs_f64(self.timeout_secs as f64))
            .send().await;
        match resp {
            Ok(r) => {
                let body = r.text().await.unwrap_or_default();
                let v: Value = serde_json::from_str(&body).unwrap_or(json!({}));
                let mut items = Vec::new();
                if let Some(arr) = v.get("items").and_then(|x| x.as_array()) {
                    for it in arr.iter().take(max) {
                        items.push(json!({
                            "name": it.get("full_name").and_then(|x| x.as_str()).unwrap_or(""),
                            "url": it.get("html_url").and_then(|x| x.as_str()).unwrap_or(""),
                            "description": it.get("description").and_then(|x| x.as_str()).unwrap_or(""),
                            "stars": it.get("stargazers_count").and_then(|x| x.as_u64()).unwrap_or(0),
                        }));
                    }
                }
                Ok(json!({"success": true, "query": query, "results": items}).to_string())
            }
            Err(e) => Ok(json!({"success": false, "error": e.to_string()}).to_string()),
        }
    }
}

fn urlencode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            b' ' => out.push_str("%20"),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_html_strips_tags_and_scripts() {
        let html = "<html><head><style>.a{color:red}</style><script>var x=1;</script></head><body><p>Hello&nbsp;<b>World</b></p></body></html>";
        let cleaned = clean_html(html);
        assert!(cleaned.contains("Hello"));
        assert!(cleaned.contains("World"));
        assert!(!cleaned.contains("color:red"));
        assert!(!cleaned.contains("var x"));
    }

    #[test]
    fn urlencode_works() {
        assert_eq!(urlencode("a b"), "a%20b");
        assert_eq!(urlencode("rust lang"), "rust%20lang");
    }

    #[test]
    fn detect_engine() {
        assert_eq!(detect_engine_type("https://api.search.brave.com"), "brave");
        assert_eq!(detect_engine_type("https://api.bing.microsoft.com"), "bing");
        assert_eq!(detect_engine_type("https://myserver.com"), "searxng");
    }

    #[tokio::test]
    async fn search_web_no_backend() {
        let t = SearchWebTool::new(SearchConfig::default(), 5);
        let cancel = AtomicBool::new(false);
        let out = t.call(&json!({"query": "rust"}), &cancel).await.unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        // Falls back to scrape or reports no results.
        let _ = v["success"].as_bool();
    }

    #[tokio::test]
    async fn fetch_url_empty() {
        let t = FetchUrlTool { project_path: std::env::temp_dir(), timeout_secs: 5 };
        let cancel = AtomicBool::new(false);
        let out = t.call(&json!({"url": ""}), &cancel).await.unwrap();
        assert_eq!(serde_json::from_str::<Value>(&out).unwrap()["success"], false);
    }

    #[test]
    fn schema_defined() {
        let s = SearchWebTool::new(SearchConfig::default(), 5).schema();
        assert_eq!(s.name, "search_web");
    }

    #[test]
    fn circuit_breaker_opens_after_failure() {
        // Use a unique key: the breaker map is process-wide.
        let t = SearchWebTool::new(SearchConfig::default(), 5);
        assert!(!t.circuit_open("test_engine_cb"));
        t.record_outcome("test_engine_cb", false);
        assert!(t.circuit_open("test_engine_cb"));
        // Success resets the breaker.
        t.record_outcome("test_engine_cb", true);
        assert!(!t.circuit_open("test_engine_cb"));
    }

    #[test]
    fn circuit_breaker_shared_across_instances() {
        // Process-wide sharing: instance B sees the circuit opened by A.
        let a = SearchWebTool::new(SearchConfig::default(), 5);
        let b = SearchWebTool::new(SearchConfig::default(), 5);
        a.record_outcome("test_engine_shared", false);
        assert!(b.circuit_open("test_engine_shared"), "breaker must be shared");
        b.record_outcome("test_engine_shared", true);
        assert!(!a.circuit_open("test_engine_shared"));
    }

    #[tokio::test]
    async fn unconfigured_engines_are_skipped_fast() {
        // With nothing configured, every API engine must be skipped without
        // network I/O; only the scrape fallback may take time.
        let t = SearchWebTool::new(SearchConfig::default(), 5);
        let start = std::time::Instant::now();
        let a = t.try_engine("brave", "q", 3, 5.0).await;
        assert!(matches!(a, Attempt::Skipped("not-configured")));
        let a = t.try_engine("google_cse", "q", 3, 5.0).await;
        assert!(matches!(a, Attempt::Skipped("not-configured")));
        let a = t.try_engine("serpapi", "q", 3, 5.0).await;
        assert!(matches!(a, Attempt::Skipped("not-configured")));
        let a = t.try_engine("searxng", "q", 3, 5.0).await;
        assert!(matches!(a, Attempt::Skipped("not-configured")));
        assert!(start.elapsed().as_millis() < 200, "skips must not hit the network");
    }

    /// Serializes tests that touch the process-wide "searxng" breaker entry.
    static SEARXNG_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[tokio::test]
    async fn refused_searxng_fails_fast_and_opens_circuit() {
        let _l = SEARXNG_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Connection refused on localhost is immediate; the second call must
        // then be skipped by the circuit breaker.
        let cfg = SearchConfig {
            searxng_url: Some("http://127.0.0.1:59999".into()),
            ..Default::default()
        };
        let t = SearchWebTool::new(cfg, 5);
        t.record_outcome("searxng", true); // reset shared breaker state
        let a = t.try_engine("searxng", "q", 3, 5.0).await;
        assert!(matches!(a, Attempt::TransportError(_)));
        t.record_outcome("searxng", false);
        let a = t.try_engine("searxng", "q", 3, 5.0).await;
        assert!(matches!(a, Attempt::Skipped("circuit-open")));
        t.record_outcome("searxng", true); // clean up for other tests
    }

    #[tokio::test]
    async fn search_reports_engines_tried() {
        let _l = SEARXNG_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let cfg = SearchConfig {
            searxng_url: Some("http://127.0.0.1:59999".into()),
            ..Default::default()
        };
        let t = SearchWebTool::new(cfg, 1);
        let cancel = AtomicBool::new(false);
        let out = t.call(&json!({"query": "rust", "timeout": 1}), &cancel).await.unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        let tried = v["engines_tried"].as_array().unwrap();
        assert!(!tried.is_empty());
        let first = tried[0].as_str().unwrap();
        assert!(first.starts_with("searxng:"), "first tried should be searxng, got {first}");
        t.record_outcome("searxng", true); // clean up shared breaker state
    }

    #[tokio::test]
    async fn cancelled_search_returns_immediately() {
        let cfg = SearchConfig {
            searxng_url: Some("http://127.0.0.1:59999".into()),
            ..Default::default()
        };
        let t = SearchWebTool::new(cfg, 8);
        let cancel = AtomicBool::new(true);
        let start = std::time::Instant::now();
        let out = t.call(&json!({"query": "rust"}), &cancel).await.unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["success"], false);
        assert!(start.elapsed().as_secs() < 2);
    }

    #[test]
    fn brief_err_truncates() {
        let long = "x".repeat(300);
        assert!(brief_err(&long).chars().count() <= 81);
        assert_eq!(brief_err("short"), "short");
    }

    #[test]
    fn ddg_redirect_url_unwrapped() {
        let u = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fblog.rust%2Dlang.org%2F2024%2F07%2F25%2FRust%2D1.80.0.html&rut=abc";
        assert_eq!(
            normalize_result_url(u),
            "https://blog.rust-lang.org/2024/07/25/Rust-1.80.0.html"
        );
        // Plain URLs pass through.
        assert_eq!(normalize_result_url("https://a.com/x"), "https://a.com/x");
        // Protocol-relative URLs get https.
        assert_eq!(normalize_result_url("//a.com/x"), "https://a.com/x");
    }

    #[test]
    fn percent_decode_works() {
        assert_eq!(percent_decode("a%20b"), "a b");
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("%2Fpath"), "/path");
    }

    #[test]
    fn quality_filter_drops_junk() {
        // Breadcrumb pseudo-titles from cite links.
        assert!(!is_quality_result("https://a.com", "rust-lang.orghttps://rust-lang.org › zh-CN"));
        // Domain-only titles.
        assert!(!is_quality_result("https://a.com", "rust-lang.org"));
        // Ads/redirects.
        assert!(!is_quality_result("https://www.bing.com/aclick?x=1", "Real Title"));
        // Good results pass.
        assert!(is_quality_result(
            "https://blog.rust-lang.org/2024/07/25/Rust-1.80.0.html",
            "Announcing Rust 1.80.0"
        ));
    }

    #[test]
    fn html_unescape_entities() {
        assert_eq!(html_unescape("a &amp; b&#39;s &lt;tag&gt;"), "a & b's <tag>");
    }
}
