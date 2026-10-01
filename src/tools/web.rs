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

/// Realistic desktop Chrome UA for plain HTTP fetches (fetch_url / scrape).
/// The webview (browser tools) uses the platform browser UA instead.
const DESKTOP_UA: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";

/// Process-wide HTTP client. Carries a persistent cookie jar and realistic
/// browser default headers so plain fetches look like a normal browser request
/// (and keep cookies across calls). `Accept-Encoding` is left to reqwest.
static HTTP_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
    use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, ACCEPT_LANGUAGE, USER_AGENT};
    let mut headers = HeaderMap::new();
    headers.insert(USER_AGENT, HeaderValue::from_static(DESKTOP_UA));
    headers.insert(
        ACCEPT,
        HeaderValue::from_static("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"),
    );
    headers.insert(
        ACCEPT_LANGUAGE,
        HeaderValue::from_static("zh-CN,zh;q=0.9,en;q=0.8"),
    );
    headers.insert("Sec-Fetch-Dest", HeaderValue::from_static("document"));
    headers.insert("Sec-Fetch-Mode", HeaderValue::from_static("navigate"));
    headers.insert("Sec-Fetch-Site", HeaderValue::from_static("none"));
    headers.insert("Upgrade-Insecure-Requests", HeaderValue::from_static("1"));
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(CONNECT_TIMEOUT_SECS))
        .default_headers(headers)
        .cookie_store(true)
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
            "searxng" => {
                searxng_search(
                    self.cfg
                        .searxng_url
                        .as_deref()
                        .unwrap_or("http://localhost:8080"),
                    query,
                    max_results,
                    budget,
                )
                .await
            }
            "brave" => {
                brave_search(
                    self.cfg.brave_api_key.as_deref().unwrap_or(""),
                    query,
                    max_results,
                    budget,
                )
                .await
            }
            "google_cse" => {
                google_cse_search(
                    self.cfg.google_cse_key.as_deref().unwrap_or(""),
                    self.cfg.google_cse_cx.as_deref().unwrap_or(""),
                    query,
                    max_results,
                    budget,
                )
                .await
            }
            "bing" => {
                bing_search(
                    self.cfg.bing_api_key.as_deref().unwrap_or(""),
                    query,
                    max_results,
                    budget,
                )
                .await
            }
            "serpapi" => {
                serpapi_search(
                    self.cfg.serpapi_key.as_deref().unwrap_or(""),
                    query,
                    max_results,
                    budget,
                )
                .await
            }
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
        let max_results = args
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(5) as usize;
        let timeout = args
            .get("timeout")
            .and_then(|v| v.as_u64())
            .unwrap_or(self.timeout_secs);

        if query.is_empty() {
            return Ok(json!({"success": false, "error": "empty query"}).to_string());
        }

        let (success, engine, results, tried) = self
            .search_with_fallback(query, max_results, timeout, cancel)
            .await;

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

async fn searxng_search(
    base: &str,
    query: &str,
    max_results: usize,
    budget: f64,
) -> Result<Vec<Value>> {
    let url = format!("{}/search", base.trim_end_matches('/'));
    let resp = HTTP_CLIENT
        .get(&url)
        .query(&[("q", query), ("format", "json")])
        .timeout(Duration::from_secs_f64(budget))
        .send()
        .await
        .map_err(|e| AacodeError::Network(e.to_string()))?;
    let body = resp
        .text()
        .await
        .map_err(|e| AacodeError::Network(e.to_string()))?;
    let v: Value = serde_json::from_str(&body)?;
    Ok(extract_results(&v, max_results, "title", "url", "content"))
}

async fn brave_search(
    api_key: &str,
    query: &str,
    max_results: usize,
    budget: f64,
) -> Result<Vec<Value>> {
    let resp = HTTP_CLIENT
        .get("https://api.search.brave.com/res/v1/web/search")
        .header("Accept", "application/json")
        .header("Accept-Encoding", "gzip")
        .header("X-Subscription-Token", api_key)
        .query(&[("q", query), ("count", &max_results.to_string())])
        .timeout(Duration::from_secs_f64(budget))
        .send()
        .await
        .map_err(|e| AacodeError::Network(e.to_string()))?;
    let body = resp
        .text()
        .await
        .map_err(|e| AacodeError::Network(e.to_string()))?;
    let v: Value = serde_json::from_str(&body)?;
    let mut out = Vec::new();
    if let Some(arr) = v
        .get("web")
        .and_then(|r| r.get("results"))
        .and_then(|r| r.as_array())
    {
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

async fn google_cse_search(
    api_key: &str,
    cx: &str,
    query: &str,
    max_results: usize,
    budget: f64,
) -> Result<Vec<Value>> {
    let resp = HTTP_CLIENT
        .get("https://www.googleapis.com/customsearch/v1")
        .query(&[
            ("key", api_key),
            ("cx", cx),
            ("q", query),
            ("num", &max_results.to_string()),
        ])
        .timeout(Duration::from_secs_f64(budget))
        .send()
        .await
        .map_err(|e| AacodeError::Network(e.to_string()))?;
    let body = resp
        .text()
        .await
        .map_err(|e| AacodeError::Network(e.to_string()))?;
    let v: Value = serde_json::from_str(&body)?;
    Ok(extract_results(&v, max_results, "title", "link", "snippet"))
}

async fn bing_search(
    _api_key: &str,
    query: &str,
    max_results: usize,
    budget: f64,
) -> Result<Vec<Value>> {
    // Bing v7 API requires Ocp-Apim-Subscription-Key.
    let resp = HTTP_CLIENT
        .get("https://api.bing.microsoft.com/v7.0/search")
        .header("Ocp-Apim-Subscription-Key", _api_key)
        .query(&[
            ("q", query),
            ("count", &max_results.to_string()),
            ("mkt", "en-US"),
        ])
        .timeout(Duration::from_secs_f64(budget))
        .send()
        .await
        .map_err(|e| AacodeError::Network(e.to_string()))?;
    let body = resp
        .text()
        .await
        .map_err(|e| AacodeError::Network(e.to_string()))?;
    let v: Value = serde_json::from_str(&body)?;
    let mut out = Vec::new();
    if let Some(arr) = v
        .get("webPages")
        .and_then(|r| r.get("value"))
        .and_then(|r| r.as_array())
    {
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

async fn serpapi_search(
    api_key: &str,
    query: &str,
    max_results: usize,
    budget: f64,
) -> Result<Vec<Value>> {
    let resp = HTTP_CLIENT
        .get("https://serpapi.com/search")
        .query(&[
            ("api_key", api_key),
            ("q", query),
            ("engine", "google"),
            ("num", &max_results.to_string()),
        ])
        .timeout(Duration::from_secs_f64(budget))
        .send()
        .await
        .map_err(|e| AacodeError::Network(e.to_string()))?;
    let body = resp
        .text()
        .await
        .map_err(|e| AacodeError::Network(e.to_string()))?;
    let v: Value = serde_json::from_str(&body)?;
    Ok(extract_results(&v, max_results, "title", "link", "snippet"))
}

/// Extract {title, url, content} from a JSON search response's results array.
fn extract_results(
    v: &Value,
    max_results: usize,
    title_k: &str,
    url_k: &str,
    snippet_k: &str,
) -> Vec<Value> {
    let arr = match v
        .get("results")
        .or_else(|| v.get("items"))
        .and_then(|r| r.as_array())
    {
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

/// Bing RSS search (format=rss). Much cleaner and more query-accurate than
/// scraping the HTML result page, and stable across the `www` / `cn` hosts.
/// Races both endpoints and returns the first non-empty set.
async fn bing_rss_scrape(query: &str, max_results: usize, budget: f64) -> Option<Vec<Value>> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<Value>>(2);
    for base in ["https://www.bing.com/search", "https://cn.bing.com/search"] {
        let q = query.to_string();
        let bu = base.to_string();
        let tx = tx.clone();
        tokio::spawn(async move {
            if let Some(v) = scrape_bing_rss(&bu, &q, max_results, budget).await {
                let _ = tx.send(v).await;
            }
        });
    }
    drop(tx);
    tokio::time::timeout(Duration::from_secs_f64(budget), rx.recv())
        .await
        .ok()
        .flatten()
}

async fn scrape_bing_rss(
    base_url: &str,
    query: &str,
    max_results: usize,
    budget: f64,
) -> Option<Vec<Value>> {
    let resp = HTTP_CLIENT
        .get(base_url)
        .header("User-Agent", DESKTOP_UA)
        .header("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8")
        .query(&[("q", query), ("format", "rss")])
        .timeout(Duration::from_secs_f64(budget))
        .send()
        .await
        .ok()?;
    let xml = resp.text().await.ok()?;
    let results = parse_bing_rss(&xml, max_results);
    if results.is_empty() {
        None
    } else {
        Some(results)
    }
}

/// Parse a Bing RSS document into result objects (pure — unit-testable).
fn parse_bing_rss(xml: &str, max_results: usize) -> Vec<Value> {
    let Ok(item_re) = regex::Regex::new(r"(?s)<item>(.*?)</item>") else {
        return vec![];
    };
    let Ok(title_re) = regex::Regex::new(r"(?s)<title>(.*?)</title>") else {
        return vec![];
    };
    let Ok(link_re) = regex::Regex::new(r"(?s)<link>(.*?)</link>") else {
        return vec![];
    };
    let Ok(desc_re) = regex::Regex::new(r"(?s)<description>(.*?)</description>") else {
        return vec![];
    };
    let Ok(clean_re) = regex::Regex::new(r#"<[^>]+>"#) else {
        return vec![];
    };
    let pick = |re: &regex::Regex, it: &str| -> String {
        re.captures(it)
            .map(|c| strip_cdata(&c[1]))
            .unwrap_or_default()
    };
    let mut results = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for cap in item_re.captures_iter(xml) {
        let it = &cap[1];
        let title = html_unescape(&pick(&title_re, it)).trim().to_string();
        let url = normalize_result_url(&pick(&link_re, it));
        let snippet = html_unescape(clean_re.replace_all(&pick(&desc_re, it), "").trim());
        if !is_quality_result(&url, &title) || seen.contains(&url) {
            continue;
        }
        seen.insert(url.clone());
        results.push(json!({
            "title": title,
            "url": url,
            "content": snippet,
            "engine": "bing_rss",
        }));
        if results.len() >= max_results {
            break;
        }
    }
    results
}

async fn fallback_scrape(
    query: &str,
    max_results: usize,
    budget: f64,
) -> Option<(String, Vec<Value>)> {
    if budget <= 0.5 {
        return None;
    }
    // 1) Prefer Bing RSS: clean XML, query-accurate, and endpoint-agnostic.
    //    Global (`www`) and China (`cn`) endpoints are raced — both are valid
    //    for a globally distributed app (Google Play / global App Store).
    if let Some(results) = bing_rss_scrape(query, max_results, (budget * 0.6).max(2.0)).await {
        return Some(("bing_rss".to_string(), results));
    }
    // 2) Parallel HTML scrape race; first NON-EMPTY (after quality filtering) wins.
    // (name, url, result_regex, clean_regex, user_agent)
    let scrapers: [(&str, &str, &str, &str, &str); 4] = [
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
            "bing_scrape_cn",
            "https://cn.bing.com/search",
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

    tokio::time::timeout(Duration::from_secs_f64(budget), rx.recv())
        .await
        .ok()
        .flatten()
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
        .send()
        .await;
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
        let raw_url = caps
            .get(1)
            .map(|m| m.as_str().trim().to_string())
            .unwrap_or_default();
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
    // Feeds HTML-escape `&` as `&amp;` inside hrefs/links.
    let url = html_unescape(url);
    if let Some(pos) = url.find("uddg=") {
        let rest = &url[pos + 5..];
        let enc = rest.split('&').next().unwrap_or(rest);
        return percent_decode(enc);
    }
    if let Some(stripped) = url.strip_prefix("//") {
        return format!("https://{stripped}");
    }
    url
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
    let mut out = String::with_capacity(s.len());
    let mut i = 0usize;
    while i < s.len() {
        if s.as_bytes()[i] == b'&' {
            if let Some(rel) = s[i..].find(';') {
                let semi = i + rel;
                // Entity names/refs are short; ignore absurd runs.
                if semi - i <= 12 {
                    if let Some(ch) = decode_entity(&s[i + 1..semi]) {
                        out.push(ch);
                        i = semi + 1;
                        continue;
                    }
                }
            }
        }
        let ch = s[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Decode a single HTML entity body (the text between `&` and `;`): numeric
/// (`#123`, `#x1F`) or a small set of common named entities.
fn decode_entity(ent: &str) -> Option<char> {
    if let Some(hex) = ent.strip_prefix("#x").or_else(|| ent.strip_prefix("#X")) {
        return u32::from_str_radix(hex, 16).ok().and_then(char::from_u32);
    }
    if let Some(dec) = ent.strip_prefix('#') {
        return dec.parse::<u32>().ok().and_then(char::from_u32);
    }
    Some(match ent {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        "nbsp" | "ensp" | "emsp" | "thinsp" | "zwnj" | "zwj" => ' ',
        "middot" => '·',
        "copy" => '©',
        "reg" => '®',
        "trade" => '™',
        "hellip" => '…',
        "mdash" => '—',
        "ndash" | "minus" => '–',
        "laquo" => '«',
        "raquo" => '»',
        "ldquo" => '“',
        "rdquo" => '”',
        "lsquo" => '‘',
        "rsquo" => '’',
        "times" => '×',
        "divide" => '÷',
        "deg" => '°',
        "plusmn" => '±',
        "sup2" => '²',
        "sup3" => '³',
        "frac12" => '½',
        "frac14" => '¼',
        "euro" => '€',
        "pound" => '£',
        "yen" => '¥',
        "cent" => '¢',
        "sect" => '§',
        "para" => '¶',
        "bull" => '•',
        "dagger" => '†',
        "prime" => '′',
        "Prime" => '″',
        "lsaquo" => '‹',
        "rsaquo" => '›',
        "larr" => '←',
        "rarr" => '→',
        "harr" => '↔',
        "ne" => '≠',
        "le" => '≤',
        "ge" => '≥',
        _ => return None,
    })
}

/// Strip an optional `<![CDATA[ ... ]]>` wrapper (Bing RSS descriptions).
fn strip_cdata(s: &str) -> String {
    let t = s.trim();
    t.strip_prefix("<![CDATA[")
        .and_then(|x| x.strip_suffix("]]>"))
        .map(|x| x.to_string())
        .unwrap_or_else(|| t.to_string())
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
    if !title.contains(' ')
        && (title.contains("http") || title.contains(".com") || title.contains(".org"))
    {
        return false;
    }
    // Search engines / ad redirects / SERP landing pages are not content.
    if looks_like_search_url(url) {
        return false;
    }
    // Titles that are obviously a search box / SERP.
    let t = title.trim();
    if t.contains("百度一下") || t.contains("搜索结果") {
        return false;
    }
    true
}

/// Whether a URL is a search-engine / ad-redirect / SERP page rather than a
/// content page. Kept deliberately high-precision to avoid dropping legitimate
/// results: only known search hosts, or a search *path* combined with a search
/// query param (a bare `?word=` on e.g. a dictionary entry is NOT filtered).
fn looks_like_search_url(url: &str) -> bool {
    let u = url.to_ascii_lowercase();
    const HOSTS: &[&str] = &[
        "bing.com/aclick",
        "bing.com/search",
        "duckduckgo.com/y.js",
        "mc.baidu.com",
        "baidu.com/s?",
        "m.baidu.com/from=",
        "so.com/s?",
        "sogou.com/web",
        "google.com/search",
        "google.com/url?",
        "search.yahoo.com",
        "yandex.com/search",
        "sm.cn/s?",
    ];
    if HOSTS.iter().any(|h| u.contains(h)) {
        return true;
    }
    // Search *path* + a search query param, e.g. `/s?q=`, `/search?query=`.
    let (path, query) = match u.split_once('?') {
        Some((p, q)) => (p, q),
        None => return false,
    };
    let search_path = ["/s", "/search", "/web", "/results", "/find", "/query"]
        .iter()
        .any(|s| path.ends_with(s));
    if !search_path {
        return false;
    }
    const PARAMS: &[&str] = &["q=", "word=", "query=", "wd=", "keyword=", "kw="];
    PARAMS
        .iter()
        .any(|p| query.starts_with(p) || query.contains(&format!("&{p}")))
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

/// Decode an HTTP body honoring its charset. reqwest is built with
/// `default-features = false` (no `charset` feature), so `resp.text()` assumes
/// UTF-8 and mangles GBK/GB2312 pages. Prefer the Content-Type charset, then a
/// `<meta charset>` / `http-equiv` sniff in the first 4 KB, else UTF-8 lossy.
fn decode_body(bytes: &[u8], content_type: &str) -> String {
    let label = charset_from_content_type(content_type).or_else(|| sniff_meta_charset(bytes));
    match label
        .as_deref()
        .and_then(|l| encoding_rs::Encoding::for_label(l.as_bytes()))
    {
        Some(enc) if enc != encoding_rs::UTF_8 => {
            let (s, _, _) = enc.decode(bytes);
            s.into_owned()
        }
        _ => String::from_utf8_lossy(bytes).into_owned(),
    }
}

fn charset_from_content_type(ct: &str) -> Option<String> {
    let lower = ct.to_ascii_lowercase();
    let i = lower.find("charset=")?;
    // ASCII lowercasing preserves byte offsets, so `i` is valid in `ct`.
    let rest = &ct[i + "charset=".len()..];
    let val: String = rest
        .trim_start_matches(['"', '\'', ' '])
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect();
    if val.is_empty() {
        None
    } else {
        Some(val.to_ascii_lowercase())
    }
}

fn sniff_meta_charset(bytes: &[u8]) -> Option<String> {
    let head = &bytes[..bytes.len().min(4096)];
    let s = String::from_utf8_lossy(head).to_ascii_lowercase();
    let i = s.find("charset=")?;
    let rest = &s[i + "charset=".len()..];
    let val: String = rest
        .trim_start_matches(['"', '\'', ' '])
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect();
    if val.is_empty() {
        None
    } else {
        Some(val)
    }
}

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
                ToolParameter::new(
                    "url",
                    ParamType::String,
                    true,
                    "URL to fetch",
                    &["link", "uri", "address"],
                ),
                ToolParameter::new(
                    "timeout",
                    ParamType::Integer,
                    false,
                    "Timeout seconds",
                    &["time_limit", "max_time", "wait"],
                ),
                ToolParameter::new(
                    "max_content_length",
                    ParamType::Integer,
                    false,
                    "Max cleaned chars",
                    &["max_length", "max_chars"],
                ),
            ],
        )
    }

    async fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let url = args.get("url").and_then(|v| v.as_str()).unwrap_or("");
        if url.is_empty() {
            return Ok(json!({"success": false, "error": "empty url"}).to_string());
        }
        let timeout = args
            .get("timeout")
            .and_then(|v| v.as_u64())
            .unwrap_or(self.timeout_secs);
        let max_len = args
            .get("max_content_length")
            .and_then(|v| v.as_u64())
            .unwrap_or(5000) as usize;

        // Use the shared client's browser UA / Accept-Language. Do NOT override
        // the UA with a bot string: many sites (Baidu, 知乎, …) answer a bot UA
        // with 403 / an anti-bot page, so a plain fetch looked "blocked".
        let resp = HTTP_CLIENT
            .get(url)
            .timeout(Duration::from_secs_f64(timeout as f64))
            .send()
            .await;
        let (status, body) = match resp {
            Ok(r) => {
                let status = r.status().as_u16();
                let ctype = r
                    .headers()
                    .get(reqwest::header::CONTENT_TYPE)
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("")
                    .to_string();
                let bytes = r.bytes().await.unwrap_or_default();
                (status, decode_body(&bytes, &ctype))
            }
            Err(e) => {
                return Ok(
                    json!({"success": false, "url": url, "error": e.to_string()}).to_string(),
                );
            }
        };

        let raw_len = body.len();
        let cleaned = clean_html(&body);
        let cleaned_len = cleaned.chars().count();
        let content: String = cleaned.chars().take(max_len).collect();
        let saved = save_extract(&self.project_path, &cleaned);

        let block = detect_block(status, &body, cleaned_len);
        let ok = (200..300).contains(&status) && block.is_none();

        #[allow(unused_mut)]
        let mut out = json!({
            "success": ok,
            "url": url,
            "status_code": status,
            "raw_length": raw_len,
            "content_length": cleaned_len,
            "content": content,
            "extract_file": saved,
        });
        if let Some(reason) = block {
            out["blocked"] = json!(true);
            out["block_reason"] = json!(reason);
            out["hint"] = json!(
                "Blocked by the site's anti-bot / JS challenge — a plain HTTP fetch cannot pass it. \
                 Use `browser_tools` + `browser_call`: `navigate` to the URL, then `get_page_text` (or `extract_text`). \
                 For sites with a mobile/AMP version or a public API, those often bypass the challenge."
            );
        } else if !ok {
            out["hint"] = json!(format!(
                "HTTP {status} — the server returned an error page; verify the URL or try a web search."
            ));
        } else if cleaned_len < 200 {
            #[cfg(feature = "browser")]
            {
                out["hint"] = json!(
                    "little text extracted — this page may need JavaScript; try fetch_rendered, or browser_tools + browser_call (navigate → get_page_text)"
                );
            }
        }
        Ok(out.to_string())
    }
}

/// Heuristically decide whether a fetch was blocked by anti-bot / a JS
/// challenge. Primary signal is the HTTP status; for 2xx responses we look for
/// a script-heavy shell with almost no readable text plus generic challenge
/// wording (English + Chinese). Deliberately small/generic rather than matching
/// any single site.
pub(crate) fn detect_block(status: u16, body: &str, cleaned_len: usize) -> Option<&'static str> {
    if status >= 400 {
        return Some("http_status");
    }
    if status == 0 {
        return Some("no_response");
    }
    if cleaned_len >= 120 {
        return None;
    }
    let lower = body.to_ascii_lowercase();
    const CHALLENGE_WORDS: &[&str] = &[
        "captcha",
        "challenge",
        "are you a robot",
        "verify you are human",
        "access denied",
        "just a moment",
        "attention required",
        "checking your browser",
        "enable javascript",
        "安全验证",
        "验证码",
        "人机验证",
        "访问验证",
        "滑动验证",
    ];
    if CHALLENGE_WORDS.iter().any(|w| lower.contains(w)) {
        return Some("anti_bot");
    }
    // A near-empty document that is mostly <script> → likely a JS challenge.
    if lower.matches("<script").count() >= 2 && cleaned_len < 40 {
        return Some("js_challenge");
    }
    None
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
            '>' => {
                in_tag = false;
                out.push(' ');
            }
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    let decoded = html_unescape(&out);
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
            let close = if rest.starts_with("<script") {
                "</script>"
            } else {
                "</style>"
            };
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

/// Structure-aware HTML → text: drops `<script>/<style>/<noscript>` and comments,
/// turns block tags into line breaks, strips the remaining tags, decodes common
/// entities, then collapses whitespace and duplicate lines.
///
/// More readable (and usually smaller) than [`clean_html`], which flattens the
/// whole document to a single line. Used to compress browser page dumps.
pub fn html_to_text(html: &str) -> String {
    let stripped = remove_tag_blocks(html);
    let mut out = String::with_capacity(stripped.len() / 2);
    let mut in_tag = false;
    let mut tag = String::new();
    for c in stripped.chars() {
        match c {
            '<' => {
                in_tag = true;
                tag.clear();
            }
            '>' => {
                in_tag = false;
                out.push(if is_block_tag(&tag) { '\n' } else { ' ' });
            }
            _ if in_tag => tag.push(c.to_ascii_lowercase()),
            _ => out.push(c),
        }
    }
    let decoded = decode_entities(&out);
    let mut clean = String::with_capacity(decoded.len());
    let mut prev: Option<String> = None;
    for raw in decoded.lines() {
        let line = raw.split_whitespace().collect::<Vec<_>>().join(" ");
        if line.is_empty() || prev.as_deref() == Some(line.as_str()) {
            continue;
        }
        if !clean.is_empty() {
            clean.push('\n');
        }
        clean.push_str(&line);
        prev = Some(line);
    }
    clean
}

/// Remove `<script>/<style>/<noscript>` blocks and `<!-- -->` comments.
fn remove_tag_blocks(html: &str) -> String {
    let lower = html.to_ascii_lowercase();
    let mut out = String::with_capacity(html.len());
    let mut i = 0usize;
    let n = html.len();
    while i < n {
        let rest = &lower[i..];
        let close = if rest.starts_with("<script") {
            Some("</script>")
        } else if rest.starts_with("<style") {
            Some("</style>")
        } else if rest.starts_with("<noscript") {
            Some("</noscript>")
        } else if rest.starts_with("<!--") {
            Some("-->")
        } else {
            None
        };
        if let Some(c) = close {
            match rest.find(c) {
                Some(p) => {
                    i += p + c.len();
                    continue;
                }
                None => break,
            }
        }
        let ch = html[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Whether a (lowercased, `/`-or-attribute-suffixed) tag name is block-level.
fn is_block_tag(tag: &str) -> bool {
    let t = tag
        .trim_start_matches('/')
        .split(|c: char| c.is_whitespace() || c == '/')
        .next()
        .unwrap_or("");
    matches!(
        t,
        "p" | "div"
            | "br"
            | "li"
            | "ul"
            | "ol"
            | "tr"
            | "td"
            | "th"
            | "table"
            | "h1"
            | "h2"
            | "h3"
            | "h4"
            | "h5"
            | "h6"
            | "section"
            | "article"
            | "header"
            | "footer"
            | "nav"
            | "aside"
            | "blockquote"
            | "pre"
            | "hr"
            | "form"
            | "figure"
            | "figcaption"
            | "main"
    )
}

fn decode_entities(s: &str) -> String {
    s.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&mdash;", "—")
        .replace("&ndash;", "–")
        .replace("&hellip;", "…")
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
        let max = args
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(5) as usize;

        // Prefer SearXNG with IT categories (fast connect timeout so a dead
        // host degrades to the GitHub fallback quickly).
        if let Some(base) = &self.cfg.searxng_url {
            let url = format!("{}/search", base.trim_end_matches('/'));
            match HTTP_CLIENT
                .get(&url)
                .query(&[("q", query), ("format", "json"), ("categories", "it")])
                .timeout(Duration::from_secs_f64(PER_ENGINE_CAP_SECS))
                .send()
                .await
            {
                Ok(r) => {
                    let body = r.text().await.unwrap_or_default();
                    if let Ok(v) = serde_json::from_str::<Value>(&body) {
                        let results = extract_results(&v, max, "title", "url", "content");
                        if !results.is_empty() {
                            return Ok(
                                json!({"success": true, "query": query, "results": results})
                                    .to_string(),
                            );
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
            .send()
            .await;
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
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
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
    fn html_to_text_keeps_block_lines_and_drops_scripts() {
        let html = "<h1>Title</h1><script>var x=1;</script><p>Hello&nbsp;world</p>\
                    <!-- c --><div>a</div><div>a</div><div>b</div>";
        let text = html_to_text(html);
        assert_eq!(text, "Title\nHello world\na\nb");
        assert!(!text.contains('<'));
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
        let t = FetchUrlTool {
            project_path: std::env::temp_dir(),
            timeout_secs: 5,
        };
        let cancel = AtomicBool::new(false);
        let out = t.call(&json!({"url": ""}), &cancel).await.unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&out).unwrap()["success"],
            false
        );
    }

    #[tokio::test]
    async fn fetch_url_hints_when_content_short() {
        let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let port = server.server_addr().to_ip().unwrap().port();
        std::thread::spawn(move || {
            if let Ok(req) = server.recv() {
                let _ = req.respond(tiny_http::Response::from_string(
                    "<html><body>hi</body></html>",
                ));
            }
        });
        let t = FetchUrlTool {
            project_path: std::env::temp_dir(),
            timeout_secs: 5,
        };
        let cancel = AtomicBool::new(false);
        let out = t
            .call(
                &json!({"url": format!("http://127.0.0.1:{port}/")}),
                &cancel,
            )
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["success"], true);
        #[cfg(feature = "browser")]
        assert!(
            v.get("hint").is_some(),
            "short content should hint fetch_rendered: {v}"
        );
        #[cfg(not(feature = "browser"))]
        assert!(v.get("hint").is_none());
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
        assert!(
            b.circuit_open("test_engine_shared"),
            "breaker must be shared"
        );
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
        assert!(
            start.elapsed().as_millis() < 200,
            "skips must not hit the network"
        );
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
        let out = t
            .call(&json!({"query": "rust", "timeout": 1}), &cancel)
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        let tried = v["engines_tried"].as_array().unwrap();
        assert!(!tried.is_empty());
        let first = tried[0].as_str().unwrap();
        assert!(
            first.starts_with("searxng:"),
            "first tried should be searxng, got {first}"
        );
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
        assert!(!is_quality_result(
            "https://a.com",
            "rust-lang.orghttps://rust-lang.org › zh-CN"
        ));
        // Domain-only titles.
        assert!(!is_quality_result("https://a.com", "rust-lang.org"));
        // Ads/redirects.
        assert!(!is_quality_result(
            "https://www.bing.com/aclick?x=1",
            "Real Title"
        ));
        // Good results pass.
        assert!(is_quality_result(
            "https://blog.rust-lang.org/2024/07/25/Rust-1.80.0.html",
            "Announcing Rust 1.80.0"
        ));
    }

    #[test]
    fn html_unescape_entities() {
        assert_eq!(
            html_unescape("a &amp; b&#39;s &lt;tag&gt;"),
            "a & b's <tag>"
        );
    }

    #[test]
    fn decode_body_honors_charset() {
        // "中文" encoded as GBK.
        let gbk = [0xD6u8, 0xD0, 0xCE, 0xC4];
        assert_eq!(decode_body(&gbk, "text/html; charset=gbk"), "中文");
        // Header charset wins.
        assert_eq!(
            decode_body("中文".as_bytes(), "text/html; charset=utf-8"),
            "中文"
        );
        // No header charset → sniff <meta charset>.
        let mut html = b"<html><head><meta charset=\"gb18030\"></head><body>".to_vec();
        html.extend_from_slice(&gbk);
        assert!(decode_body(&html, "text/html").contains("中文"));
        // Plain UTF-8 fallback.
        assert_eq!(decode_body("中文".as_bytes(), "text/html"), "中文");
        // Charset parsing helpers.
        assert_eq!(
            charset_from_content_type("text/html; Charset=\"GB2312\""),
            Some("gb2312".to_string())
        );
        assert_eq!(charset_from_content_type("text/html"), None);
    }

    #[test]
    fn charset_from_content_type_edge_cases() {
        // Extra parameters after charset are ignored.
        assert_eq!(
            charset_from_content_type("text/html; charset=gbk; x=1"),
            Some("gbk".to_string())
        );
        // No space, uppercase name/value.
        assert_eq!(
            charset_from_content_type("CHARSET=UTF-8"),
            Some("utf-8".to_string())
        );
        // Empty value yields None.
        assert_eq!(charset_from_content_type("text/html; charset="), None);
        // No charset at all.
        assert_eq!(charset_from_content_type("application/json"), None);
    }

    #[test]
    fn sniff_meta_charset_variants() {
        assert_eq!(
            sniff_meta_charset(
                b"<meta http-equiv=\"Content-Type\" content=\"text/html; charset=gb2312\">"
            ),
            Some("gb2312".to_string())
        );
        assert_eq!(sniff_meta_charset(b"<html>no meta</html>"), None);
    }

    #[test]
    fn html_unescape_extended_entities() {
        assert_eq!(
            html_unescape("a&ensp;b&#0183;c&#176;d&mdash;e"),
            "a b·c°d—e"
        );
        assert_eq!(html_unescape("&lt;x&gt;&amp;&#x27;"), "<x>&'");
        assert_eq!(html_unescape("&unknown; stays"), "&unknown; stays");
    }

    #[test]
    fn detect_block_classifies() {
        // Non-2xx is always blocked.
        assert_eq!(detect_block(403, "百度安全验证", 6), Some("http_status"));
        assert_eq!(detect_block(404, "not found", 9), Some("http_status"));
        // Healthy page.
        assert_eq!(
            detect_block(200, "<html><body>hello</body></html>", 500),
            None
        );
        // 2xx challenge page.
        assert_eq!(
            detect_block(200, "<html><body>安全验证</body></html>", 4),
            Some("anti_bot")
        );
        assert_eq!(
            detect_block(200, "<html><script>a</script><script>b</script></html>", 0),
            Some("js_challenge")
        );
    }

    #[test]
    fn parse_bing_rss_items() {
        let xml = r#"<rss><channel>
          <item><title><![CDATA[西安市10月份气温查询]]></title><link>https://t.com/x</link><description>历史每年西安10月的气温</description></item>
          <item><title>西安2025年10月份历史天气</title><link>https://ip.cn/a</link><description>天气查询 &amp; 数据</description></item>
        </channel></rss>"#;
        let v = parse_bing_rss(xml, 5);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0]["title"], "西安市10月份气温查询");
        assert_eq!(v[0]["url"], "https://t.com/x");
        assert_eq!(v[0]["engine"], "bing_rss");
        assert!(v[1]["content"]
            .as_str()
            .unwrap()
            .contains("天气查询 & 数据"));
    }

    #[test]
    fn quality_filter_rejects_search_landing() {
        // Search / SERP / ad landing pages are not content.
        assert!(!is_quality_result(
            "https://mc.baidu.com/s?word=%E7%99%BE%E5%BA%A6",
            "百度官方网站入口 - 百度"
        ));
        assert!(!is_quality_result(
            "https://www.bing.com/search?q=x",
            "x - Bing"
        ));
        // A bare `?q=` on a non-search path is NOT treated as a SERP (kept).
        assert!(is_quality_result(
            "https://example.com/?q=weather",
            "Weather"
        ));
        assert!(!is_quality_result("https://baidu.com/s?wd=x", "百度一下"));
        // Search path + param (generic SERP) is dropped.
        assert!(!is_quality_result(
            "https://shop.example.com/s?q=shoes",
            "shoes"
        ));
        // Real content pages survive (no false positives).
        assert!(is_quality_result(
            "https://baike.baidu.com/item/%E8%9C%98%E8%9B%9B",
            "蜘蛛_百度百科"
        ));
        assert!(is_quality_result(
            "https://www.tianqi24.com/xian/history10.html",
            "西安市10月份气温查询"
        ));
        // A bare `?word=` on a dictionary entry (content, not a SERP) survives.
        assert!(is_quality_result(
            "https://dict.youdao.com/result?word=apple",
            "apple - 有道词典"
        ));
        // A content page that merely has a `&q=` tracking param survives.
        assert!(is_quality_result(
            "https://example.com/article?id=1&q=foo",
            "An article"
        ));
    }

    #[test]
    fn normalize_url_unescapes_amp() {
        assert_eq!(
            normalize_result_url("https://a.com/x?a=1&amp;b=2"),
            "https://a.com/x?a=1&b=2"
        );
    }
}
