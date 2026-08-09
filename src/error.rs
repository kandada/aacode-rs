// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Unified error type for aacode-rs.

use std::fmt;

/// The central error type for the agent. Kept dependency-free (no thiserror)
/// to minimize the compiled binary footprint on mobile targets.
#[derive(Debug)]
pub enum AacodeError {
    /// Configuration problem (missing api key, bad model, etc.)
    Config(String),
    /// Network / HTTP transport failure (retryable classification lives in llm).
    Network(String),
    /// LLM API returned an error status or malformed payload.
    Api(String),
    /// Tool invocation failed (bad args, execution error).
    Tool(String),
    /// JSON (de)serialization failure.
    Json(String),
    /// Filesystem / IO failure.
    Io(String),
    /// The task was cancelled by the caller.
    Cancelled,
    /// Anything that does not fit the buckets above.
    Other(String),
}

impl fmt::Display for AacodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AacodeError::Config(m) => write!(f, "config error: {m}"),
            AacodeError::Network(m) => write!(f, "network error: {m}"),
            AacodeError::Api(m) => write!(f, "api error: {m}"),
            AacodeError::Tool(m) => write!(f, "tool error: {m}"),
            AacodeError::Json(m) => write!(f, "json error: {m}"),
            AacodeError::Io(m) => write!(f, "io error: {m}"),
            AacodeError::Cancelled => write!(f, "cancelled"),
            AacodeError::Other(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for AacodeError {}

impl From<serde_json::Error> for AacodeError {
    fn from(e: serde_json::Error) -> Self {
        AacodeError::Json(e.to_string())
    }
}

impl From<std::io::Error> for AacodeError {
    fn from(e: std::io::Error) -> Self {
        AacodeError::Io(e.to_string())
    }
}

/// Convenience result alias used across the crate.
pub type Result<T> = std::result::Result<T, AacodeError>;

impl AacodeError {
    /// Whether this error is worth retrying (transient network / 5xx / timeout).
    pub fn is_retryable(&self) -> bool {
        match self {
            AacodeError::Network(_) => true,
            AacodeError::Json(_) => true,
            AacodeError::Api(m) => {
                let l = m.to_lowercase();
                l.contains("timeout")
                    || l.contains("timed out")
                    || l.contains("connection")
                    || l.contains(" 500")
                    || l.contains(" 502")
                    || l.contains(" 503")
                    || l.contains(" 504")
                    || l.contains("rate limit")
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_formats() {
        assert_eq!(AacodeError::Cancelled.to_string(), "cancelled");
        assert!(AacodeError::Config("x".into()).to_string().contains("config"));
    }

    #[test]
    fn retryable_classification() {
        assert!(AacodeError::Network("reset".into()).is_retryable());
        assert!(AacodeError::Api("HTTP 503 unavailable".into()).is_retryable());
        assert!(AacodeError::Api("rate limit exceeded".into()).is_retryable());
        assert!(AacodeError::Json("malformed response".into()).is_retryable());
        assert!(!AacodeError::Api("HTTP 401 unauthorized".into()).is_retryable());
        assert!(!AacodeError::Config("no key".into()).is_retryable());
    }

    #[test]
    fn json_error_converts() {
        let e: AacodeError = serde_json::from_str::<i32>("not json").unwrap_err().into();
        matches!(e, AacodeError::Json(_));
    }
}
