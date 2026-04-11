//! Fermyon Wasm prefix-cache handler
//!
//! # Request flow
//!
//! ```text
//! POST /v1/completions  {"prompt": "..."}
//!        |
//!        v
//!   parse + validate
//!        |
//!        v
//!   SHA-256(prompt[:prefix_chars])  ← cache key
//!        |
//!     GET key
//!    /         \
//!  HIT         MISS
//!   |             |
//!   |         POST → vLLM /v1/completions
//!   |             |
//!   |         SET key ← best-effort; never fails the request
//!   |             |
//!   └─────────────┘
//!        |
//!        v
//!   {"response":…, "cache_hit":…, "latency_ms":…}
//! ```
//!
//! # Caching strategy
//!
//! This handler implements **exact-match prefix caching**: two prompts share
//! a cache entry if and only if their first `prefix_chars` *characters* (not
//! bytes) are byte-for-byte identical.
//!
//! TODO(semantic-cache): A future implementation could replace the SHA-256
//! hash step with an embedding similarity lookup against a vector store
//! (e.g. pgvector, Qdrant). That would allow semantically equivalent prompts
//! (differently worded system prompts, synonym substitutions) to share a
//! cache entry. This is explicitly out of scope for Phase 2.

use anyhow::{anyhow, Context};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use spin_sdk::{
    http::{IntoResponse, Method, Request, Response},
    http_component,
    redis::Connection as RedisConn,
    variables,
};
use std::time::{SystemTime, UNIX_EPOCH};

/// Default number of prompt *characters* to include in the cache key prefix.
const DEFAULT_PREFIX_CHARS: usize = 128;

// ---------------------------------------------------------------------------
// Request / response shapes
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct InboundPayload {
    prompt: String,
}

#[derive(Serialize)]
struct HandlerResponse {
    /// The model response text (or raw vLLM JSON on a cache miss).
    response: String,
    /// True if the response was served from the Valkey cache.
    cache_hit: bool,
    /// Wall-clock milliseconds from request receipt to response sent.
    latency_ms: u64,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Build the Valkey cache key for a prompt.
///
/// # Algorithm
///
/// 1. Take the first `prefix_chars` Unicode *scalar values* (characters) of
///    `prompt`.  Using characters, not bytes, ensures that a multibyte UTF-8
///    sequence is never split mid-character.
/// 2. SHA-256 hash the UTF-8 encoding of that prefix.
/// 3. Return the lowercase hexadecimal digest (always 64 ASCII characters).
///
/// # Exact-match semantics
///
/// Two prompts produce the same key only when their first `prefix_chars`
/// characters are **identical**.  There is no tolerance for rephrasing,
/// reordering, or synonym substitution.
///
/// TODO(semantic-cache): To handle semantically equivalent prompts, embed
/// the prefix with a sentence-transformer model and query a vector store for
/// approximate nearest neighbours.  The cache key would then be the nearest
/// neighbour's stored key rather than the hash of the input itself.
pub fn make_cache_key(prompt: &str, prefix_chars: usize) -> String {
    let prefix: String = prompt.chars().take(prefix_chars).collect();
    let digest = Sha256::digest(prefix.as_bytes());
    format!("{digest:x}")
}

fn json_response(status: u16, body: String) -> Response {
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(body)
        .build()
}

fn error_response(status: u16, message: &str) -> Response {
    Response::builder()
        .status(status)
        .body(message.to_string())
        .build()
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

#[http_component]
async fn handle(req: Request) -> anyhow::Result<impl IntoResponse> {
    let t_start = now_ms();

    // --- Only POST is accepted ---
    if *req.method() != Method::Post {
        return Ok(error_response(405, "Method Not Allowed — use POST"));
    }

    // --- Read Spin variables ---
    let valkey_addr = variables::get("valkey_address")
        .context("Spin variable 'valkey_address' is required but not set")?;
    let vllm_url = variables::get("vllm_url")
        .context("Spin variable 'vllm_url' is required but not set")?;
    let prefix_chars: usize = variables::get("prefix_chars")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_PREFIX_CHARS);

    // --- Parse and validate request body ---
    let body_bytes = req.body().to_vec();
    let payload: InboundPayload = serde_json::from_slice(&body_bytes)
        .map_err(|e| anyhow!("invalid JSON body: {e}"))?;

    if payload.prompt.is_empty() {
        return Ok(error_response(400, "'prompt' must not be empty"));
    }

    // --- Valkey cache lookup ---
    let cache_key = make_cache_key(&payload.prompt, prefix_chars);
    let conn = RedisConn::open(&valkey_addr)?;

    if let Some(cached_bytes) = conn.get(&cache_key)? {
        let cached_text = String::from_utf8(cached_bytes)
            .unwrap_or_else(|_| String::from("<non-utf8 cached value>"));

        let out = HandlerResponse {
            response: cached_text,
            cache_hit: true,
            latency_ms: now_ms().saturating_sub(t_start),
        };
        return Ok(json_response(200, serde_json::to_string(&out)?));
    }

    // --- Cache MISS: forward to vLLM ---
    let vllm_req = Request::builder()
        .method(Method::Post)
        .uri(format!("{vllm_url}/v1/completions"))
        .header("content-type", "application/json")
        .body(body_bytes)
        .build();

    let vllm_resp: Response = spin_sdk::http::send(vllm_req).await?;
    let vllm_status = vllm_resp.status();

    if vllm_status != 200 {
        return Ok(error_response(
            vllm_status,
            &format!("upstream vLLM returned HTTP {vllm_status}"),
        ));
    }

    let vllm_body = vllm_resp.body().to_vec();

    // Store in Valkey.  This is best-effort: a write failure must not cause
    // the client request to fail.  The next identical-prefix request will
    // simply be another miss.
    let _ = conn.set(&cache_key, &vllm_body);

    let response_text = String::from_utf8(vllm_body)
        .unwrap_or_else(|_| String::from("<non-utf8 vllm response>"));

    let out = HandlerResponse {
        response: response_text,
        cache_hit: false,
        latency_ms: now_ms().saturating_sub(t_start),
    };
    Ok(json_response(200, serde_json::to_string(&out)?))
}
