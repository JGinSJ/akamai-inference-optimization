"""
Phase 2 — Fermyon prefix-cache benchmark.

Measures cache value: how much latency does a Fermyon/Valkey HIT save vs a
MISS, and at what hit rate does the cache layer pay for itself?

Three passes in order
---------------------
Pass 1 — Cold cache (MISS path)
    Sends 10 requests through Fermyon with a shared ~500-token prefix and a
    per-request unique suffix.  A uuid4 run nonce embedded in the prefix
    guarantees the cache is cold for this run without touching Valkey.
    Aborts if any response returns X-Cache: HIT.

Pass 2 — Warm cache (HIT path)
    Replays the identical 10 requests through Fermyon.  Pass 1 populated
    the cache; all responses should return X-Cache: HIT.
    Aborts if any response returns X-Cache: MISS.

Pass 3 — Direct vLLM baseline
    Sends the same 10 requests directly to vLLM on port 8000, bypassing
    Fermyon and Valkey entirely.  Establishes raw inference latency with no
    cache layer overhead.  No X-Cache header check (vLLM does not set one).

Endpoint: POST /v1/chat/completions  (same path, same payload, both targets)
Cache status: detected from X-Cache response header (not the JSON body).

Prompt construction
-------------------
Shared prefix: "The quick brown fox jumps over the lazy dog. " repeated 50
times ≈ 500 tokens.  Same approximation method as Phase 4: ~10 BPE tokens per
phrase repeat.  The exact count is approximate but the method is honest.

Run nonce: a uuid4 string prepended to the shared prefix guarantees the cache
is cold for this run.  The nonce is the same for all 10 requests in a run, so
the same cache entry is created in Pass 1 and found in Pass 2.

Unique suffix: " REQUEST:{i}" appended to each content string makes each of
the 10 requests produce a distinct cache key within a pass, while the key is
identical between Pass 1 and Pass 2 (same nonce + same index).

Summary metrics
---------------
  - p50 and p95 e2e latency per pass
  - Miss overhead : Pass 1 p50 − Pass 3 p50  (Valkey round-trip cost on MISS)
  - Hit saving    : Pass 3 p50 − Pass 2 p50  (latency saved on a HIT)
  - Break-even hit rate = miss_overhead / (miss_overhead + hit_saving) × 100
    — the minimum hit rate for net-positive cache impact
  - Observed hit rate from Pass 2 (should be 100% for a correct warm cache)

Usage
-----
    cd phases/phase2-prefix-cache
    pip install -r benchmark/requirements.txt
    python benchmark/bench_cache.py \\
        --fermyon-url http://localhost:8082 \\
        --vllm-url    http://localhost:8000 \\
        --model       mistralai/Mistral-7B-Instruct-v0.2

Dependencies: requests, numpy  (see benchmark/requirements.txt)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid

import numpy as np
import requests as http_requests

# ---------------------------------------------------------------------------
# Run parameters
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_TOKENS = 64
NUM_REQUESTS = 10
REQUEST_TIMEOUT_S = 120.0

# Prompt construction — same approximation as Phase 4 benchmark:
# "The quick brown fox jumps over the lazy dog. " ≈ 10 BPE tokens per repeat.
# 50 repeats ≈ 500 tokens.  Honest approximation: scales with actual token
# boundaries rather than using character_count / 4 as a proxy.
_BASE_PHRASE = "The quick brown fox jumps over the lazy dog. "
_TOKENS_PER_REPEAT = 10
_TARGET_PREFIX_TOKENS = 500
_PHRASE_REPEATS = _TARGET_PREFIX_TOKENS // _TOKENS_PER_REPEAT  # 50


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_content(run_nonce: str, request_index: int) -> str:
    """
    Build the user message content for one request.

    Layout: [RUN:<nonce>] <shared_phrase × 50> REQUEST:<index>

    The run_nonce is identical across all 10 requests in a run, so they all
    share the same cache key prefix — cold in Pass 1, warm in Pass 2.
    The request_index differentiates the 10 cache keys within a pass.
    """
    shared = _BASE_PHRASE * _PHRASE_REPEATS
    return f"[RUN:{run_nonce}] {shared} REQUEST:{request_index}"


# ---------------------------------------------------------------------------
# Single HTTP request
# ---------------------------------------------------------------------------

def _send_one(
    base_url: str,
    model: str,
    content: str,
    max_tokens: int,
    timeout_s: float,
) -> dict:
    """
    POST /v1/chat/completions and record e2e latency plus the X-Cache header.

    Returns a dict with keys:
        latency_ms    float   wall-clock e2e latency in milliseconds
        x_cache       str     value of X-Cache header (uppercased), or "" if absent
        http_status   int     HTTP status code, or None on connection error
        error         str     error message, or None on success
    """
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    t0 = time.perf_counter()
    try:
        resp = http_requests.post(endpoint, json=payload, timeout=timeout_s)
        latency_ms = (time.perf_counter() - t0) * 1000
        x_cache = resp.headers.get("X-Cache", "").upper().strip()
        error = None if resp.ok else f"HTTP {resp.status_code}"
        return {
            "latency_ms": round(latency_ms, 2),
            "x_cache": x_cache,
            "http_status": resp.status_code,
            "error": error,
        }
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "latency_ms": round(latency_ms, 2),
            "x_cache": "",
            "http_status": None,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Pass runner
# ---------------------------------------------------------------------------

def _run_pass(
    label: str,
    base_url: str,
    model: str,
    contents: list[str],
    max_tokens: int,
    timeout_s: float,
    expected_x_cache: str | None,
) -> tuple[list[dict], bool]:
    """
    Send all requests sequentially and collect results.

    If expected_x_cache is "MISS" or "HIT", aborts on the first response
    that does not match and returns (results_so_far, False).

    Returns (results, True) on success.
    """
    results: list[dict] = []
    print(f"\n--- {label} ---")

    for i, content in enumerate(contents):
        print(f"  [{i + 1:2d}/{len(contents)}] ", end="", flush=True)
        r = _send_one(base_url, model, content, max_tokens, timeout_s)
        results.append({**r, "request_index": i})

        if r["error"]:
            print(f"ERROR  {r['error']}  (status={r['http_status']})")
            # Surface error but do not abort — counted in summary error count
        else:
            cache_tag = f"X-Cache={r['x_cache']}" if r["x_cache"] else "(no X-Cache header)"
            print(f"{r['latency_ms']:6.0f} ms  {cache_tag}")

            if expected_x_cache and r["x_cache"] != expected_x_cache:
                print(
                    f"\nFATAL: request {i + 1} returned X-Cache: {r['x_cache']!r} "
                    f"but expected {expected_x_cache!r}."
                )
                if expected_x_cache == "MISS":
                    print(
                        "  The cache was not cold for this run.\n"
                        "  Possible causes: a prior run used the same nonce (extremely unlikely),\n"
                        "  or the Valkey TTL has not expired since a previous identical request."
                    )
                else:
                    print(
                        "  The warm-cache pass failed — Pass 1 may not have populated the cache,\n"
                        "  or the Valkey TTL expired between passes (TTL is 3600s)."
                    )
                return results, False

    return results, True


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _stats(latencies: list[float]) -> dict:
    if not latencies:
        return {"n": 0, "p50": None, "p95": None, "mean": None, "min": None, "max": None}
    arr = np.array(latencies, dtype=float)
    return {
        "n": len(latencies),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "mean": round(float(arr.mean()), 2),
        "min": round(float(arr.min()), 2),
        "max": round(float(arr.max()), 2),
    }


def _ok_latencies(results: list[dict]) -> list[float]:
    return [r["latency_ms"] for r in results if r["error"] is None]


def _error_count(results: list[dict]) -> int:
    return sum(1 for r in results if r["error"] is not None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 Fermyon prefix-cache benchmark — three-pass latency measurement."
    )
    parser.add_argument(
        "--fermyon-url",
        default="http://localhost:8082",
        help="Fermyon proxy base URL (default: http://localhost:8082)",
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000",
        help="vLLM direct base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model ID passed in the chat completion request (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("results", "phase2_cache_benchmark.json"),
        help="Output JSON path (default: results/phase2_cache_benchmark.json)",
    )
    args = parser.parse_args()

    run_nonce = str(uuid.uuid4())
    approx_prefix_tokens = _PHRASE_REPEATS * _TOKENS_PER_REPEAT

    print("\n=== Phase 2 Fermyon Cache Benchmark ===")
    print(f"Run nonce     : {run_nonce}")
    print(f"Fermyon URL   : {args.fermyon_url}")
    print(f"vLLM URL      : {args.vllm_url}")
    print(f"Model         : {args.model}")
    print(f"Requests/pass : {NUM_REQUESTS}")
    print(f"max_tokens    : {MAX_TOKENS}")
    print(f"Shared prefix : ~{approx_prefix_tokens} tokens ({_PHRASE_REPEATS} phrase repeats)")
    print(f"Output        : {args.output}")

    # Build the 10 message contents — identical across all three passes
    contents = [_build_content(run_nonce, i) for i in range(NUM_REQUESTS)]

    # ------------------------------------------------------------------
    # Pass 1 — Cold cache (expect all MISSes)
    # ------------------------------------------------------------------
    pass1, ok = _run_pass(
        label=f"Pass 1 — Cold cache  (expect X-Cache: MISS × {NUM_REQUESTS})",
        base_url=args.fermyon_url,
        model=args.model,
        contents=contents,
        max_tokens=MAX_TOKENS,
        timeout_s=REQUEST_TIMEOUT_S,
        expected_x_cache="MISS",
    )
    if not ok:
        sys.exit(1)

    # ------------------------------------------------------------------
    # Pass 2 — Warm cache (expect all HITs)
    # ------------------------------------------------------------------
    pass2, ok = _run_pass(
        label=f"Pass 2 — Warm cache  (expect X-Cache: HIT × {NUM_REQUESTS})",
        base_url=args.fermyon_url,
        model=args.model,
        contents=contents,
        max_tokens=MAX_TOKENS,
        timeout_s=REQUEST_TIMEOUT_S,
        expected_x_cache="HIT",
    )
    if not ok:
        sys.exit(1)

    # ------------------------------------------------------------------
    # Pass 3 — Direct vLLM baseline (no X-Cache check)
    # ------------------------------------------------------------------
    pass3, ok = _run_pass(
        label=f"Pass 3 — Direct vLLM baseline  (port 8000, no cache layer)",
        base_url=args.vllm_url,
        model=args.model,
        contents=contents,
        max_tokens=MAX_TOKENS,
        timeout_s=REQUEST_TIMEOUT_S,
        expected_x_cache=None,
    )
    if not ok:
        sys.exit(1)

    # ------------------------------------------------------------------
    # Compute statistics
    # ------------------------------------------------------------------
    s1 = _stats(_ok_latencies(pass1))
    s2 = _stats(_ok_latencies(pass2))
    s3 = _stats(_ok_latencies(pass3))

    miss_overhead_ms: float | None = None
    hit_saving_ms: float | None = None
    break_even_pct: float | None = None

    if s1["p50"] is not None and s3["p50"] is not None:
        miss_overhead_ms = round(s1["p50"] - s3["p50"], 2)
    if s2["p50"] is not None and s3["p50"] is not None:
        hit_saving_ms = round(s3["p50"] - s2["p50"], 2)
    if miss_overhead_ms is not None and hit_saving_ms is not None:
        denom = miss_overhead_ms + hit_saving_ms
        if denom > 0:
            break_even_pct = round(miss_overhead_ms / denom * 100, 1)

    # Observed hit rate from Pass 2 actual X-Cache headers
    pass2_ok = [r for r in pass2 if r["error"] is None]
    pass2_hits = sum(1 for r in pass2_ok if r["x_cache"] == "HIT")
    observed_hit_rate_pct = (
        round(pass2_hits / len(pass2_ok) * 100, 1) if pass2_ok else None
    )

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    w = 34
    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)
    print(f"\n  {'Pass':<{w}} {'p50 (ms)':>9}  {'p95 (ms)':>9}  {'errors':>6}")
    print(f"  {'-'*w} {'-'*9}  {'-'*9}  {'-'*6}")

    def _fmt(v):
        return f"{v:.0f}" if v is not None else "N/A"

    print(f"  {'Pass 1 — MISS (Fermyon)':<{w}} {_fmt(s1['p50']):>9}  {_fmt(s1['p95']):>9}  {_error_count(pass1):>6}")
    print(f"  {'Pass 2 — HIT  (Fermyon)':<{w}} {_fmt(s2['p50']):>9}  {_fmt(s2['p95']):>9}  {_error_count(pass2):>6}")
    print(f"  {'Pass 3 — Direct vLLM':<{w}} {_fmt(s3['p50']):>9}  {_fmt(s3['p95']):>9}  {_error_count(pass3):>6}")

    print()
    if miss_overhead_ms is not None:
        sign = "+" if miss_overhead_ms >= 0 else ""
        print(f"  Miss overhead  (Pass1 p50 − Pass3 p50) : {sign}{miss_overhead_ms:.0f} ms")
        print(f"    → Valkey round-trip cost added on every MISS")
    if hit_saving_ms is not None:
        print(f"  Hit saving     (Pass3 p50 − Pass2 p50) : {hit_saving_ms:.0f} ms")
        print(f"    → Latency saved vs direct vLLM on every HIT")
    if break_even_pct is not None:
        print(f"  Break-even hit rate                     : {break_even_pct:.1f}%")
        print(f"    → Formula: miss_overhead / (miss_overhead + hit_saving)")
        print(f"    → Cache layer has net-positive impact above this hit rate")
    if observed_hit_rate_pct is not None:
        print(f"  Observed hit rate (Pass 2)              : {observed_hit_rate_pct:.1f}%")
        print(f"    → {pass2_hits}/{len(pass2_ok)} warm-cache responses returned X-Cache: HIT")

    # ------------------------------------------------------------------
    # Write JSON
    # ------------------------------------------------------------------
    summary = {
        "pass1_cold_miss": s1,
        "pass2_warm_hit": s2,
        "pass3_direct_vllm": s3,
        "miss_overhead_ms": miss_overhead_ms,
        "hit_saving_ms": hit_saving_ms,
        "break_even_hit_rate_pct": break_even_pct,
        "observed_hit_rate_pass2_pct": observed_hit_rate_pct,
        "error_counts": {
            "pass1": _error_count(pass1),
            "pass2": _error_count(pass2),
            "pass3": _error_count(pass3),
        },
    }
    output_doc = {
        "config": {
            "run_nonce": run_nonce,
            "fermyon_url": args.fermyon_url,
            "vllm_url": args.vllm_url,
            "model": args.model,
            "num_requests_per_pass": NUM_REQUESTS,
            "max_tokens": MAX_TOKENS,
            "shared_prefix_approx_tokens": approx_prefix_tokens,
            "phrase_repeats": _PHRASE_REPEATS,
            "tokens_per_repeat": _TOKENS_PER_REPEAT,
        },
        "summary": summary,
        "raw": {
            "pass1_cold": pass1,
            "pass2_warm": pass2,
            "pass3_direct": pass3,
        },
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_doc, f, indent=2)
    print(f"\nResults written to: {args.output}")


if __name__ == "__main__":
    main()
