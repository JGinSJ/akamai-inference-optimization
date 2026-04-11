"""
Phase 2 load generator — prefix-cache hit-rate benchmark.

Sends N requests to the Fermyon handler with a configurable fraction sharing
the same prompt prefix.  Records per-request latency and cache_hit flag.
Writes raw results to results/phase2_loadgen.json.

No conclusions are drawn here — use report.py to analyse results.
No benchmark numbers are fabricated — all figures come from live measurements.

Usage
-----
    cd phases/phase2-prefix-cache
    pip install -r benchmark/requirements.txt
    python benchmark/load_gen.py \\
        --url http://<fermyon-host>/v1/completions \\
        --requests 200 \\
        --prefix-share 0.5

Arguments
---------
--url            Fermyon function endpoint (required)
--requests       Total number of requests to send (default: 100)
--prefix-share   Fraction of requests that use the shared prefix (0.0–1.0,
                 default: 0.5 — matches Phase 2 success criterion)
--shared-prefix  The shared prompt prefix text (default: a fixed system prompt)
--max-tokens     max_tokens value forwarded to vLLM (default: 64)
--output         Path to write JSON results (default: results/phase2_loadgen.json)
--concurrency    Number of parallel threads (default: 1 — sequential)
"""

import argparse
import json
import os
import random
import string
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests as http_requests

# Default shared prefix — a fixed system prompt representative of real usage
DEFAULT_SHARED_PREFIX = (
    "You are a helpful assistant. Answer concisely and accurately. "
    "Always cite your sources when making factual claims. "
)

# Seed for reproducible unique-suffix generation
RANDOM_SEED = 42


def _random_suffix(length: int = 40) -> str:
    """Generate a random alphanumeric string to make unique prompts."""
    chars = string.ascii_letters + string.digits + " "
    return "".join(random.choices(chars, k=length))


def _send_one(
    url: str,
    prompt: str,
    max_tokens: int,
    request_index: int,
    shared: bool,
) -> dict[str, Any]:
    payload = {"prompt": prompt, "max_tokens": max_tokens}
    t0 = time.perf_counter()
    try:
        resp = http_requests.post(url, json=payload, timeout=60)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text}
        return {
            "request_index": request_index,
            "shared_prefix": shared,
            "prompt_len": len(prompt),
            "http_status": resp.status_code,
            "cache_hit": body.get("cache_hit"),
            "latency_ms": round(elapsed_ms, 2),
            "error": None,
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "request_index": request_index,
            "shared_prefix": shared,
            "prompt_len": len(prompt),
            "http_status": None,
            "cache_hit": None,
            "latency_ms": round(elapsed_ms, 2),
            "error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 load generator")
    parser.add_argument("--url", required=True, help="Fermyon handler URL")
    parser.add_argument("--requests", type=int, default=100, dest="num_requests")
    parser.add_argument("--prefix-share", type=float, default=0.5,
                        help="Fraction of requests with the shared prefix (0–1)")
    parser.add_argument("--shared-prefix", default=DEFAULT_SHARED_PREFIX)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", default=os.path.join("results", "phase2_loadgen.json"))
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    if not 0.0 <= args.prefix_share <= 1.0:
        parser.error("--prefix-share must be between 0.0 and 1.0")

    random.seed(RANDOM_SEED)

    # Build request list — shared-prefix requests first, then unique
    n_shared = round(args.num_requests * args.prefix_share)
    n_unique = args.num_requests - n_shared

    requests_list = []
    for i in range(n_shared):
        # Shared prefix + a short unique suffix so each request is distinct
        # at the suffix level, but shares the cache key (prefix hash matches)
        prompt = args.shared_prefix + _random_suffix(20)
        requests_list.append((prompt, True))
    for i in range(n_unique):
        # Entirely unique prefix — guaranteed cache miss
        prompt = _random_suffix(60) + " " + _random_suffix(40)
        requests_list.append((prompt, False))

    # Shuffle so shared and unique requests are interleaved
    random.shuffle(requests_list)

    print(f"\n=== Phase 2 Load Generator ===")
    print(f"Target URL    : {args.url}")
    print(f"Total requests: {args.num_requests}")
    print(f"Shared prefix : {n_shared} ({args.prefix_share:.0%})")
    print(f"Unique prefix : {n_unique}")
    print(f"Concurrency   : {args.concurrency}")
    print(f"Output        : {args.output}")
    print()

    results = []
    t_total_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(_send_one, args.url, prompt, args.max_tokens, idx, shared): idx
            for idx, (prompt, shared) in enumerate(requests_list)
        }
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            if completed % 10 == 0 or completed == args.num_requests:
                print(f"  {completed}/{args.num_requests} done", end="\r", flush=True)

    total_elapsed = (time.perf_counter() - t_total_start) * 1000
    print(f"\nAll {args.num_requests} requests completed in {total_elapsed:.0f} ms")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    payload = {
        "config": {
            "url": args.url,
            "num_requests": args.num_requests,
            "prefix_share": args.prefix_share,
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "random_seed": RANDOM_SEED,
        },
        "total_elapsed_ms": round(total_elapsed, 2),
        "results": sorted(results, key=lambda r: r["request_index"]),
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Results written to: {args.output}")
    print("Run benchmark/report.py to analyse.")


if __name__ == "__main__":
    main()
