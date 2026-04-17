"""
Synchronous inference benchmark for a vLLM /v1/completions endpoint.

Uses requests + ThreadPoolExecutor — no asyncio, no aiohttp, no SSE.

Typical usage (single run):
    python benchmark/benchmark.py \\
        --url http://localhost:8000/v1/completions \\
        --model mistralai/Mistral-7B-Instruct-v0.2 \\
        --prompt-tokens 256 --max-tokens 128 \\
        --concurrency 4 --num-requests 40 \\
        --gpu-hourly-usd 0.96 --gpu-label "RTX 4000 Ada" \\
        --output-csv results/ada_sweep.csv --tag ada-c4-p256

Sweep example (shell loop):
    for c in 1 4 8 16; do
        python benchmark/benchmark.py --concurrency $c --tag ada-c${c}-p256 ...
    done

Each run appends one row to --output-csv (prior rows are never overwritten).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests

# Add the phase root to sys.path so harness.* is importable when the script is
# invoked directly from any working directory.
_PHASE_ROOT = Path(__file__).resolve().parent.parent
if str(_PHASE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PHASE_ROOT))

from harness.cost_model import CostBreakdown, compute_cost
from harness.metrics import BenchmarkSummary, LatencyPercentiles, RequestResult

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

# Prompt length is approximated by repeating a short English phrase.
# "The quick brown fox jumps over the lazy dog" tokenises to roughly 10 tokens
# with most BPE vocabularies (Mistral, Llama, Qwen).  Repeating it N times
# gives ~10*N tokens.  This is an approximation, not an exact count, but the
# method is honest: it scales with actual token boundaries rather than using
# character_count / 4 as a proxy.
_BASE_PHRASE = "The quick brown fox jumps over the lazy dog. "
_TOKENS_PER_REPEAT = 10  # empirically ~10 BPE tokens for the phrase above


def _build_prompt(target_tokens: int) -> str:
    repeats = max(1, round(target_tokens / _TOKENS_PER_REPEAT))
    return _BASE_PHRASE * repeats


# ---------------------------------------------------------------------------
# Single request
# ---------------------------------------------------------------------------

def _send_one(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_s: float,
) -> RequestResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": False,
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
        e2e_s = time.perf_counter() - t0
        resp.raise_for_status()
        body = resp.json()
        tokens_generated = body["usage"]["completion_tokens"]
        return RequestResult(ttft_s=None, e2e_s=e2e_s, tokens_generated=tokens_generated)
    except Exception as exc:  # noqa: BLE001
        e2e_s = time.perf_counter() - t0
        return RequestResult(ttft_s=None, e2e_s=e2e_s, tokens_generated=0, error=str(exc))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_benchmark(
    url: str,
    model: str,
    prompt_tokens: int,
    max_tokens: int,
    concurrency: int,
    num_requests: int,
    timeout_s: float,
) -> tuple[List[RequestResult], float]:
    """
    Send num_requests to the endpoint, concurrency at a time.

    Returns (results, wall_time_s).
    """
    prompt = _build_prompt(prompt_tokens)
    results: List[RequestResult] = []

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(_send_one, url, model, prompt, max_tokens, timeout_s)
            for _ in range(num_requests)
        ]
        for fut in as_completed(futures):
            results.append(fut.result())
    wall_time_s = time.perf_counter() - t_start

    return results, wall_time_s


# ---------------------------------------------------------------------------
# Summarise (inline — avoids importing harness.metrics.summarise which
# requires wall_time_s already available; we build BenchmarkSummary directly)
# ---------------------------------------------------------------------------

def _percentiles(values: List[float]) -> LatencyPercentiles:
    arr = np.array(values, dtype=float)
    return LatencyPercentiles(
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        mean=float(arr.mean()),
        min=float(arr.min()),
        max=float(arr.max()),
    )


def _summarise(
    results: List[RequestResult],
    wall_time_s: float,
    tag: str,
    gpu_label: str,
    model: str,
    prompt_tokens: int,
    max_tokens: int,
    concurrency: int,
) -> BenchmarkSummary:
    successes = [r for r in results if r.error is None]
    errors = [r for r in results if r.error is not None]

    if not successes:
        raise RuntimeError(
            f"All {len(results)} requests failed. First error: {errors[0].error}"
        )

    total_tokens = sum(r.tokens_generated for r in successes)
    tok_per_s = total_tokens / wall_time_s if wall_time_s > 0 else 0.0
    req_per_s = len(successes) / wall_time_s if wall_time_s > 0 else 0.0

    e2e_ms = _percentiles([r.e2e_s * 1000 for r in successes])

    return BenchmarkSummary(
        tag=tag,
        gpu_label=gpu_label,
        model=model,
        # batch_size field repurposed to carry prompt_tokens for cost model rows
        batch_size=prompt_tokens,
        concurrency=concurrency,
        total_requests=len(results),
        error_count=len(errors),
        tokens_per_second=tok_per_s,
        requests_per_second=req_per_s,
        ttft_ms=None,
        e2e_ms=e2e_ms,
        itl_ms=None,
        mean_tokens_generated=total_tokens / len(successes),
        raw=results,
    )


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

_CSV_FIELDNAMES = [
    "tag",
    "gpu_label",
    "model",
    "prompt_tokens",
    "max_tokens",
    "concurrency",
    "num_requests",
    "error_count",
    "wall_time_s",
    "tokens_per_second",
    "requests_per_second",
    "e2e_p50_ms",
    "e2e_p95_ms",
    "e2e_p99_ms",
    "mean_tokens_generated",
    "gpu_hourly_usd",
    "cost_per_token_usd",
    "cost_per_request_usd",
    "cost_per_million_tokens_usd",
]


def _append_csv(
    path: Path,
    summary: BenchmarkSummary,
    cost: CostBreakdown,
    wall_time_s: float,
    prompt_tokens: int,
    max_tokens: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    row = {
        "tag": summary.tag,
        "gpu_label": summary.gpu_label,
        "model": summary.model,
        "prompt_tokens": prompt_tokens,
        "max_tokens": max_tokens,
        "concurrency": summary.concurrency,
        "num_requests": summary.total_requests,
        "error_count": summary.error_count,
        "wall_time_s": round(wall_time_s, 3),
        "tokens_per_second": round(summary.tokens_per_second, 2),
        "requests_per_second": round(summary.requests_per_second, 4),
        "e2e_p50_ms": round(summary.e2e_ms.p50, 2),
        "e2e_p95_ms": round(summary.e2e_ms.p95, 2),
        "e2e_p99_ms": round(summary.e2e_ms.p99, 2),
        "mean_tokens_generated": round(summary.mean_tokens_generated, 2),
        "gpu_hourly_usd": cost.gpu_hourly_usd,
        "cost_per_token_usd": f"{cost.cost_per_token_usd:.8f}",
        "cost_per_request_usd": f"{cost.cost_per_request_usd:.6f}",
        "cost_per_million_tokens_usd": f"{cost.cost_per_million_tokens_usd:.4f}",
    }
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# stdout summary table
# ---------------------------------------------------------------------------

def _print_summary(summary: BenchmarkSummary, cost: CostBreakdown, wall_time_s: float) -> None:
    e = summary.e2e_ms
    lines = [
        "",
        f"  tag              : {summary.tag}",
        f"  gpu              : {summary.gpu_label}",
        f"  model            : {summary.model}",
        f"  concurrency      : {summary.concurrency}",
        f"  requests         : {summary.total_requests}  ({summary.error_count} errors)",
        f"  wall time        : {wall_time_s:.1f}s",
        f"  throughput       : {summary.tokens_per_second:.1f} tok/s  "
        f"({summary.requests_per_second:.2f} req/s)",
        f"  e2e latency      : p50={e.p50:.0f}ms  p95={e.p95:.0f}ms  p99={e.p99:.0f}ms",
        f"  mean tokens out  : {summary.mean_tokens_generated:.1f}",
        f"  cost/token       : ${cost.cost_per_token_usd:.8f}",
        f"  cost/request     : ${cost.cost_per_request_usd:.6f}",
        f"  cost/M tokens    : ${cost.cost_per_million_tokens_usd:.4f}",
        "",
    ]
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synchronous vLLM benchmark — appends one CSV row per run."
    )
    p.add_argument("--url", required=True, help="vLLM /v1/completions endpoint URL")
    p.add_argument("--model", required=True, help="Model ID (must match vLLM --model)")
    p.add_argument(
        "--prompt-tokens",
        type=int,
        default=256,
        choices=[128, 256, 512],
        help="Approximate input prompt length in tokens (128, 256, or 512)",
    )
    p.add_argument("--max-tokens", type=int, default=128, help="Max output tokens per request")
    p.add_argument("--concurrency", type=int, default=4, help="Concurrent in-flight requests")
    p.add_argument("--num-requests", type=int, default=40, help="Total requests to send")
    p.add_argument("--timeout", type=float, default=120.0, help="Per-request HTTP timeout (s)")
    p.add_argument("--gpu-label", default="RTX 4000 Ada", help="Human-readable GPU label")
    p.add_argument("--gpu-hourly-usd", type=float, required=True, help="GPU node hourly price in USD")
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/benchmark_results.csv"),
        help="CSV file to append results to",
    )
    p.add_argument(
        "--tag",
        default=None,
        help="Run label (default: auto-generated from concurrency and prompt-tokens)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.gpu_hourly_usd <= 0:
        print(f"ERROR: --gpu-hourly-usd must be > 0, got {args.gpu_hourly_usd}", file=sys.stderr)
        sys.exit(1)

    tag = args.tag or f"c{args.concurrency}-p{args.prompt_tokens}-m{args.max_tokens}"

    print(
        f"Running: {args.num_requests} requests | "
        f"concurrency={args.concurrency} | "
        f"prompt~{args.prompt_tokens}tok | "
        f"max_tokens={args.max_tokens}"
    )

    results, wall_time_s = run_benchmark(
        url=args.url,
        model=args.model,
        prompt_tokens=args.prompt_tokens,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        num_requests=args.num_requests,
        timeout_s=args.timeout,
    )

    summary = _summarise(
        results=results,
        wall_time_s=wall_time_s,
        tag=tag,
        gpu_label=args.gpu_label,
        model=args.model,
        prompt_tokens=args.prompt_tokens,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
    )

    cost = compute_cost(summary, args.gpu_hourly_usd)

    _print_summary(summary, cost, wall_time_s)
    _append_csv(args.output_csv, summary, cost, wall_time_s, args.prompt_tokens, args.max_tokens)
    print(f"Results appended to {args.output_csv}")


if __name__ == "__main__":
    main()
