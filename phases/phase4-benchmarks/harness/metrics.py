"""
Benchmark metrics — pure functions, no GPU or network dependency.

Takes the list of per-request result dicts produced by load_gen.py and
returns a BenchmarkSummary.  All timing values are in seconds unless the
attribute name ends in _ms.

Glossary
--------
TTFT   Time-to-first-token: wall time from sending the request to receiving
       the first streamed token.  Dominated by prefill latency.
ITL    Inter-token latency: (total_time - TTFT) / (tokens_generated - 1).
       Dominated by decode latency per step.
E2E    End-to-end latency: total wall time for the full response.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class RequestResult:
    """
    Raw timing record for a single completed request.

    Fields
    ------
    ttft_s          : time-to-first-token in seconds (None if streaming
                      was not used or first-token time was not captured)
    e2e_s           : end-to-end latency in seconds
    tokens_generated: number of tokens in the generated response
    error           : if not None, the request failed with this message
    prompt_tokens   : number of tokens in the input prompt (optional)
    """

    ttft_s: Optional[float]
    e2e_s: float
    tokens_generated: int
    error: Optional[str] = None
    prompt_tokens: Optional[int] = None


@dataclass
class LatencyPercentiles:
    p50: float
    p95: float
    p99: float
    mean: float
    min: float
    max: float


@dataclass
class BenchmarkSummary:
    """
    Aggregated results from one benchmark run.

    All latency values are in milliseconds for readability.
    Throughput is in tokens per second.
    """

    # Run metadata
    tag: str
    gpu_label: str
    model: str
    batch_size: int
    concurrency: int
    total_requests: int
    error_count: int

    # Throughput
    tokens_per_second: float          # aggregate across all workers
    requests_per_second: float

    # Latency (ms)
    ttft_ms: Optional[LatencyPercentiles]   # None if TTFT not captured
    e2e_ms: LatencyPercentiles
    itl_ms: Optional[LatencyPercentiles]    # None if TTFT not captured

    # Per-request token counts
    mean_tokens_generated: float

    # Raw data retained for CSV export
    raw: List[RequestResult] = field(default_factory=list, repr=False)


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


def _itl(result: RequestResult) -> Optional[float]:
    """
    Compute inter-token latency for a single request.

    Returns None if TTFT is unavailable or only one token was generated
    (no inter-token interval to measure).
    """
    if result.ttft_s is None or result.tokens_generated <= 1:
        return None
    decode_time = result.e2e_s - result.ttft_s
    if decode_time <= 0 or result.tokens_generated < 2:
        return None
    return decode_time / (result.tokens_generated - 1)


def summarise(
    results: List[RequestResult],
    *,
    tag: str,
    gpu_label: str,
    model: str,
    batch_size: int,
    concurrency: int,
    wall_time_s: float,
) -> BenchmarkSummary:
    """
    Compute a BenchmarkSummary from a list of RequestResult objects.

    Parameters
    ----------
    results      : per-request results from load_gen.py
    tag          : run label (e.g. "baseline-ada-b4")
    gpu_label    : human-readable GPU name (e.g. "RTX 4000 Ada")
    model        : model ID used in the run
    batch_size   : vLLM max_tokens used (proxy for batch size)
    concurrency  : number of concurrent workers
    wall_time_s  : total wall-clock time for the run in seconds
    """
    successes = [r for r in results if r.error is None]
    errors = [r for r in results if r.error is not None]

    if not successes:
        raise ValueError(
            f"No successful requests in run '{tag}'. "
            f"{len(errors)} errors recorded."
        )

    total_tokens = sum(r.tokens_generated for r in successes)
    tok_per_s = total_tokens / wall_time_s if wall_time_s > 0 else 0.0
    req_per_s = len(successes) / wall_time_s if wall_time_s > 0 else 0.0

    e2e_ms_vals = [r.e2e_s * 1000 for r in successes]
    e2e_pct = _percentiles(e2e_ms_vals)

    ttft_vals = [r.ttft_s * 1000 for r in successes if r.ttft_s is not None]
    ttft_pct = _percentiles(ttft_vals) if ttft_vals else None

    itl_vals = [v * 1000 for r in successes if (v := _itl(r)) is not None]
    itl_pct = _percentiles(itl_vals) if itl_vals else None

    mean_tokens = total_tokens / len(successes)

    return BenchmarkSummary(
        tag=tag,
        gpu_label=gpu_label,
        model=model,
        batch_size=batch_size,
        concurrency=concurrency,
        total_requests=len(results),
        error_count=len(errors),
        tokens_per_second=tok_per_s,
        requests_per_second=req_per_s,
        ttft_ms=ttft_pct,
        e2e_ms=e2e_pct,
        itl_ms=itl_pct,
        mean_tokens_generated=mean_tokens,
        raw=results,
    )


def error_rate(summary: BenchmarkSummary) -> float:
    """Return the fraction of requests that failed, 0.0–1.0."""
    if summary.total_requests == 0:
        return 0.0
    return summary.error_count / summary.total_requests
