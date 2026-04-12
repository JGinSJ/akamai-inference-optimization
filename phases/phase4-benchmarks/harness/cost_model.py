"""
Cost model: GPU-hour price → cost-per-token / cost-per-request.

All monetary values are in USD.  GPU hourly prices are read from the
benchmark config YAML.  The field gpu_hourly_usd must be present and
non-zero; if it is the literal string "PLACEHOLDER" the model will raise
a clear error rather than silently producing wrong numbers.

No prices are hardcoded here.  Fill in gpu_hourly_usd in the config file
when Akamai publishes pricing for the target GPU node pool.

Outputs
-------
Three CSV files are written to the output directory:

  cost_by_batch.csv        — cost metrics keyed by (tag, batch_size)
  cost_by_concurrency.csv  — cost metrics keyed by (tag, concurrency)
  comparison.csv           — one row per run, all metrics side by side

CSV columns (all three files share the same metric columns):
  tag, gpu_label, model, batch_size, concurrency,
  tokens_per_second, requests_per_second,
  gpu_hourly_usd, cost_per_token_usd, cost_per_request_usd,
  cost_per_million_tokens_usd,
  e2e_p50_ms, e2e_p95_ms, e2e_p99_ms,
  ttft_p50_ms, ttft_p95_ms, ttft_p99_ms,
  error_rate
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .metrics import BenchmarkSummary, error_rate

log = logging.getLogger(__name__)

_PLACEHOLDER = "PLACEHOLDER"


@dataclass
class CostBreakdown:
    """Cost figures derived from one BenchmarkSummary."""

    gpu_hourly_usd: float
    cost_per_token_usd: float
    cost_per_request_usd: float
    cost_per_million_tokens_usd: float


def compute_cost(summary: BenchmarkSummary, gpu_hourly_usd: float) -> CostBreakdown:
    """
    Derive cost metrics from a BenchmarkSummary and a GPU hourly price.

    Parameters
    ----------
    summary         : output of metrics.summarise()
    gpu_hourly_usd  : hourly cost of the GPU node in USD
                      (PLACEHOLDER — fill in from Akamai pricing)

    Returns
    -------
    CostBreakdown with per-token, per-request, and per-million-token costs.

    Notes
    -----
    cost_per_token = gpu_hourly_usd / 3600 / tokens_per_second
    cost_per_request = cost_per_token * mean_tokens_generated
    cost_per_million_tokens = cost_per_token * 1_000_000
    """
    if summary.tokens_per_second <= 0:
        raise ValueError(
            f"tokens_per_second must be > 0, got {summary.tokens_per_second!r} "
            f"in run '{summary.tag}'"
        )

    cost_per_second = gpu_hourly_usd / 3600.0
    cost_per_token = cost_per_second / summary.tokens_per_second
    cost_per_request = cost_per_token * summary.mean_tokens_generated
    cost_per_million = cost_per_token * 1_000_000

    return CostBreakdown(
        gpu_hourly_usd=gpu_hourly_usd,
        cost_per_token_usd=cost_per_token,
        cost_per_request_usd=cost_per_request,
        cost_per_million_tokens_usd=cost_per_million,
    )


def _validate_hourly_price(value: object, config_path: str) -> float:
    """
    Parse and validate gpu_hourly_usd from config.

    Raises ValueError if the value is the PLACEHOLDER string or non-positive.
    """
    if value == _PLACEHOLDER or value is None:
        raise ValueError(
            f"gpu_hourly_usd is '{value}' in {config_path}. "
            "Fill in the actual Akamai GPU node hourly price before running "
            "the cost model."
        )
    price = float(value)
    if price <= 0:
        raise ValueError(
            f"gpu_hourly_usd must be > 0, got {price!r} in {config_path}."
        )
    return price


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

_CSV_FIELDNAMES = [
    "tag",
    "gpu_label",
    "model",
    "batch_size",
    "concurrency",
    "tokens_per_second",
    "requests_per_second",
    "gpu_hourly_usd",
    "cost_per_token_usd",
    "cost_per_request_usd",
    "cost_per_million_tokens_usd",
    "e2e_p50_ms",
    "e2e_p95_ms",
    "e2e_p99_ms",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "ttft_p99_ms",
    "error_rate",
]


def _row(summary: BenchmarkSummary, cost: CostBreakdown) -> dict:
    ttft = summary.ttft_ms
    return {
        "tag": summary.tag,
        "gpu_label": summary.gpu_label,
        "model": summary.model,
        "batch_size": summary.batch_size,
        "concurrency": summary.concurrency,
        "tokens_per_second": round(summary.tokens_per_second, 2),
        "requests_per_second": round(summary.requests_per_second, 4),
        "gpu_hourly_usd": cost.gpu_hourly_usd,
        "cost_per_token_usd": f"{cost.cost_per_token_usd:.8f}",
        "cost_per_request_usd": f"{cost.cost_per_request_usd:.6f}",
        "cost_per_million_tokens_usd": f"{cost.cost_per_million_tokens_usd:.4f}",
        "e2e_p50_ms": round(summary.e2e_ms.p50, 2),
        "e2e_p95_ms": round(summary.e2e_ms.p95, 2),
        "e2e_p99_ms": round(summary.e2e_ms.p99, 2),
        "ttft_p50_ms": round(ttft.p50, 2) if ttft else "",
        "ttft_p95_ms": round(ttft.p95, 2) if ttft else "",
        "ttft_p99_ms": round(ttft.p99, 2) if ttft else "",
        "error_rate": f"{error_rate(summary):.4f}",
    }


def _write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d rows to %s", len(rows), path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_csvs(
    summaries: List[BenchmarkSummary],
    gpu_hourly_usd: float,
    output_dir: Path,
    *,
    config_path: str = "<config>",
) -> None:
    """
    Compute cost breakdowns for every summary and write three CSV files.

    Parameters
    ----------
    summaries      : list of BenchmarkSummary objects (one per run)
    gpu_hourly_usd : hourly GPU price — call _validate_hourly_price() first
    output_dir     : directory to write CSV files into
    config_path    : config file path, used in error messages only
    """
    rows: List[dict] = []
    for s in summaries:
        cost = compute_cost(s, gpu_hourly_usd)
        rows.append(_row(s, cost))

    if not rows:
        log.warning("No summaries provided; no CSVs written.")
        return

    # comparison.csv — all runs, one row each
    _write_csv(output_dir / "comparison.csv", rows)

    # cost_by_batch.csv — sorted by batch_size numerically
    by_batch = sorted(rows, key=lambda r: int(r["batch_size"]))
    _write_csv(output_dir / "cost_by_batch.csv", by_batch)

    # cost_by_concurrency.csv — sorted by concurrency numerically
    by_conc = sorted(rows, key=lambda r: int(r["concurrency"]))
    _write_csv(output_dir / "cost_by_concurrency.csv", by_conc)
