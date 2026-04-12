"""
Generate REPORT.md from benchmark result JSON files in results/.

Reads every *.json file in the results directory, reconstructs
BenchmarkSummary objects, and writes a Markdown report with:

  - Latency table (TTFT, E2E, ITL percentiles by GPU and concurrency)
  - Throughput table (tokens/sec by GPU and batch configuration)
  - Cost comparison table (requires gpu_hourly_usd to be set in CSVs;
    if comparison.csv is absent, this section is marked PLACEHOLDER)
  - Tensor-parallel vs data-parallel recommendation section

Usage
-----
    python report/generate_report.py \
        --results-dir results/ \
        --output results/REPORT.md

All PLACEHOLDER values in the output mark figures that require either
confirmed Akamai pricing or measured benchmark data to fill in.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load raw result files
# ---------------------------------------------------------------------------

def _load_run(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Skipping %s: %s", path, exc)
        return None


def _load_all_runs(results_dir: Path) -> List[dict]:
    runs = []
    for p in sorted(results_dir.glob("*.json")):
        run = _load_run(p)
        if run:
            runs.append(run)
    return runs


# ---------------------------------------------------------------------------
# Reconstruct metrics from raw runs (no dependency on harness package)
# ---------------------------------------------------------------------------

def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return float("nan")
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * pct / 100)
    idx = min(idx, len(sorted_v) - 1)
    return sorted_v[idx]


def _summarise_run(run: dict) -> Optional[dict]:
    """
    Return a flat dict of metrics for one run, or None if no successes.
    This mirrors metrics.summarise() without importing it, so the report
    generator can run without harness dependencies.
    """
    results = run.get("results", [])
    successes = [r for r in results if not r.get("error")]
    if not successes:
        return None

    wall = run.get("wall_time_s", 1.0)
    total_tokens = sum(r.get("tokens_generated", 0) for r in successes)
    e2e_vals = [r["e2e_s"] * 1000 for r in successes]
    ttft_vals = [r["ttft_s"] * 1000 for r in successes if r.get("ttft_s") is not None]

    itl_vals = []
    for r in successes:
        ttft = r.get("ttft_s")
        toks = r.get("tokens_generated", 0)
        if ttft is not None and toks > 1:
            itl = (r["e2e_s"] - ttft) / (toks - 1) * 1000
            if itl > 0:
                itl_vals.append(itl)

    return {
        "tag": run.get("tag", "?"),
        "model": run.get("model", "?"),
        "concurrency": run.get("concurrency", "?"),
        "max_tokens": run.get("max_tokens", "?"),
        "wall_time_s": wall,
        "total_requests": len(results),
        "error_count": len(results) - len(successes),
        "tokens_per_second": total_tokens / wall if wall > 0 else 0,
        "requests_per_second": len(successes) / wall if wall > 0 else 0,
        "e2e_p50": _percentile(e2e_vals, 50),
        "e2e_p95": _percentile(e2e_vals, 95),
        "e2e_p99": _percentile(e2e_vals, 99),
        "ttft_p50": _percentile(ttft_vals, 50) if ttft_vals else None,
        "ttft_p95": _percentile(ttft_vals, 95) if ttft_vals else None,
        "ttft_p99": _percentile(ttft_vals, 99) if ttft_vals else None,
        "itl_p50": _percentile(itl_vals, 50) if itl_vals else None,
        "itl_p99": _percentile(itl_vals, 99) if itl_vals else None,
    }


def _fmt(v: Optional[float], decimals: int = 1) -> str:
    if v is None or v != v:  # None or NaN
        return "—"
    return f"{v:.{decimals}f}"


# ---------------------------------------------------------------------------
# Markdown sections
# ---------------------------------------------------------------------------

def _latency_table(summaries: List[dict]) -> str:
    lines = [
        "| Tag | Concurrency | E2E P50 (ms) | E2E P95 (ms) | E2E P99 (ms) "
        "| TTFT P50 (ms) | TTFT P95 (ms) | ITL P50 (ms) | ITL P99 (ms) |",
        "|-----|-------------|--------------|--------------|--------------|"
        "---------------|---------------|--------------|--------------|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['tag']} | {s['concurrency']} "
            f"| {_fmt(s['e2e_p50'])} | {_fmt(s['e2e_p95'])} | {_fmt(s['e2e_p99'])} "
            f"| {_fmt(s.get('ttft_p50'))} | {_fmt(s.get('ttft_p95'))} "
            f"| {_fmt(s.get('itl_p50'))} | {_fmt(s.get('itl_p99'))} |"
        )
    return "\n".join(lines)


def _throughput_table(summaries: List[dict]) -> str:
    lines = [
        "| Tag | Concurrency | max_tokens | Tokens/sec | Requests/sec | Errors |",
        "|-----|-------------|------------|------------|--------------|--------|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['tag']} | {s['concurrency']} | {s['max_tokens']} "
            f"| {_fmt(s['tokens_per_second'], 1)} | {_fmt(s['requests_per_second'], 3)} "
            f"| {s['error_count']}/{s['total_requests']} |"
        )
    return "\n".join(lines)


def _cost_section(results_dir: Path) -> str:
    csv_path = results_dir / "comparison.csv"
    if not csv_path.exists():
        return (
            "> PLACEHOLDER: Run `harness/cost_model.py` after filling in "
            "`gpu_hourly_usd` in the config files.  Results will appear here.\n"
        )

    import csv as _csv
    rows = []
    with csv_path.open() as f:
        reader = _csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return "> PLACEHOLDER: comparison.csv is empty.\n"

    lines = [
        "| Tag | GPU | tok/s | Cost/token (USD) | Cost/req (USD) | Cost/M tokens (USD) |",
        "|-----|-----|-------|-----------------|----------------|---------------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['tag']} | {r['gpu_label']} | {r['tokens_per_second']} "
            f"| {r['cost_per_token_usd']} | {r['cost_per_request_usd']} "
            f"| {r['cost_per_million_tokens_usd']} |"
        )
    return "\n".join(lines)


_TP_VS_DP_SECTION = """\
## Tensor parallelism vs data parallelism

This section explains when to use each strategy on the two target GPUs.
Numbers marked PLACEHOLDER require measured results from this benchmark.

### What each strategy does

**Tensor parallelism (TP)** — splits a single model's weight matrices
across N GPUs using column/row partitioning (Megatron-LM style).  Each GPU
holds 1/N of every layer.  A forward pass requires an all-reduce collective
after each tensor-parallel layer.  vLLM exposes this via
`--tensor-parallel-size N`.

**Data parallelism (DP)** — runs N independent full-model replicas, one per
GPU.  Each replica handles different requests.  No inter-GPU communication
during inference; routing is handled by a load balancer.

### When to use tensor parallelism

Use TP when:

1. **The model does not fit in a single GPU's VRAM.**  For Qwen2.5-VL-7B at
   float16 (~16 GB), a single RTX 4000 Ada (PLACEHOLDER GB VRAM) may have
   insufficient headroom once KV cache is included.  TP across two GPUs
   halves the per-GPU model footprint.

2. **Latency matters more than throughput.**  TP reduces time-to-first-token
   because the prefill computation is parallelised across GPUs.  The
   all-reduce overhead (~0.1–0.5 ms on NVLink) is small relative to the
   prefill savings for long prompts.

3. **You are running a single large model with long context.**  TP is most
   efficient when the all-reduce cost is amortised over large matrix
   multiplications, which happens at high sequence lengths or with large
   models (72B+).

### When to use data parallelism

Use DP when:

1. **The model fits in a single GPU's VRAM with headroom for KV cache.**
   For Qwen2.5-VL-7B on RTX PRO 6000 Blackwell (PLACEHOLDER GB VRAM), DP
   is likely the right choice — each GPU runs a full replica, throughput
   scales linearly with GPU count, and there is no all-reduce overhead.

2. **Throughput matters more than per-request latency.**  Two DP replicas
   serve twice as many concurrent requests as one TP-2 deployment at the
   same total VRAM cost, provided the model fits in one GPU.

3. **You want operational simplicity.**  DP replicas are independent pods.
   A failed replica does not affect others.  Rolling updates are simpler.
   TP requires all GPUs on the same node and fails completely if one GPU
   goes offline.

### Decision table for the two target GPUs

| Scenario | RTX 4000 Ada | RTX PRO 6000 Blackwell |
|---|---|---|
| Qwen2.5-VL-7B, float16 | PLACEHOLDER — measure VRAM headroom | PLACEHOLDER — measure VRAM headroom |
| Qwen2.5-VL-72B, float16 | TP required (model > single-GPU VRAM) | TP required |
| Latency-critical (TTFT < 100 ms) | TP-2 may help — measure | DP likely sufficient — measure |
| Throughput-critical (>N req/s) | DP if model fits | DP if model fits |
| Single-node, two GPUs | TP-2 viable via NVLink/PCIe | TP-2 viable |
| Multi-node | TP not supported by vLLM multi-node without pipeline parallel | DP across nodes |

> All entries marked PLACEHOLDER require measured benchmark results to
> replace.  Run `harness/run_benchmark.py` with both configs and fill in
> `docs/hardware.md` before drawing conclusions.

### All-reduce overhead estimate

At TP-2 on NVLink, each transformer layer adds one all-reduce over
`hidden_dim` floats.  For Qwen2.5-VL-7B (hidden_dim=3584, float16):

    per-layer all-reduce data = 2 × 3584 × 2 bytes ≈ 14 KB
    NVLink bandwidth (PLACEHOLDER GB/s) → latency PLACEHOLDER µs/layer

This is typically negligible vs compute time at batch size ≥ 4.  On PCIe
(no NVLink) the overhead is 5–20× higher and may become visible at small
batch sizes.  Confirm with the measured ITL values above.
"""


# ---------------------------------------------------------------------------
# Main report assembly
# ---------------------------------------------------------------------------

def generate(results_dir: Path, output_path: Path) -> None:
    runs = _load_all_runs(results_dir)
    if not runs:
        log.error("No result JSON files found in %s", results_dir)
        sys.exit(1)

    summaries = [s for r in runs if (s := _summarise_run(r)) is not None]
    if not summaries:
        log.error("All runs had zero successful requests.")
        sys.exit(1)

    log.info("Loaded %d runs, %d with successful requests.", len(runs), len(summaries))

    report = f"""\
# Phase 4 Benchmark Report

> Generated from {len(runs)} result file(s) in `{results_dir}/`.
> Values marked PLACEHOLDER require measured data or confirmed pricing.

---

## Latency

{_latency_table(summaries)}

---

## Throughput

{_throughput_table(summaries)}

---

## Cost

{_cost_section(results_dir)}

---

{_TP_VS_DP_SECTION}
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    log.info("Report written to %s", output_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Generate Phase 4 benchmark report")
    parser.add_argument("--results-dir", default="results", help="Directory containing result JSON files")
    parser.add_argument("--output", default="results/REPORT.md", help="Output path for REPORT.md")
    args = parser.parse_args()
    generate(Path(args.results_dir), Path(args.output))


if __name__ == "__main__":
    main()
