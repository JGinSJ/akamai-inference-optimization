"""
Main benchmark runner — CLI entry point for K8s Jobs and manual runs.

Loads a config YAML, drives load_gen.py at each (batch_size, concurrency)
combination defined in the config, calls the cost model, and writes:

  results/<tag>_<batch>_c<concurrency>.json   per-run raw results
  results/cost_by_batch.csv
  results/cost_by_concurrency.csv
  results/comparison.csv

Usage
-----
    python -m harness.run_benchmark --config configs/rtx4000ada.yaml

Config YAML schema
------------------
See configs/rtx4000ada.yaml for a fully annotated example.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import yaml

from .cost_model import _validate_hourly_price, write_csvs
from .load_gen import run as load_gen_run, save_results
from .metrics import RequestResult, summarise

log = logging.getLogger(__name__)


def _load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _result_from_dict(d: dict) -> RequestResult:
    return RequestResult(
        ttft_s=d.get("ttft_s"),
        e2e_s=d["e2e_s"],
        tokens_generated=d.get("tokens_generated", 0),
        error=d.get("error"),
        prompt_tokens=d.get("prompt_tokens"),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Phase 4 benchmark runner")
    parser.add_argument("--config", required=True, help="Path to benchmark config YAML")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory to write output files (default: results/)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_config(cfg_path)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    gpu_label = cfg["gpu_label"]
    model = cfg["model"]
    url = cfg["vllm_url"]
    gpu_hourly_usd = _validate_hourly_price(cfg.get("gpu_hourly_usd"), str(cfg_path))
    request_timeout_s = cfg.get("request_timeout_s", 120.0)
    num_requests = cfg.get("num_requests", 40)

    batch_sizes: list[int] = cfg.get("batch_sizes", [1, 4, 16])
    concurrency_levels: list[int] = cfg.get("concurrency_levels", [1, 4, 16])

    all_summaries = []

    for batch_size in batch_sizes:
        for concurrency in concurrency_levels:
            tag = f"{cfg.get('tag_prefix', 'run')}-b{batch_size}-c{concurrency}"
            out_path = results_dir / f"{tag}.json"

            log.info("--- Run: %s ---", tag)
            log.info("  batch_size=%d  concurrency=%d  requests=%d", batch_size, concurrency, num_requests)

            results, wall_time_s = asyncio.run(
                load_gen_run(
                    url=url,
                    model=model,
                    num_requests=num_requests,
                    concurrency=concurrency,
                    max_tokens=batch_size,
                    request_timeout_s=request_timeout_s,
                )
            )

            save_results(
                out_path,
                tag=tag,
                model=model,
                concurrency=concurrency,
                num_requests=num_requests,
                max_tokens=batch_size,
                wall_time_s=wall_time_s,
                results=results,
            )

            errors = sum(1 for r in results if r.error)
            successes = len(results) - errors
            log.info(
                "  Completed: %d ok / %d errors / %.2fs wall time",
                successes,
                errors,
                wall_time_s,
            )

            if successes == 0:
                log.warning("  Skipping summary for %s — no successful requests.", tag)
                continue

            summary = summarise(
                results,
                tag=tag,
                gpu_label=gpu_label,
                model=model,
                batch_size=batch_size,
                concurrency=concurrency,
                wall_time_s=wall_time_s,
            )

            log.info(
                "  tok/s=%.1f  e2e_p50=%.0fms  e2e_p99=%.0fms  ttft_p50=%s",
                summary.tokens_per_second,
                summary.e2e_ms.p50,
                summary.e2e_ms.p99,
                f"{summary.ttft_ms.p50:.0f}ms" if summary.ttft_ms else "n/a",
            )

            all_summaries.append(summary)

    if all_summaries:
        write_csvs(all_summaries, gpu_hourly_usd, results_dir, config_path=str(cfg_path))
        log.info("Cost CSVs written to %s/", results_dir)
    else:
        log.warning("No summaries produced — no CSVs written.")


if __name__ == "__main__":
    main()
