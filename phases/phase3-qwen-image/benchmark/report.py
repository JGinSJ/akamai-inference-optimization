"""
Phase 3 benchmark report.

Reads one or two load_gen.py output JSON files and prints:
  - Per-batch-size throughput (req/s) and latency (P50/P95/P99)
  - Side-by-side baseline vs optimized comparison (if both files provided)

All figures come from measured data.  No numbers are fabricated.

IMPORTANT: Latency figures include network round-trip to the server.
They are not pure GPU inference times.  For isolated GPU timing, use
server-side latency_ms from the response body (report.py prints both).

Usage
-----
    # Single run
    python benchmark/report.py --input results/phase3_loadgen.json

    # Baseline vs optimized comparison
    python benchmark/report.py \\
        --baseline results/phase3_baseline.json \\
        --optimized results/phase3_optimized.json
"""

import argparse
import json
import statistics
import sys
from typing import Any, Optional


def _pct(data: list[float], p: float) -> float:
    if not data:
        return float("nan")
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _analyse_batch(results: list[dict], batch_size: int) -> Optional[dict]:
    subset = [r for r in results if r.get("batch_size") == batch_size and not r.get("error")]
    if not subset:
        return None
    latencies = [r["latency_ms"] for r in subset]
    server_latencies = [r["server_latency_ms"] for r in subset if r.get("server_latency_ms")]
    n = len(subset)
    # Throughput: requests / total wall time.
    # Approximated as n / sum(latency_s) — accurate for sequential requests.
    total_s = sum(latencies) / 1000
    rps = n / total_s if total_s > 0 else float("nan")
    return {
        "n": n,
        "batch_size": batch_size,
        "rps": rps,
        "client_latency": {
            "mean": statistics.mean(latencies),
            "p50": _pct(latencies, 50),
            "p95": _pct(latencies, 95),
            "p99": _pct(latencies, 99),
            "min": min(latencies),
            "max": max(latencies),
        },
        "server_latency": {
            "mean": statistics.mean(server_latencies),
            "p50": _pct(server_latencies, 50),
            "p95": _pct(server_latencies, 95),
            "p99": _pct(server_latencies, 99),
        } if server_latencies else None,
        "errors": len([r for r in results if r.get("batch_size") == batch_size and r.get("error")]),
        "gpu": next((r["gpu"] for r in subset if r.get("gpu")), "unknown"),
        "optimized": next((r["optimized"] for r in subset if r.get("optimized") is not None), None),
    }


def _print_run(data: dict[str, Any], label: str) -> dict:
    cfg = data.get("config", {})
    results = data.get("results", [])
    batch_sizes = cfg.get("batch_sizes", sorted({r["batch_size"] for r in results}))

    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    print(f"  URL    : {cfg.get('url', 'N/A')}")
    print(f"  Tag    : {cfg.get('tag') or '(none)'}")
    print(f"  Prompt : {cfg.get('prompt', '')[:60]}")
    print()

    analyses = {}
    for bs in batch_sizes:
        a = _analyse_batch(results, bs)
        if a is None:
            print(f"  Batch size {bs}: no data")
            continue
        analyses[bs] = a
        print(f"  Batch size {bs}  (n={a['n']}, errors={a['errors']})")
        print(f"    Throughput (client)   : {a['rps']:.2f} req/s")
        cl = a["client_latency"]
        print(f"    Client latency        : "
              f"mean={cl['mean']:.0f}ms  P50={cl['p50']:.0f}ms  "
              f"P95={cl['p95']:.0f}ms  P99={cl['p99']:.0f}ms")
        if a["server_latency"]:
            sl = a["server_latency"]
            print(f"    Server latency        : "
                  f"mean={sl['mean']:.0f}ms  P50={sl['p50']:.0f}ms  "
                  f"P95={sl['p95']:.0f}ms  P99={sl['p99']:.0f}ms")
            print(f"    (server latency excludes network round-trip)")
        print(f"    GPU                   : {a['gpu']}")
        if a["optimized"] is not None:
            flag = "EXPERIMENTAL optimized" if a["optimized"] else "baseline"
            print(f"    Model path            : {flag}")
        print()
    return analyses


def _print_comparison(baseline: dict, optimized: dict) -> None:
    batch_sizes = sorted(set(baseline) & set(optimized))
    if not batch_sizes:
        print("  No overlapping batch sizes for comparison.")
        return

    print(f"\n{'─' * 60}")
    print("  Baseline vs EXPERIMENTAL optimized")
    print(f"{'─' * 60}")
    print("  NOTE: These are measured results from a specific hardware run.")
    print("  Differences may not generalise to other hardware or workloads.")
    print()

    for bs in batch_sizes:
        b = baseline[bs]
        o = optimized[bs]
        rps_delta = ((o["rps"] - b["rps"]) / b["rps"] * 100) if b["rps"] else float("nan")
        p50_delta = o["client_latency"]["p50"] - b["client_latency"]["p50"]
        print(f"  Batch size {bs}:")
        print(f"    Throughput  baseline={b['rps']:.2f} req/s  "
              f"optimized={o['rps']:.2f} req/s  "
              f"delta={rps_delta:+.1f}%")
        print(f"    P50 latency baseline={b['client_latency']['p50']:.0f}ms  "
              f"optimized={o['client_latency']['p50']:.0f}ms  "
              f"delta={p50_delta:+.0f}ms")
        print()


def _load(path: str) -> dict[str, Any]:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {path} not found.  Run benchmark/load_gen.py first.", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 benchmark report")
    parser.add_argument("--input", help="Single run JSON (load_gen.py output)")
    parser.add_argument("--baseline", help="Baseline run JSON")
    parser.add_argument("--optimized", help="EXPERIMENTAL optimized run JSON")
    args = parser.parse_args()

    if not args.input and not (args.baseline or args.optimized):
        parser.error("Provide --input or --baseline / --optimized")

    print("\n=== Phase 3 Qwen-Image — Benchmark Report ===")

    if args.input:
        _print_run(_load(args.input), "Single run")

    elif args.baseline or args.optimized:
        baseline_analyses = {}
        optimized_analyses = {}

        if args.baseline:
            baseline_analyses = _print_run(_load(args.baseline), "Baseline")
        if args.optimized:
            optimized_analyses = _print_run(
                _load(args.optimized), "EXPERIMENTAL optimized"
            )
        if args.baseline and args.optimized:
            _print_comparison(baseline_analyses, optimized_analyses)


if __name__ == "__main__":
    main()
