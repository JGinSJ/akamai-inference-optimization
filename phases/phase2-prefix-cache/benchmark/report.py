"""
Phase 2 benchmark report.

Reads the JSON produced by load_gen.py and prints a formatted summary:
  - Overall cache hit rate
  - Per-request-type (shared-prefix vs unique) breakdown
  - P50 / P95 / P99 latency for hits and misses

No numbers are fabricated — all figures are derived from the input file.

Usage
-----
    python benchmark/report.py --input results/phase2_loadgen.json
"""

import argparse
import json
import statistics
import sys
from typing import Any


def _percentile(data: list[float], p: float) -> float:
    """Return the p-th percentile of data (0–100)."""
    if not data:
        return float("nan")
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


def _latency_table(latencies: list[float], label: str) -> None:
    if not latencies:
        print(f"  {label}: no data")
        return
    print(f"  {label} (n={len(latencies)}):")
    print(f"    mean : {statistics.mean(latencies):8.1f} ms")
    print(f"    P50  : {_percentile(latencies, 50):8.1f} ms")
    print(f"    P95  : {_percentile(latencies, 95):8.1f} ms")
    print(f"    P99  : {_percentile(latencies, 99):8.1f} ms")
    print(f"    min  : {min(latencies):8.1f} ms")
    print(f"    max  : {max(latencies):8.1f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 benchmark report")
    parser.add_argument(
        "--input",
        default="results/phase2_loadgen.json",
        help="Path to load_gen.py output JSON",
    )
    args = parser.parse_args()

    try:
        with open(args.input) as f:
            data: dict[str, Any] = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {args.input} not found. Run benchmark/load_gen.py first.",
              file=sys.stderr)
        sys.exit(1)

    cfg = data.get("config", {})
    results: list[dict] = data.get("results", [])

    if not results:
        print("ERROR: no results in input file.", file=sys.stderr)
        sys.exit(1)

    # Partition results
    errors = [r for r in results if r.get("error")]
    ok = [r for r in results if not r.get("error")]
    hits = [r for r in ok if r.get("cache_hit") is True]
    misses = [r for r in ok if r.get("cache_hit") is False]
    unknown = [r for r in ok if r.get("cache_hit") is None]

    n_total = len(results)
    n_ok = len(ok)
    n_errors = len(errors)
    hit_rate = len(hits) / n_ok if n_ok else 0.0

    print()
    print("=== Phase 2 Prefix Cache — Benchmark Report ===")
    print()
    print("Run configuration")
    print(f"  URL            : {cfg.get('url', 'N/A')}")
    print(f"  Total requests : {cfg.get('num_requests', n_total)}")
    print(f"  Prefix share   : {cfg.get('prefix_share', '?'):.0%}")
    print(f"  Concurrency    : {cfg.get('concurrency', '?')}")
    print(f"  Total elapsed  : {data.get('total_elapsed_ms', '?'):.0f} ms")
    print()

    print("Cache results")
    print(f"  Successful requests : {n_ok}")
    print(f"  Errors              : {n_errors}")
    print(f"  Cache HITs          : {len(hits)}  ({hit_rate:.1%})")
    print(f"  Cache MISSes        : {len(misses)}")
    if unknown:
        print(f"  Unknown (no flag)   : {len(unknown)}")
    print()

    print("Latency breakdown")
    _latency_table([r["latency_ms"] for r in hits], "Cache HITs")
    print()
    _latency_table([r["latency_ms"] for r in misses], "Cache MISSes")
    print()
    _latency_table([r["latency_ms"] for r in ok], "All successful")

    if hits and misses:
        hit_median = _percentile([r["latency_ms"] for r in hits], 50)
        miss_median = _percentile([r["latency_ms"] for r in misses], 50)
        if miss_median > 0:
            speedup = miss_median / hit_median
            print()
            print(f"  Cache speedup (miss P50 / hit P50) : {speedup:.1f}x")

    if n_errors:
        print()
        print(f"Errors ({n_errors}):")
        for r in errors[:5]:
            print(f"  request {r['request_index']}: {r['error']}")
        if n_errors > 5:
            print(f"  ... and {n_errors - 5} more")

    print()

    # Phase 2 success criteria check (informational only — requires live stack)
    print("Phase 2 success criteria (requires live stack to validate):")
    hit_ok = "PASS" if hit_rate >= 0.40 else "FAIL" if n_ok > 0 else "NO DATA"
    print(f"  Hit rate >= 40%  : {hit_ok}  (measured: {hit_rate:.1%})")
    print(f"  Hits faster than misses: "
          + ("CHECK LATENCY TABLE ABOVE" if hits and misses else "NO DATA"))
    print()


if __name__ == "__main__":
    main()
