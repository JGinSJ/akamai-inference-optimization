"""
Phase 3 optimized-path benchmark.

Sends 10 sequential requests to /v1/generate using the same 320×240
synthetic JPEG that was used for the Phase 3 baseline measurement
(phase3_baseline_first_result.json: solid-color JPEG, seed=0).

Run 1 is discarded to absorb torch.compile warm-up latency.
Mean and standard deviation are reported over runs 2–10.

Results are written to results/phase3_optimized_bench.json.

Usage
-----
    cd phases/phase3-qwen-image
    pip install -r benchmark/requirements.txt
    python benchmark/bench_optimized.py --url http://<host>:8080

Arguments
---------
--url          Server endpoint base URL, e.g. http://localhost:8080
               The /v1/generate path is appended automatically.
--prompt       Text prompt (default: matches baseline prompt)
--max-tokens   max_new_tokens (default: 256, matches baseline)
--output       Path for JSON results
               (default: results/phase3_optimized_bench.json)
"""

import argparse
import base64
import io
import json
import os
import statistics
import time

import requests as http_requests
from PIL import Image

# ---------------------------------------------------------------------------
# Image spec — must match the baseline in phase3_baseline_first_result.json:
#   "320x240 solid color JPEG (synthetic test image)"
# seed=0 → same RGB color produced by the load_gen.py generator at index 0.
# ---------------------------------------------------------------------------
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
IMAGE_SEED = 0

# Baseline prompt (from phase3_baseline_first_result.json)
DEFAULT_PROMPT = "Describe what you see."

TOTAL_RUNS = 10
WARMUP_RUNS = 1  # discard this many leading results (torch.compile warm-up)
MEASURED_RUNS = TOTAL_RUNS - WARMUP_RUNS  # 9


def _make_image_b64(width: int, height: int, seed: int) -> str:
    """
    Generate a solid-color JPEG, base64-encoded, matching the load_gen.py
    convention: color is derived from the seed using Python's random.Random
    so the same seed always produces the same bytes.
    """
    import random
    rng = random.Random(seed)
    color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _send_one(url: str, image_b64: str, prompt: str, max_tokens: int) -> dict:
    payload = {"image": image_b64, "prompt": prompt, "max_new_tokens": max_tokens}
    t0 = time.perf_counter()
    try:
        resp = http_requests.post(url, json=payload, timeout=300)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        body = {}
        try:
            body = resp.json()
        except Exception:
            pass
        return {
            "http_status": resp.status_code,
            "client_latency_ms": round(elapsed_ms, 2),
            "server_latency_ms": body.get("latency_ms"),
            "optimized": body.get("optimized"),
            "dtype": body.get("dtype"),
            "gpu": body.get("gpu"),
            "response": body.get("response", ""),
            "error": None,
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "http_status": None,
            "client_latency_ms": round(elapsed_ms, 2),
            "server_latency_ms": None,
            "optimized": None,
            "dtype": None,
            "gpu": None,
            "response": "",
            "error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 optimized-path benchmark")
    parser.add_argument(
        "--url", required=True,
        help="Server base URL, e.g. http://localhost:8080",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="max_new_tokens (default: 256, matches baseline)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("results", "phase3_optimized_bench.json"),
    )
    args = parser.parse_args()

    endpoint = args.url.rstrip("/") + "/v1/generate"
    image_b64 = _make_image_b64(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_SEED)

    # Read dtype from /health before the run — /v1/generate responses do not
    # include a dtype field, so per-request body.get("dtype") always returns null.
    health_dtype = "unknown"
    try:
        health_resp = http_requests.get(args.url.rstrip("/") + "/health", timeout=10)
        if health_resp.status_code == 200:
            health_dtype = health_resp.json().get("dtype", "unknown")
    except Exception:
        pass

    print(f"\n=== Phase 3 Optimized-Path Benchmark ===")
    print(f"Endpoint     : {endpoint}")
    print(f"Image        : {IMAGE_WIDTH}×{IMAGE_HEIGHT} synthetic JPEG (seed={IMAGE_SEED})")
    print(f"Prompt       : {args.prompt}")
    print(f"max_tokens   : {args.max_tokens}")
    print(f"Total runs   : {TOTAL_RUNS}")
    print(f"Warm-up      : run 1 discarded (torch.compile first-call overhead)")
    print(f"Measured     : runs 2–{TOTAL_RUNS} (n={MEASURED_RUNS})")
    print(f"Output       : {args.output}")
    print()

    raw_results = []
    for i in range(1, TOTAL_RUNS + 1):
        label = "(warm-up)" if i <= WARMUP_RUNS else f"        "
        print(f"  Run {i:2d}/{TOTAL_RUNS} {label} ... ", end="", flush=True)
        result = _send_one(endpoint, image_b64, args.prompt, args.max_tokens)
        result["run"] = i
        result["warmup"] = i <= WARMUP_RUNS
        raw_results.append(result)

        if result["error"]:
            print(f"ERROR: {result['error']}")
        else:
            srv = (f"server={result['server_latency_ms']:.0f} ms  "
                   if result["server_latency_ms"] is not None else "")
            print(
                f"client={result['client_latency_ms']:.0f} ms  "
                f"{srv}"
                f"status={result['http_status']}"
            )

    # ------------------------------------------------------------------
    # Stats over measured runs (runs 2–10, discarding warm-up)
    # ------------------------------------------------------------------
    measured = [r for r in raw_results if not r["warmup"] and r["error"] is None]
    client_latencies = [r["client_latency_ms"] for r in measured]
    server_latencies = [r["server_latency_ms"] for r in measured
                        if r["server_latency_ms"] is not None]

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
        return {
            "n": len(vals),
            "mean": round(statistics.mean(vals), 2),
            "std": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0,
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
        }

    client_stats = _stats(client_latencies)
    server_stats = _stats(server_latencies)

    # Read GPU and optimized flag from first measured result.
    # dtype comes from health_dtype (fetched from /health before the run),
    # not from the per-request body which does not include a dtype field.
    first_measured = next((r for r in measured), {})
    gpu = first_measured.get("gpu", "unknown")
    dtype = health_dtype
    optimized = first_measured.get("optimized")

    print()
    print("--- Results (runs 2–10, warm-up discarded) ---")
    print(f"  n measured       : {client_stats['n']}")
    print(f"  GPU              : {gpu}")
    print(f"  dtype            : {dtype}")
    print(f"  optimized        : {optimized}")
    if client_stats["mean"] is not None:
        print(f"  Client latency   : {client_stats['mean']:.0f} ± {client_stats['std']:.0f} ms"
              f"  [min {client_stats['min']:.0f}  max {client_stats['max']:.0f}]")
    if server_stats["mean"] is not None:
        print(f"  Server latency   : {server_stats['mean']:.0f} ± {server_stats['std']:.0f} ms"
              f"  [min {server_stats['min']:.0f}  max {server_stats['max']:.0f}]")
        print(f"  (server latency excludes network round-trip)")

    # Compare to known baseline if available
    baseline_path = os.path.join("results", "phase3_baseline_first_result.json")
    if os.path.exists(baseline_path):
        try:
            with open(baseline_path) as f:
                baseline = json.load(f)
            baseline_ms = baseline.get("result", {}).get("latency_ms")
            if baseline_ms and server_stats["mean"] is not None:
                delta_pct = (server_stats["mean"] - baseline_ms) / baseline_ms * 100
                print()
                print(f"--- vs. Baseline ---")
                print(f"  Baseline (run 1) : {baseline_ms:.0f} ms")
                print(f"  Optimized mean   : {server_stats['mean']:.0f} ms")
                print(f"  Delta            : {delta_pct:+.1f}%"
                      f"  ({'faster' if delta_pct < 0 else 'slower'})")
                print(f"  NOTE: baseline was a single run; optimized is mean of {server_stats['n']}.")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Write JSON
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output = {
        "config": {
            "endpoint": endpoint,
            "image_spec": f"{IMAGE_WIDTH}x{IMAGE_HEIGHT} synthetic JPEG seed={IMAGE_SEED}",
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "total_runs": TOTAL_RUNS,
            "warmup_runs": WARMUP_RUNS,
            "measured_runs": MEASURED_RUNS,
        },
        "summary": {
            "gpu": gpu,
            "dtype": dtype,
            "optimized": optimized,
            "client_latency_ms": client_stats,
            "server_latency_ms": server_stats,
        },
        "raw": raw_results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to: {args.output}")


if __name__ == "__main__":
    main()
