"""
Phase 3 load generator — Qwen-Image throughput benchmark.

Sends image+text requests to the Phase 3 serving endpoint at configurable
batch sizes (sequential bursts of 1, 4, and 16).  Records per-request
latency and writes raw results to results/phase3_loadgen.json.

No performance conclusions are drawn here.  Use report.py to analyse.
All timings are wall-clock from request-sent to response-received;
they include network round-trip and are not pure GPU time.

Synthetic images
----------------
Images are generated programmatically — no external files required.
Each image is a solid-color JPEG sized IMAGE_SIZE × IMAGE_SIZE pixels.
Using solid colors keeps the visual content trivial while exercising the
full image tokenisation pipeline (encode → base64 → decode → vision encoder).

Usage
-----
    cd phases/phase3-qwen-image
    pip install -r benchmark/requirements.txt
    python benchmark/load_gen.py --url http://<host>:8080 --runs 20

Arguments
---------
--url           Server endpoint (required)
--runs          Number of requests per batch-size level (default: 10)
--batch-sizes   Comma-separated batch sizes to test (default: 1,4,16)
                NOTE: "batch size" here means sequential burst size —
                      requests are sent one at a time within each burst
                      to measure per-request latency cleanly.
--prompt        Text prompt sent with every image (default: generic caption prompt)
--max-tokens    max_new_tokens forwarded to the server (default: 64)
--image-size    Side length of synthetic images in pixels (default: 224)
--output        Path for JSON results (default: results/phase3_loadgen.json)
--tag           Optional label for this run (e.g. "baseline" or "optimized")
"""

import argparse
import base64
import io
import json
import os
import random
import time
from typing import Any

import requests as http_requests
from PIL import Image

DEFAULT_PROMPT = "Describe this image in one sentence."
IMAGE_SIZE = 224
SEED = 42


def _make_image_b64(size: int = IMAGE_SIZE, seed: int = 0) -> str:
    """Generate a solid-color synthetic image as a base64-encoded JPEG string."""
    rng = random.Random(seed)
    color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
    img = Image.new("RGB", (size, size), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _send_one(
    url: str,
    image_b64: str,
    prompt: str,
    max_tokens: int,
    request_index: int,
    batch_size: int,
) -> dict[str, Any]:
    payload = {"image": image_b64, "prompt": prompt, "max_new_tokens": max_tokens}
    t0 = time.perf_counter()
    try:
        resp = http_requests.post(url, json=payload, timeout=180)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        try:
            body = resp.json()
        except Exception:
            body = {}
        return {
            "request_index": request_index,
            "batch_size": batch_size,
            "http_status": resp.status_code,
            "latency_ms": round(elapsed_ms, 2),
            "server_latency_ms": body.get("latency_ms"),
            "optimized": body.get("optimized"),
            "gpu": body.get("gpu"),
            "error": None,
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "request_index": request_index,
            "batch_size": batch_size,
            "http_status": None,
            "latency_ms": round(elapsed_ms, 2),
            "server_latency_ms": None,
            "optimized": None,
            "gpu": None,
            "error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 load generator")
    parser.add_argument("--url", required=True,
                        help="Server endpoint, e.g. http://localhost:8080/v1/generate")
    parser.add_argument("--runs", type=int, default=10,
                        help="Requests per batch-size level")
    parser.add_argument("--batch-sizes", default="1,4,16",
                        help="Comma-separated burst sizes to test")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--output", default=os.path.join("results", "phase3_loadgen.json"))
    parser.add_argument("--tag", default="",
                        help="Label for this run, e.g. 'baseline' or 'optimized'")
    args = parser.parse_args()

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    random.seed(SEED)

    # Pre-generate images — one unique image per request index so results
    # are reproducible and each request exercises the vision tokenizer.
    total_requests = args.runs * len(batch_sizes)
    images = [_make_image_b64(args.image_size, seed=i) for i in range(total_requests)]

    print(f"\n=== Phase 3 Load Generator ===")
    print(f"Target        : {args.url}")
    print(f"Batch sizes   : {batch_sizes}")
    print(f"Runs/level    : {args.runs}")
    print(f"Image size    : {args.image_size}×{args.image_size} px (synthetic)")
    print(f"Tag           : {args.tag or '(none)'}")
    print(f"Output        : {args.output}")
    print()

    all_results = []
    img_idx = 0

    for batch_size in batch_sizes:
        print(f"--- Batch size {batch_size} ---")
        burst_results = []
        for run in range(args.runs):
            burst = []
            for _ in range(batch_size):
                img_b64 = images[img_idx % len(images)]
                img_idx += 1
                result = _send_one(
                    url=args.url,
                    image_b64=img_b64,
                    prompt=args.prompt,
                    max_tokens=args.max_tokens,
                    request_index=len(all_results) + len(burst),
                    batch_size=batch_size,
                )
                burst.append(result)
            burst_results.extend(burst)
            ok = sum(1 for r in burst if r["error"] is None)
            avg_ms = (
                sum(r["latency_ms"] for r in burst if r["error"] is None) / ok
                if ok else 0
            )
            print(f"  run {run + 1:3d}/{args.runs}: "
                  f"{ok}/{batch_size} ok, avg {avg_ms:.0f} ms")
        all_results.extend(burst_results)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    payload = {
        "config": {
            "url": args.url,
            "runs_per_level": args.runs,
            "batch_sizes": batch_sizes,
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "image_size": args.image_size,
            "tag": args.tag,
            "seed": SEED,
        },
        "results": all_results,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults written to: {args.output}")
    print("Run benchmark/report.py to analyse.")


if __name__ == "__main__":
    main()
