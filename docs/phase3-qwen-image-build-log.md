# Phase 3 Qwen-Image Build Log

*2026-04-16*

---

## Optimization results

### Test conditions

- **GPU**: NVIDIA RTX 4000 Ada Generation (Akamai LKE, `g2-gpu-rtx4000a1-l`)
- **Model**: `Qwen/Qwen2.5-VL-7B-Instruct`
- **Image**: 320×240 synthetic JPEG, solid color, seed=0 — same image as the baseline measurement
- **Prompt**: "Describe what you see."
- **max\_new\_tokens**: 256
- **Metric**: server-side `latency_ms` from the `/v1/generate` response body (excludes network round-trip)

### Baseline

Single request, no optimizations, April 14 2026.
Source: `results/phase3_baseline_first_result.json`.

```
3,434 ms
```

### Optimized

10 sequential requests. Run 1 discarded (torch.compile warm-up). Runs 2–10 measured.
Source: `results/phase3_optimized_bench.json`. Recorded April 16 2026.

Active optimizations: `OPTIMIZED=1`

- **bfloat16** — model loaded with `torch_dtype=torch.bfloat16`
- **sdpa** — `attn_implementation="sdpa"`, PyTorch's built-in `scaled_dot_product_attention`, dispatching to fused CUDA kernels on the RTX 4000 Ada (compute capability 8.9)
- **torch.compile** — `mode="reduce-overhead"`, applied after model load

```
Server latency (runs 2–10, n=9)
  mean : 2,069.47 ms
  std  :     2.94 ms
  min  : 2,067.44 ms
  max  : 2,075.35 ms
```

Warmup run (run 1, discarded): 2,873.89 ms server latency — consistent with torch.compile
tracing on first call.

### Improvement

```
Baseline (single run)   : 3,434 ms
Optimized mean (n=9)    : 2,069 ms
Reduction               : −1,365 ms  (−39.7%)
```

The 2.94 ms standard deviation over runs 2–10 indicates the optimized path is stable once
the torch.compile warm-up is complete.

---

## Build decisions

### flash-attn abandoned — sdpa used instead

The original plan called for `attn_implementation="flash_attention_2"` via the `flash-attn`
package. Two attempts were made to build it in the Dockerfile:

1. **CUDA version mismatch** — the initial base image (`nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04`)
   provided CUDA 12.1 headers but `torch` installed as `2.11.0+cu130` (compiled against CUDA 13.0).
   Flash-attn's build system requires an exact match. Fixed by switching to
   `nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04`.

2. **Memory exhaustion** — even with `MAX_JOBS=2`, parallel CUDA kernel compilation exhausted
   available RAM on the Mac CPU build host. The build could not complete regardless of job count.

**Resolution**: switched `attn_implementation` from `"flash_attention_2"` to `"sdpa"`.
PyTorch 2.x includes `torch.nn.functional.scaled_dot_product_attention` as a built-in that
dispatches to the same fused CUDA kernels on Ada and newer GPUs at runtime — no source
compilation, no extra package, same hardware path. The flash-attn install step and the
`MAX_JOBS` env var were removed from the Dockerfile entirely.

### dtype field reads null in benchmark output — fixed in bench_optimized.py

`dtype` was `null` in every row of `results/phase3_optimized_bench.json`. Root cause:
`bench_optimized.py` read `dtype` from the `/v1/generate` response body, but `GenerateResponse`
in `serve/app.py` does not include a `dtype` field — it is only exposed by `/health`.

**Fix applied to `benchmark/bench_optimized.py`:** the script now calls `GET /health` once
before the run loop and stores `dtype` from the response. This value is used in the summary
printout and JSON output in place of the always-null per-request field. The `/v1/generate`
response schema was not changed (adding `dtype` there would require a server rebuild; reading
from `/health` is the minimal fix with no deployment impact).

The existing `results/phase3_optimized_bench.json` still shows `"dtype": null` in its raw
rows — it was recorded before this fix. The `/health` endpoint confirmed `"dtype": "bfloat16"`
at the time of that run. The latency numbers are unaffected.
