# Phase 3 — Qwen-Image Inference on Akamai Cloud

## Goal

Deploy Qwen-Image for end-to-end image model serving on Akamai LKE,
covering model loading, batching, and request routing between GPU tiers.

## Inputs and outputs

| | Detail |
|---|---|
| Input | HTTP POST with `{"image": "<base64>", "prompt": "..."}` |
| Output | `{"response": "...", "latency_ms": N, "gpu": "..."}` |
| Side output | Throughput (requests/sec) at batch sizes 1, 4, 16 |
| Side output | GPU memory utilisation during inference |

## Key technologies

- **Qwen2.5-VL** (`Qwen/Qwen2.5-VL-7B-Instruct`) — vision-language model (not FLUX or other image models)
- **FastAPI** — serving wrapper with dynamic batching via `DynamicBatcher`
- **Akamai LKE** — Kubernetes cluster with GPU node pools
- **RTX 4000 Ada** and/or **RTX PRO 6000 Blackwell** GPU nodes
- Python 3.11+, PyTorch 2.4+, transformers 4.x

## File layout

```
phases/phase3-qwen-image/
├── README.md
├── requirements.txt          # torch>=2.4, transformers>=4.45,<5.0, Pillow, FastAPI
├── serve/
│   ├── __init__.py           # Exports only batching layer (no ML deps at import time)
│   ├── app.py                # FastAPI server — POST /v1/generate + GET /health
│   ├── model.py              # Baseline loader (float16, eager attention, no compile)
│   ├── model_optimized.py    # EXPERIMENTAL: flash_attention_2, bfloat16, torch.compile
│   ├── batching.py           # DynamicBatcher, Future, InferenceRequest
│   └── image_utils.py        # decode_image() — no PyTorch/transformers dependency
├── k8s/
│   ├── deployment.yaml       # LKE Deployment (GPU, ConfigMap)
│   ├── service.yaml          # ClusterIP on port 8080
│   └── gpu-node-pool.yaml    # Akamai LKE node pool spec (PLACEHOLDER)
├── benchmark/
│   ├── requirements.txt
│   ├── load_gen.py           # Synthetic-image load generator
│   └── report.py             # Throughput and latency report
└── tests/
    ├── __init__.py
    └── test_model.py         # Infrastructure tests (no GPU); GPU tests skipped without CUDA
```

## Success criteria

- [ ] Model loads without error on target GPU.
- [ ] Single-request smoke test passes.
- [ ] Throughput benchmark runs at batch sizes 1, 4, 16.
- [ ] Results are written to a reproducible output file (JSON or CSV).

## Decisions

| Decision | Resolution |
|---|---|
| Model variant | `Qwen/Qwen2.5-VL-7B-Instruct`, float16 (overrideable via `MODEL_NAME` env var) |
| Quantization | None by default; bfloat16 available as EXPERIMENTAL flag in `model_optimized.py` |
| Serving framework | FastAPI with a custom `DynamicBatcher` — separate from Phase 2's vLLM stack |
| Phase 2 reuse | Phase 3 runs a separate serving stack; integration with Phase 2 Valkey cache is noted as future work in the README |
| Local development | `torch>=2.4` wheels unavailable on Intel Mac; infrastructure tests run without torch by installing only `transformers<5.0` and Pillow |
