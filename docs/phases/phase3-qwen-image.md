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

- **Qwen-Image** — image-language model (not FLUX or other image models)
- **Akamai LKE** — Kubernetes cluster with GPU node pools
- **RTX 4000 Ada** and/or **RTX PRO 6000 Blackwell** GPU nodes
- Python 3.11+ serving wrapper (FastAPI or similar — TBD)

## File layout (target)

```
phases/phase3-qwen-image/
├── README.md
├── requirements.txt
├── serve/
│   ├── app.py                # FastAPI serving wrapper
│   ├── model.py              # Qwen-Image model loading and inference
│   └── batching.py           # Dynamic batching logic
├── k8s/
│   ├── deployment.yaml       # LKE deployment manifest
│   ├── service.yaml          # LKE service manifest
│   └── gpu-node-pool.yaml    # Node pool configuration
├── benchmark/
│   ├── requirements.txt
│   ├── load_gen.py           # Image + prompt request generator
│   └── report.py             # Throughput and latency report
└── tests/
    └── test_model.py         # Smoke test: model loads and returns output
```

## Success criteria

- [ ] Model loads without error on target GPU.
- [ ] Single-request smoke test passes.
- [ ] Throughput benchmark runs at batch sizes 1, 4, 16.
- [ ] Results are written to a reproducible output file (JSON or CSV).

## Open questions

> TODO: Confirm which Qwen-Image variant and quantization level to use
> given GPU VRAM constraints (see docs/hardware.md PLACEHOLDER values).

> TODO: Decide serving framework — FastAPI, Triton Inference Server, or
> vLLM multimodal mode.

> TODO: Decide whether Phase 3 reuses the vLLM deployment from Phase 2
> or runs a separate serving stack.
