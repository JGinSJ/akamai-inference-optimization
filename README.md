# Zero-Waste Inference on Akamai Cloud

A production-quality demonstration of inference cost reduction through
computation reuse, deployed on Akamai Cloud (LKE) with NVIDIA RTX 4000 Ada
and RTX PRO 6000 Blackwell GPU targets.

## What this project shows

LLM inference is expensive because compute is repeatedly thrown away.
This project demonstrates four techniques for reusing computation at
different points in the inference stack:

| Phase | Technique | Key Technologies |
|-------|-----------|-----------------|
| 1 | KV cache from scratch | PyTorch transformer demo |
| 2 | Prefix cache at the edge | Fermyon Wasm, Valkey, vLLM |
| 3 | Image model inference | Qwen-Image on Akamai Cloud |
| 4 | Multi-GPU benchmarking & cost model | RTX 4000 Ada vs RTX PRO 6000 Blackwell |

## Repository layout

```
akamai-inference-optimization/
├── phases/
│   ├── phase1-kv-cache/        # PyTorch KV cache demo
│   ├── phase2-prefix-cache/    # Fermyon + Valkey + vLLM
│   ├── phase3-qwen-image/      # Qwen-Image inference
│   └── phase4-benchmarks/      # Multi-GPU cost model
├── docs/
│   ├── architecture.md         # System diagram and phase map
│   ├── hardware.md             # GPU target specifications
│   └── phases/                 # Per-phase scope documents
├── CLAUDE.md                   # AI session working rules
├── LICENSE                     # Apache 2.0
└── pyproject.toml              # Root workspace marker
```

## Prerequisites

- Python 3.11 or later
- pip
- NVIDIA GPU (RTX 4000 Ada or RTX PRO 6000 Blackwell for full benchmarks)
- Access to an Akamai Cloud / LKE cluster (Phase 2+)
- Fermyon Cloud account or self-hosted Spin runtime (Phase 2)

## Quick start

> TODO: Fill in once Phase 1 is complete.

Each phase is self-contained. Navigate to the relevant `phases/` directory
and follow the README there.

```bash
# Example (Phase 1 — available after Phase 1 implementation)
cd phases/phase1-kv-cache
pip install -r requirements.txt
python demo.py
```

## Phases at a glance

### Phase 1 — KV Cache from Scratch

A minimal PyTorch transformer that makes the key-value cache visible and
measurable. No HuggingFace dependencies. Goal: show exactly what is being
reused and what is being recomputed on each forward pass.

See [docs/phases/phase1-kv-cache.md](docs/phases/phase1-kv-cache.md).

### Phase 2 — Fermyon + Valkey + vLLM Prefix Caching

A Fermyon Wasm Function sits at the front door. It hashes prompt prefixes,
checks a Valkey cache, and short-circuits requests that share a prefix.
Misses fall through to a vLLM backend with prefix caching enabled.

See [docs/phases/phase2-fermyon-valkey.md](docs/phases/phase2-fermyon-valkey.md).

### Phase 3 — Qwen-Image Inference on Akamai Cloud

End-to-end image model serving using Qwen-Image, deployed to Akamai LKE
with GPU node pools. Covers model loading, batching, and request routing.

See [docs/phases/phase3-qwen-image.md](docs/phases/phase3-qwen-image.md).

### Phase 4 — Multi-GPU Benchmarking and Cost Model

Head-to-head throughput and cost comparison between RTX 4000 Ada and
RTX PRO 6000 Blackwell on Akamai Cloud. Produces a reproducible cost model.

See [docs/phases/phase4-benchmarks.md](docs/phases/phase4-benchmarks.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).
