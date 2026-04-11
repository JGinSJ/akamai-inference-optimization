# Phase 2 — Fermyon + Valkey + vLLM Prefix Caching

## Goal

Deploy a prefix-aware inference pipeline where a Fermyon Wasm Function
at the Akamai edge intercepts requests, checks a Valkey cache for
matching prompt prefixes, and only forwards cache misses to a vLLM
backend running on Akamai LKE.

## Inputs and outputs

| | Detail |
|---|---|
| Input | HTTP POST to Fermyon function with `{"prompt": "..."}` |
| Output | `{"response": "...", "cache_hit": true/false, "latency_ms": N}` |
| Side output | Valkey hit/miss rate over a benchmark run |
| Side output | End-to-end latency comparison: cached vs uncached requests |

## Key technologies

- **Fermyon Wasm Functions** — front door, edge hash check, cache lookup
- **Valkey** — prefix KV cache (not Redis)
- **vLLM** — LLM backend with `--enable-prefix-caching` flag
- **Akamai LKE** — Kubernetes cluster hosting Valkey and vLLM pods
- Python 3.11+ for benchmark harness and load generation

## Architecture

```
Client
  |
  v
Fermyon Wasm Function (edge)
  |-- hash prompt prefix
  |-- GET from Valkey
  |   |-- HIT  --> return cached response
  |   `-- MISS --> forward to vLLM
  |
  v
vLLM pod (LKE GPU node)
  |-- prefix caching enabled
  `-- store response in Valkey
```

## File layout (target)

```
phases/phase2-prefix-cache/
├── README.md
├── fermyon/
│   ├── spin.toml             # Fermyon app manifest
│   └── src/
│       └── handler.rs        # Wasm function (Rust or TinyGo — TBD)
├── valkey/
│   ├── valkey.yaml           # LKE deployment manifest
│   └── config/
│       └── valkey.conf       # Valkey configuration
├── vllm/
│   ├── vllm.yaml             # LKE deployment manifest
│   └── serve_config.yaml     # vLLM serving arguments
├── benchmark/
│   ├── requirements.txt
│   ├── load_gen.py           # Request generator with shared prefixes
│   └── report.py             # Parse results, print hit rate + latency
└── tests/
    └── test_prefix_hash.py   # Unit test: same prefix -> same hash
```

## Success criteria

- [ ] Fermyon function builds and deploys to Fermyon Cloud or local Spin.
- [ ] Valkey pod is healthy in LKE.
- [ ] vLLM pod starts with prefix caching enabled.
- [ ] Load generator produces requests with a 50% shared-prefix rate.
- [ ] Valkey hit rate exceeds 40% under that load.
- [ ] Cached requests show lower median latency than uncached requests.

## Open questions

> TODO: Decide Fermyon Wasm language — Rust (mature SDK) or TinyGo (smaller
> binary). Affects fermyon/ source layout.

> TODO: Confirm Valkey version and whether standalone or cluster mode is
> needed for Phase 2 scale.

> TODO: Confirm vLLM model to use (size, quantization) given LKE GPU pool.

> TODO: Define prefix hashing scheme — full prefix bytes, or token IDs?
