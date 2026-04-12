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

## File layout

```
phases/phase2-prefix-cache/
├── README.md
├── fermyon/
│   ├── Cargo.toml            # Rust crate — spin-sdk 3, sha2, serde_json
│   ├── Cargo.lock
│   ├── spin.toml             # Fermyon app manifest
│   └── src/
│       └── lib.rs            # Wasm handler: hash → Valkey GET → vLLM → Valkey SET
├── valkey/
│   ├── valkey.yaml           # LKE Deployment + Service + ConfigMap
│   └── config/
│       └── valkey.conf       # Standalone, allkeys-lru, 2 GB cap
├── vllm/
│   ├── vllm.yaml             # LKE Deployment + Service (GPU node)
│   └── serve_config.yaml     # vLLM flags including --enable-prefix-caching
├── benchmark/
│   ├── requirements.txt
│   ├── load_gen.py           # Request generator with configurable prefix-share rate
│   └── report.py             # Hit rate + latency report from load_gen output
└── tests/
    ├── __init__.py
    └── test_prefix_hash.py   # Hash contract tests + semantic-cache stub (skipped)
```

## Success criteria

- [ ] Fermyon function builds and deploys to Fermyon Cloud or local Spin.
- [ ] Valkey pod is healthy in LKE.
- [ ] vLLM pod starts with prefix caching enabled.
- [ ] Load generator produces requests with a 50% shared-prefix rate.
- [ ] Valkey hit rate exceeds 40% under that load.
- [ ] Cached requests show lower median latency than uncached requests.

## Decisions

| Decision | Resolution |
|---|---|
| Wasm language | Rust, using spin-sdk 3.x — mature SDK, strong async support |
| Valkey version | 8.0, standalone mode, allkeys-lru eviction, 2 GB memory cap |
| Prefix hashing | SHA-256 of the first 128 Unicode characters (not bytes) of the prompt, encoded as UTF-8 |
| vLLM model | Configured via `MODEL_NAME` in the Deployment ConfigMap — no model hardcoded in the serving layer |
