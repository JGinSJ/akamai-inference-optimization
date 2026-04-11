# Phase 2 — Fermyon + Valkey + vLLM Prefix Caching

A prefix-aware inference pipeline where a Fermyon Wasm Function at the
Akamai edge intercepts requests, checks a Valkey cache, and only forwards
cache misses to a vLLM backend.

## Why this reduces inference cost

Phase 1 showed that KV caching saves compute *within* a single model run.
Phase 2 saves compute *across* runs by caching entire responses.

### Two complementary caching layers

```
Request
  │
  ▼
Fermyon Wasm Function          ← Layer 1: response cache (Valkey)
  │  hash(prompt[:N])
  │  GET from Valkey
  ├── HIT  ──────────────────────► return cached response  (0 GPU work)
  │
  └── MISS ─────────────────────► vLLM backend
                                     │  --enable-prefix-caching
                                     │                          ← Layer 2: KV cache (GPU)
                                     │  requests sharing a prefix
                                     │  reuse cached KV states
                                     ▼
                                  GPU generates response
                                     │
                                     └── store in Valkey ──► return to client
```

**Layer 1 (Valkey):** Full response cache.  If the same (or same-prefix)
prompt was seen before, return the cached answer immediately — no GPU
compute at all.

**Layer 2 (vLLM):** KV-state cache on the GPU.  For cache misses that share
a prompt prefix (e.g. a system prompt), vLLM reuses the already-computed
attention key/value states for the shared prefix, reducing the prefill cost.
This is the same mechanism demonstrated from scratch in Phase 1.

### Cache key derivation

The Valkey key for a prompt is:

```
SHA-256( prompt[:prefix_chars].encode("utf-8") ).hexdigest()
```

`prefix_chars` defaults to 128 Unicode characters (not bytes).  Two prompts
that share the same first 128 characters share a cache entry.

This is **exact-match** prefix caching.  See [Future work](#future-work) for
the semantic caching path.

---

## Components

| Component | Technology | Where it runs |
|---|---|---|
| Front door | Fermyon Wasm (Rust) | Fermyon Cloud or self-hosted Spin |
| Response cache | Valkey 8.0 standalone | Akamai LKE (CPU node) |
| Inference backend | vLLM with prefix caching | Akamai LKE (GPU node) |

---

## File layout

```
phase2-prefix-cache/
├── fermyon/
│   ├── Cargo.toml          # Rust crate — spin-sdk 3, sha2, serde
│   ├── spin.toml           # Fermyon app manifest
│   └── src/
│       └── lib.rs          # Async HTTP handler: hash → Valkey → vLLM
├── valkey/
│   ├── valkey.yaml         # LKE Deployment + Service + ConfigMap
│   └── config/
│       └── valkey.conf     # Standalone, allkeys-lru, 2 GB cap
├── vllm/
│   ├── vllm.yaml           # LKE Deployment + Service (GPU node)
│   └── serve_config.yaml   # vLLM flags including --enable-prefix-caching
├── benchmark/
│   ├── requirements.txt
│   ├── load_gen.py         # Send N requests with configurable prefix-share rate
│   └── report.py           # Hit rate + latency report from load_gen output
└── tests/
    └── test_prefix_hash.py # Hash algorithm contract tests + semantic-cache stub
```

---

## Setup

### Prerequisites

| Tool | Purpose |
|---|---|
| Rust + `wasm32-wasip1` target | Build the Fermyon Wasm binary |
| `spin` CLI | Run and deploy the Fermyon app |
| `kubectl` | Apply Valkey and vLLM manifests to LKE |
| Python 3.11+ | Benchmark and tests |

Install the Rust Wasm target:
```bash
rustup target add wasm32-wasip1
```

Install Spin CLI: follow [developer.fermyon.com](https://developer.fermyon.com/spin/install).

### Build the Wasm handler

```bash
cd phases/phase2-prefix-cache/fermyon
cargo build --target wasm32-wasip1 --release
```

The binary is written to
`fermyon/target/wasm32-wasip1/release/prefix_cache_handler.wasm`.

### Run locally with Spin

```bash
cd phases/phase2-prefix-cache/fermyon
spin up \
  --variable valkey_address=redis://localhost:6379 \
  --variable vllm_url=http://localhost:8000
```

The handler listens on `http://localhost:3000/v1/completions`.

### Deploy to LKE

```bash
# Create the namespace and deploy Valkey
kubectl apply -f phases/phase2-prefix-cache/valkey/valkey.yaml

# Deploy vLLM (edit vllm/vllm.yaml first: set MODEL_NAME and node selector)
kubectl apply -f phases/phase2-prefix-cache/vllm/vllm.yaml

# Deploy Fermyon app to Fermyon Cloud (or use `spin up` pointed at LKE services)
cd phases/phase2-prefix-cache/fermyon
spin deploy \
  --variable valkey_address=redis://valkey-svc.inference:6379 \
  --variable vllm_url=http://vllm-svc.inference:8000
```

**Before deploying vLLM:** edit `vllm/vllm.yaml` and set:
- `MODEL_NAME` in the ConfigMap
- `akamai.com/gpu-node-pool` nodeSelector label matching your LKE pool

---

## Request / response format

**Request:**
```json
POST /v1/completions
{"prompt": "You are a helpful assistant. What is 2+2?"}
```

**Response:**
```json
{
  "response": "<model output or cached response>",
  "cache_hit": true,
  "latency_ms": 4
}
```

`cache_hit: true` means the response was served from Valkey without
touching the GPU.

---

## Running the benchmark

```bash
cd phases/phase2-prefix-cache
pip install -r benchmark/requirements.txt

# Generate load (requires a running stack)
python benchmark/load_gen.py \
    --url http://<fermyon-host>/v1/completions \
    --requests 200 \
    --prefix-share 0.5

# Analyse results (works offline — reads results/phase2_loadgen.json)
python benchmark/report.py
```

---

## Running the tests

The prefix-hash tests require no running infrastructure.

```bash
cd phases/phase2-prefix-cache
python -m pytest tests/ -v
```

Tests cover:
- Key format: 64-character lowercase hex string
- Determinism: same prompt → same key, always
- Prefix truncation: `prefix_chars` controls the cache granularity
- Same-prefix/different-suffix → same cache key
- UTF-8 correctness: characters (not bytes) are counted
- Pinned SHA-256 values as a cross-language compatibility check
- `test_semantic_cache_equivalent_prompts_share_key` — **skipped**, marks
  the semantic caching work as not yet implemented

---

## Future work

### Semantic caching (TODO)

The current implementation is **exact-match**: two prompts share a cache
entry only if their first `prefix_chars` characters are byte-for-byte
identical.

A future implementation would:
1. Embed the prompt prefix with a sentence-transformer model.
2. Query a vector store (e.g. pgvector, Qdrant) for the nearest cached
   embedding within a similarity threshold.
3. Return the nearest neighbour's cached response if similarity exceeds the
   threshold; otherwise proceed as a cache miss.

This would catch semantically equivalent prompts that are worded differently
(rephrased system prompts, synonym substitutions, language variants).

Tracking: `tests/test_prefix_hash.py::test_semantic_cache_equivalent_prompts_share_key`
(currently skipped with a full TODO explanation).

### Other deferred items

- Valkey cluster mode for horizontal scale and HA
- TLS between Fermyon and Valkey
- Cache TTL / expiry policy (currently entries live until LRU eviction)
- CI/CD pipeline for Wasm build and LKE deploy
- `kubectl` wait / health-check scripts for the deploy sequence

---

## Success criteria

- [ ] `cargo build --target wasm32-wasip1 --release` succeeds.
- [ ] `spin up` starts the handler locally with mocked Valkey/vLLM.
- [ ] Valkey pod is healthy in LKE (`kubectl get pods -n inference`).
- [ ] vLLM pod starts with `--enable-prefix-caching` confirmed in logs.
- [ ] `python -m pytest tests/ -v` passes (excluding skipped semantic test).
- [ ] Load generator produces requests with a 50 % shared-prefix rate.
- [ ] `report.py` shows Valkey hit rate ≥ 40 % under that load.
- [ ] Cached requests show lower median latency than uncached requests.
