# System Architecture

## High-level request flow (Phase 2 target state)

```
Client Request
      |
      v
+---------------------+
| Fermyon Wasm Fn     |  <-- Phase 2 front door
| (prefix hash check) |      Runs on Akamai LKE CPU node
+---------------------+
      |           |
   Cache HIT   Cache MISS
      |           |
      v           v
+----------+  +------------------+
| Valkey   |  | vLLM backend     |  <-- Phase 2 backend
| (prefix  |  | LMCacheConnector |      Runs on Akamai LKE
|  cache)  |  |  + Valkey KV     |      GPU node pool
+----------+  +------------------+
                    |
                    v
          +------------------+
          | GPU Node Pool    |  <-- Phase 3 / Phase 4
          | RTX 4000 Ada     |      Akamai LKE GPU nodes
          |   (×2)           |
          +------------------+
```

## Phase map

| Phase | Layer | Component | Where it runs |
|---|---|---|---|
| 1 | Local demo | PyTorch KV cache | Developer laptop / CI |
| 2 | Edge | Fermyon Wasm Function | Akamai LKE (CPU node) |
| 2 | Cache | Valkey 8.0 standalone | Akamai LKE (CPU node) |
| 2 | Inference | vLLM + LMCacheConnectorV1 | Akamai LKE (GPU node) |
| 3 | Inference | Qwen-Image serving | Akamai LKE (GPU node) |
| 4 | Benchmarking | Throughput + cost harness | Akamai LKE (GPU node) |

## Computation reuse points

Each phase targets a different level of the stack:

1. **KV cache** (Phase 1) — reuse attention key/value tensors within a
   single model forward pass for autoregressive decoding.

2. **Prefix cache** (Phase 2) — reuse computed KV states across requests
   that share a common prompt prefix (system prompts, few-shot examples).

3. **Request deduplication** (Phase 2, Fermyon layer) — short-circuit
   semantically identical requests at the edge before they reach the GPU.

4. **Multi-GPU benchmarking** (Phase 4) — measure throughput and cost on
   each GPU tier to inform tensor-parallel vs data-parallel deployment
   decisions and produce a reproducible cost-per-token model.

---

## Akamai LKE cluster topology

**Cluster:** `akamai-lke-us-ord`  
**Region:** us-ord (Chicago)

### Node pools

| Pool ID | Plan | Type | Count | Workloads |
|---|---|---|---|---|
| 865821 | g6-dedicated-4 | CPU | 1 | Valkey, Fermyon |
| 868011 | g2-gpu-rtx4000a1-l | RTX 4000 Ada 20 GB | 2 | vLLM, Qwen-Image |

### Node names and assignments

| Node name | Pool | Role | Workload |
|---|---|---|---|
| `lke591117-865821-4a82b6ec0000` | 865821 (CPU) | CPU | Valkey, Fermyon |
| `lke591117-868011-613ea4520000` | 868011 (GPU) | Ada GPU 1 | qwen-image |
| `lke591117-868011-4171fee90000` | 868011 (GPU) | Ada GPU 2 | vllm |

### Node labels

Labels are applied manually after provisioning — LKE does not automatically
propagate pool-level labels to Kubernetes Node objects.

| Label | Value | Nodes | Set by |
|---|---|---|---|
| `workload-type` | `cpu` | CPU node | `kubectl label node` (manual) |
| `gpu-type` | `rtx4000ada` | Both GPU nodes | `kubectl label node` (manual) |

These labels are the targets for all `nodeSelector` fields in phase manifests.
Node names change on pool recreation; labels survive as long as they are
reapplied — see `infrastructure/README.md` for the post-provisioning checklist.

---

## Deployments

**Namespace:** `inference`

| Deployment | Phase | Node | Image |
|---|---|---|---|
| `vllm` | 2 | GPU (rtx4000ada) | `ghcr.io/jginsj/vllm-lmcache:v0.18.1` |
| `qwen-image` | 3 | GPU (rtx4000ada) | `ghcr.io/jginsj/qwen-image-server:latest` |
| `valkey` | 2 | CPU (workload-type=cpu) | `valkey/valkey:8.0-alpine` |
| `fermyon-prefix-cache` | 2 | CPU (workload-type=cpu) | `ghcr.io/jginsj/fermyon-prefix-cache:latest` |

### Services

| Service | Port | Selector |
|---|---|---|
| `vllm-svc` | 8000 | `app: vllm` |
| `qwen-image` | 8080 | `app: qwen-image` |
| `valkey-svc` | 6379 | `app: valkey` |
| `fermyon-svc` | 8082 | `app: fermyon-prefix-cache` |

### PersistentVolumeClaims

| PVC name | Size | Used by | Phase |
|---|---|---|---|
| `vllm-model-cache` | 20 Gi | vllm Deployment | 2 |
| `qwen-image-hf-cache` | 30 Gi | qwen-image Deployment | 3 |

Both PVCs use the default `linode-block-storage` storage class (no
`storageClassName` set).

---

## Repository layout

```
akamai-inference-optimization/
├── CLAUDE.md                          ← session rules and open questions log
├── README.md
├── pyproject.toml
│
├── infrastructure/
│   ├── README.md                      ← post-cluster-creation checklist
│   └── terraform/
│       ├── cluster.tf                 ← LKE cluster + CPU node pool
│       ├── main.tf
│       ├── variables.tf
│       ├── outputs.tf
│       ├── node-pool-ada.tf           ← RTX 4000 Ada pool (node_count=2)
│       ├── node-pool-ada-2.tf         ← intentionally empty; scaling via node_count
│       └── node-pool-blackwell.tf     ← RTX PRO 6000 Blackwell stub (not yet active)
│
├── docs/
│   ├── architecture.md                ← this file
│   ├── hardware.md
│   ├── phase1-kv-cache-build-log.md
│   ├── phase2-fermyon-build-log.md
│   ├── phase3-qwen-image-build-log.md
│   ├── phase4-build-log.md
│   └── phases/
│       ├── phase1-kv-cache.md
│       ├── phase2-fermyon-valkey.md
│       ├── phase3-qwen-image.md
│       └── phase4-benchmarks.md
│
├── phases/
│   ├── phase1-kv-cache/
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── demo.py
│   │   ├── kv_cache/
│   │   │   ├── __init__.py
│   │   │   ├── attention.py
│   │   │   ├── generate.py
│   │   │   ├── model.py
│   │   │   └── tokenizer.py
│   │   ├── tests/
│   │   │   ├── __init__.py
│   │   │   └── test_attention.py
│   │   └── results/
│   │       └── phase1_timing.json
│   │
│   ├── phase2-prefix-cache/
│   │   ├── README.md
│   │   ├── fermyon/
│   │   │   ├── Cargo.toml             ← workspace root
│   │   │   ├── spin.toml
│   │   │   ├── src/lib.rs             ← legacy (workspace uses proxy/ and health/)
│   │   │   ├── proxy/
│   │   │   │   ├── Cargo.toml
│   │   │   │   └── src/lib.rs         ← POST /v1/chat/completions handler
│   │   │   ├── health/
│   │   │   │   ├── Cargo.toml
│   │   │   │   └── src/lib.rs         ← GET /health handler
│   │   │   └── k8s/
│   │   │       └── fermyon-deployment.yaml
│   │   ├── valkey/
│   │   │   └── valkey.yaml
│   │   ├── vllm/
│   │   │   ├── Dockerfile
│   │   │   ├── lmcache-configmap.yaml
│   │   │   ├── pvc-model-cache.yaml
│   │   │   ├── serve_config.yaml
│   │   │   └── vllm.yaml
│   │   ├── benchmark/
│   │   │   ├── requirements.txt
│   │   │   ├── bench_cache.py         ← three-pass cache-value benchmark
│   │   │   ├── load_gen.py
│   │   │   └── report.py
│   │   ├── tests/
│   │   │   ├── __init__.py
│   │   │   ├── test_chat_cache.py
│   │   │   └── test_prefix_hash.py
│   │   └── results/
│   │       ├── phase2_cache_benchmark.json
│   │       ├── phase2_prefix_cache_baseline.json
│   │       └── phase2_valkey_verified.json
│   │
│   ├── phase3-qwen-image/
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── Dockerfile
│   │   ├── serve/
│   │   │   ├── __init__.py
│   │   │   ├── app.py
│   │   │   ├── batching.py
│   │   │   ├── image_utils.py
│   │   │   ├── model.py
│   │   │   └── model_optimized.py
│   │   ├── k8s/
│   │   │   ├── deployment.yaml
│   │   │   ├── service.yaml
│   │   │   ├── pvc-model-cache.yaml
│   │   │   └── gpu-node-pool.yaml
│   │   ├── benchmark/
│   │   │   ├── requirements.txt
│   │   │   ├── bench_optimized.py
│   │   │   ├── load_gen.py
│   │   │   └── report.py
│   │   ├── tests/
│   │   │   ├── __init__.py
│   │   │   └── test_model.py
│   │   └── results/
│   │       ├── phase3_baseline_first_result.json
│   │       └── phase3_optimized_bench.json
│   │
│   └── phase4-benchmarks/
│       ├── README.md
│       ├── requirements.txt
│       ├── benchmark/
│       │   ├── __init__.py
│       │   └── benchmark.py           ← synchronous concurrency sweep, CSV output
│       ├── harness/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   ├── cost_model.py
│       │   ├── load_gen.py
│       │   └── run_benchmark.py
│       ├── configs/
│       │   ├── rtx4000ada.yaml        ← gpu_hourly_usd: 0.96
│       │   └── rtxpro6000.yaml        ← PLACEHOLDER pricing
│       ├── k8s/
│       │   ├── vllm-ada.yaml          ← single-GPU Ada, Phase 4 target
│       │   ├── vllm-tp.yaml           ← TP-2 Blackwell (future work)
│       │   ├── service.yaml
│       │   ├── benchmark-job-ada.yaml
│       │   └── benchmark-job-blackwell.yaml
│       ├── report/
│       │   └── generate_report.py
│       └── tests/
│           ├── __init__.py
│           ├── test_benchmark.py
│           └── test_cost_model.py
│
└── results/
    └── phase4_raw_benchmark.csv       ← gitignored; generated by benchmark.py
```

---

## Critical Gotchas

- **Akamai Object Storage limited access keys only work on E0 and E1 endpoints.**
  E2 and E3 endpoints return 403 on PutObject with limited keys. Use an unlimited
  access key or an E1 bucket for Terraform state.
