# System Architecture

> TODO: Replace ASCII diagram with a rendered diagram once the full system
> is built. All layer descriptions below are accurate to the design intent;
> update as each phase is implemented.

## High-level request flow (Phase 2 target state)

```
Client Request
      |
      v
+---------------------+
| Fermyon Wasm Fn     |  <-- Phase 2 front door
| (prefix hash check) |      Runs at Akamai edge PoP
+---------------------+
      |           |
   Cache HIT   Cache MISS
      |           |
      v           v
+----------+  +------------------+
| Valkey   |  | vLLM backend     |  <-- Phase 2 backend
| (prefix  |  | (prefix caching  |      Runs on Akamai LKE
|  cache)  |  |  enabled)        |      GPU node pool
+----------+  +------------------+
                    |
                    v
          +------------------+
          | GPU Node Pool    |  <-- Phase 3 / Phase 4
          | RTX 4000 Ada     |      Akamai LKE GPU nodes
          |   or             |
          | RTX PRO 6000     |
          | Blackwell        |
          +------------------+
```

## Phase map

| Phase | Layer | Component | Where it runs |
|-------|-------|-----------|---------------|
| 1 | Local demo | PyTorch KV cache | Developer laptop / CI |
| 2 | Edge | Fermyon Wasm Function | Akamai edge PoP |
| 2 | Cache | Valkey | Akamai LKE (CPU node) |
| 2 | Inference | vLLM with prefix caching | Akamai LKE (GPU node) |
| 3 | Inference | Qwen-Image serving | Akamai LKE (GPU node) |
| 4 | Benchmarking | Throughput + cost harness | Akamai LKE (GPU nodes) |

## Computation reuse points

Each phase targets a different level of the stack:

1. **KV cache** (Phase 1) — reuse attention key/value tensors within a
   single model forward pass for autoregressive decoding.

2. **Prefix cache** (Phase 2) — reuse computed KV states across requests
   that share a common prompt prefix (system prompts, few-shot examples).

3. **Request deduplication** (Phase 2, Fermyon layer) — short-circuit
   semantically identical requests at the edge before they reach the GPU.

4. **Multi-GPU scheduling** (Phase 4) — route requests to the right GPU
   tier (throughput-optimized vs latency-optimized) to minimise idle compute.

## Akamai LKE cluster topology

> TODO: Fill in once cluster is provisioned. Document node pool sizes,
> GPU counts, and network topology here.

```
PLACEHOLDER — cluster topology diagram goes here
```
