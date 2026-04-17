# Phase 4 — Single-GPU Inference Cost Benchmark

Measures raw decode throughput and per-token cost for a single RTX 4000 Ada
node on Akamai LKE, with a sweep across concurrency levels and prompt lengths.
The target deployment is `k8s/vllm-ada.yaml` — vanilla vLLM, no external cache,
no prefix reuse — so the numbers reflect the true cost of inference before any
caching optimisation is applied.

> **Note on scope:** This phase does not measure the Fermyon/Valkey prefix-cache
> hit rate or the latency reduction it produces.  That is a Phase 2 measurement.
> Mixing cache effects into Phase 4 would contaminate the per-token cost
> baseline.  Run Phase 2 and Phase 4 independently and compare the results.

---

## File layout

```
phase4-benchmarks/
├── benchmark/
│   └── benchmark.py          # Synchronous sweep script — one CSV row per run
├── harness/
│   ├── metrics.py            # BenchmarkSummary, E2E percentiles
│   ├── cost_model.py         # cost/token, cost/req, cost/M-tokens → CSV
│   ├── load_gen.py           # Async load generator (retained; not used in sweep)
│   └── run_benchmark.py      # Async runner (retained; not used in sweep)
├── configs/
│   ├── rtx4000ada.yaml       # RTX 4000 Ada config — gpu_hourly_usd: 0.96
│   └── rtxpro6000.yaml       # RTX PRO 6000 Blackwell config (PLACEHOLDER pricing)
├── k8s/
│   ├── vllm-ada.yaml         # Single-GPU Ada deployment (Phase 4 target)
│   ├── vllm-tp.yaml          # TP-2 Blackwell deployment (future work — see appendix)
│   ├── service.yaml          # ClusterIP on port 8000
│   ├── benchmark-job-ada.yaml
│   └── benchmark-job-blackwell.yaml
├── tests/
│   ├── test_benchmark.py     # Unit tests for benchmark.py (no GPU required)
│   └── test_cost_model.py    # Unit tests for harness (no GPU required)
└── results/                  # gitignored — generated CSV outputs land here
```

---

## Setup

```bash
cd phases/phase4-benchmarks
pip install -r requirements.txt
```

No GPU is required for the tests.
A running vLLM endpoint is required for `benchmark/benchmark.py`.

---

## Run the tests

```bash
python -m pytest tests/ -v
```

All 57 tests pass without a GPU, network connection, or vLLM instance.

---

## Deploy to LKE

```bash
# Apply the Ada deployment (single GPU, no prefix caching)
kubectl apply -f k8s/vllm-ada.yaml

kubectl rollout status deployment/vllm-ada -n inference
kubectl logs -n inference -l app=vllm-ada -f
```

The deployment uses `mistralai/Mistral-7B-Instruct-v0.2` with
`--no-enable-prefix-caching` and `--max-model-len 25664` — the ceiling
measured on the RTX 4000 Ada (20 GB VRAM).

Port-forward for local benchmarking:

```bash
kubectl port-forward -n inference svc/vllm-ada 8000:8000
```

---

## Run the benchmark

### Single run

```bash
python benchmark/benchmark.py \
    --url http://localhost:8000/v1/completions \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --prompt-tokens 256 --max-tokens 128 \
    --concurrency 4 --num-requests 40 \
    --gpu-hourly-usd 0.96 --gpu-label "RTX 4000 Ada" \
    --output-csv results/ada_sweep.csv \
    --tag ada-c4-p256
```

### Concurrency sweep (shell loop)

```bash
for c in 1 4 8 16; do
    python benchmark/benchmark.py \
        --url http://localhost:8000/v1/completions \
        --model mistralai/Mistral-7B-Instruct-v0.2 \
        --prompt-tokens 256 --max-tokens 128 \
        --concurrency $c --num-requests 40 \
        --gpu-hourly-usd 0.96 --gpu-label "RTX 4000 Ada" \
        --output-csv results/ada_sweep.csv \
        --tag ada-c${c}-p256
done
```

Each run appends one row to `results/ada_sweep.csv`.  Prior rows are never
overwritten, so you can re-run individual concurrency levels without losing
other results.

---

## Results — RTX 4000 Ada

### Throughput and latency by concurrency (prompt≈256 tok, max_tokens=64)

| Concurrency | tok/s | req/s | e2e p50 (ms) | e2e p95 (ms) | e2e p99 (ms) |
|:-----------:|------:|------:|-------------:|-------------:|-------------:|
| 1           | 20.9  | 0.33  | 3069         | 3135         | 3146         |
| 4           | 67.5  | 1.07  | 3094         | 3152         | 3189         |
| 8           | 92.4  | 1.53  | 3359         | 3360         | 3360         |
| 16          | 184.8 | 2.95  | 3328         | 3360         | 3380         |

### Cost by concurrency (gpu_hourly_usd = $0.96, prompt≈256 tok, max_tokens=64)

| Concurrency | cost/token (USD) | cost/request (USD) | cost/M tokens (USD) |
|:-----------:|-----------------:|-------------------:|--------------------:|
| 1           | 0.00001277       | 0.000817           | 12.77               |
| 4           | 0.00000395       | 0.000250           |  3.95               |
| 8           | 0.00000289       | 0.000174           |  2.89               |
| 16          | 0.00000144       | 0.000090           |  1.44               |

---

## Cost model

The cost model in `harness/cost_model.py` derives per-token cost from the GPU
hourly price and measured throughput:

```
cost_per_token = gpu_hourly_usd / 3600 / tokens_per_second
cost_per_request = cost_per_token × mean_tokens_generated
cost_per_million_tokens = cost_per_token × 1_000_000
```

No prices are hardcoded.  `gpu_hourly_usd` for the Ada node
(`g2-gpu-rtx4000a1-l`) is set to `0.96` in `configs/rtx4000ada.yaml`.
Update it if Akamai adjusts pricing.

---

## Success criteria

- [ ] `python -m pytest tests/ -v` passes (no GPU required).
- [ ] `benchmark/benchmark.py` completes on the Ada node without error.
- [ ] `results/ada_sweep.csv` contains non-zero cost figures at all four concurrency levels.
- [x] PLACEHOLDER result tables above are filled in with measured values.
- [ ] No performance claims appear without a corresponding row in the CSV.

---

## Appendix — future multi-GPU work

The k8s manifest `k8s/vllm-tp.yaml` is a tensor-parallel deployment targeting
the RTX PRO 6000 Blackwell node pool (two GPUs, `--tensor-parallel-size 2`).
It is not used in the Phase 4 Ada benchmark.  It is retained here for when
Blackwell nodes are available on the cluster.

### Tensor parallelism vs data parallelism

**Tensor parallelism (TP)** splits one model's weight matrices across N GPUs
on the same node.  Each GPU holds 1/N of every layer.  Forward passes require
an all-reduce collective after each TP layer.  vLLM flag:
`--tensor-parallel-size N`.

**Data parallelism (DP)** runs N independent full-model replicas, one per GPU.
Each replica handles different requests.  No inter-GPU communication during
inference; a load balancer distributes requests.

### When each strategy is appropriate

**Prefer TP when:**
- The model does not fit in one GPU's VRAM.
- You need the lowest possible time-to-first-token (TP parallelises prefill).
- Running a very large model (72B+) where TP is the only single-node option.

**Prefer DP when:**
- The model fits in one GPU with VRAM headroom for the KV cache.
- Throughput matters more than per-request latency (DP scales linearly).
- You want operational simplicity — DP replicas are independent pods.
- Multi-node scaling is required (standard vLLM TP is single-node only).

### TP all-reduce overhead (Qwen2.5-VL-7B, TP-2)

For Qwen2.5-VL-7B (hidden_dim ≈ 3584, float16, 28 layers), each TP-2 forward
pass adds one all-reduce over ~392 KB across all layers.

- **NVLink** (if available on Blackwell): typically < 1 ms — negligible at
  batch size ≥ 4.
- **PCIe 4.0 ×16**: ~6 µs for the same data — visible at batch size 1,
  amortised at larger batches.

> PLACEHOLDER: Confirm NVLink availability on the Blackwell node pool and
> replace bandwidth figures once `infrastructure/docs/hardware.md` is filled in.

### Decision table

| Condition | Recommended strategy |
|---|---|
| Model > single-GPU VRAM | TP |
| Model fits with KV cache headroom | DP |
| Minimise TTFT (latency-critical) | TP (measure first) |
| Maximise tok/s (throughput-critical) | DP |
| Two GPUs, NVLink available | TP-2 viable — measure vs DP |
| Two GPUs, PCIe only | DP preferred unless model doesn't fit |
| Multi-node scaling required | DP |
| Operational simplicity | DP |

**The only correct answer is the one supported by measured numbers.**
Run `benchmark.py` with both configurations, compare the tokens/sec and
e2e latency columns, then decide.
