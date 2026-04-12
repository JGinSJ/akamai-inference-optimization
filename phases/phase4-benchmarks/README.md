# Phase 4 — Multi-GPU Benchmarking and Cost Model

Head-to-head throughput and cost comparison of RTX 4000 Ada vs RTX PRO 6000
Blackwell on Akamai LKE, with a vLLM tensor-parallel deployment path and a
cost model that converts GPU-hour pricing to cost-per-token.

---

## File layout

```
phase4-benchmarks/
├── harness/
│   ├── run_benchmark.py      # CLI runner — drives load_gen, writes CSVs
│   ├── load_gen.py           # Async load generator (aiohttp, SSE streaming)
│   ├── metrics.py            # BenchmarkSummary, TTFT/ITL/E2E percentiles
│   └── cost_model.py         # cost/token, cost/req, cost/M-tokens → CSV
├── configs/
│   ├── rtx4000ada.yaml       # RTX 4000 Ada benchmark config (PLACEHOLDER pricing)
│   └── rtxpro6000.yaml       # RTX PRO 6000 Blackwell config (PLACEHOLDER pricing)
├── k8s/
│   ├── vllm-tp.yaml          # vLLM Deployment with --tensor-parallel-size 2
│   ├── service.yaml          # ClusterIP on 8000
│   ├── benchmark-job-ada.yaml
│   └── benchmark-job-blackwell.yaml
├── report/
│   └── generate_report.py    # Reads results/*.json, writes REPORT.md
├── tests/
│   └── test_cost_model.py    # Unit tests (no GPU required)
└── results/                  # gitignored — generated outputs go here
```

---

## Setup

```bash
cd phases/phase4-benchmarks
pip install -r requirements.txt
```

No GPU is required for the tests or the report generator.
A running vLLM endpoint is required for `load_gen.py` and `run_benchmark.py`.

---

## Run the tests

```bash
python -m pytest tests/ -v
```

All tests pass without a GPU, network connection, or vLLM instance.

---

## Run the benchmark

### Prerequisites

1. A running vLLM instance — see [Deploy to LKE](#deploy-to-lke) or start
   locally:

   ```bash
   pip install vllm
   vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
       --tensor-parallel-size 1 \
       --enable-prefix-caching \
       --port 8000
   ```

2. Fill in `gpu_hourly_usd` in the config file you intend to run.  The cost
   model will raise an error if the value is still `PLACEHOLDER`.

### Single-GPU run

```bash
python -m harness.run_benchmark --config configs/rtx4000ada.yaml
```

Outputs written to `results/`:

```
results/ada-b32-c1.json
results/ada-b32-c4.json
...
results/comparison.csv
results/cost_by_batch.csv
results/cost_by_concurrency.csv
```

### Generate the report

```bash
python report/generate_report.py \
    --results-dir results/ \
    --output results/REPORT.md
```

---

## Tensor parallelism vs data parallelism

This is the central trade-off for multi-GPU deployment.  The right answer
depends on model size, VRAM, and whether you are optimising for latency or
throughput.

### What each strategy does

**Tensor parallelism (TP)** splits one model's weight matrices across N GPUs
on the same node.  Each GPU holds 1/N of every layer.  Forward passes require
an all-reduce collective after each TP layer.  vLLM flag:
`--tensor-parallel-size N`.

**Data parallelism (DP)** runs N independent full-model replicas, one per GPU.
Each replica handles different requests.  No inter-GPU communication during
inference; a load balancer distributes requests.

### When to prefer tensor parallelism

**1. Model does not fit in one GPU.**

The most common reason to use TP.  Qwen2.5-VL-7B at float16 needs ~16 GB of
VRAM just for weights.  Add KV cache overhead and the RTX 4000 Ada may not
have enough headroom for a single replica at useful batch sizes.  TP-2 cuts
the per-GPU model footprint in half.

**2. You need the lowest possible time-to-first-token.**

TP parallelises the prefill computation across GPUs.  For long prompts (512+
tokens), the prefill dominates TTFT.  Splitting it across two GPUs roughly
halves the prefill time, minus the all-reduce overhead.  The all-reduce over
NVLink is typically < 1 ms and is negligible relative to prefill savings for
large models.

**3. You are running a very large model (72B+).**

At 72B parameters, TP is the only option on any single-node configuration
with fewer than ~8 GPUs.  TP is most compute-efficient when the all-reduce
cost is small relative to the matrix multiply time, which is true for large
hidden dimensions and long sequences.

### When to prefer data parallelism

**1. Model fits in a single GPU with VRAM headroom.**

If Qwen2.5-VL-7B fits in one RTX PRO 6000 Blackwell (PLACEHOLDER GB VRAM)
with room for a useful KV cache, DP is simpler and more efficient.  Two DP
replicas serve exactly twice as many concurrent requests as one TP-2
deployment, with no all-reduce latency tax.

**2. Throughput matters more than per-request latency.**

DP scales throughput linearly with GPU count.  TP does not — the all-reduce
becomes a bottleneck at high concurrency.  At batch sizes where the GPU is
saturated, DP typically achieves higher aggregate tokens/sec.

**3. You want operational simplicity.**

DP replicas are independent Kubernetes pods.  A failed replica does not affect
the others.  Rolling updates redeploy one replica at a time with no downtime.
TP requires all N GPUs to be on the same physical node; if one GPU fails the
entire replica is lost.

**4. You need multi-node scaling.**

Standard vLLM TP is limited to GPUs on the same node.  Scaling to more nodes
requires pipeline parallelism (or moving to a different serving framework).
DP scales across nodes without any special configuration.

### Overhead to expect from TP all-reduce

At TP-2, each transformer layer adds one all-reduce over `hidden_dim` elements.
For Qwen2.5-VL-7B (hidden_dim ≈ 3584, float16, 28 layers):

- **NVLink** (RTX PRO 6000 Blackwell — confirm availability):
  ~14 KB per layer × 28 layers = ~392 KB per forward pass.
  At NVLink bandwidth of PLACEHOLDER GB/s → PLACEHOLDER µs total.
  Typically negligible at batch size ≥ 4.

- **PCIe** (if NVLink is unavailable):
  PCIe 4.0 ×16 provides ~64 GB/s; the same 392 KB takes ~6 µs.
  At batch size 1 this is visible in ITL; at larger batches it is amortised.

> PLACEHOLDER: Replace bandwidth figures once hardware.md is filled in.
> Measure actual ITL at TP-1 vs TP-2 with the benchmark harness.

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
Run `run_benchmark.py` with `tensor-parallel-size 1` and `2`, compare the
TTFT, ITL, and tokens/sec columns in REPORT.md, then decide.

---

## Deploy to LKE

### vLLM tensor-parallel deployment

```bash
# Create namespace
kubectl create namespace inference

# Deploy vLLM (edit k8s/vllm-tp.yaml first — set image, nodeSelector)
kubectl apply -f k8s/vllm-tp.yaml
kubectl apply -f k8s/service.yaml

kubectl rollout status deployment/vllm-tp2 -n inference
kubectl logs -n inference -l app=vllm-tp2 -f
```

The vLLM deployment uses `--tensor-parallel-size 2` and requests 2 GPUs from
the node.  The node must have at least 2 GPUs.  Edit `k8s/vllm-tp.yaml` to
change `--tensor-parallel-size` if running on a single-GPU node.

### Benchmark Jobs

```bash
# Fill in gpu_hourly_usd in the ConfigMap before applying.
# Edit k8s/benchmark-job-ada.yaml and k8s/benchmark-job-blackwell.yaml.

kubectl apply -f k8s/benchmark-job-ada.yaml
kubectl wait --for=condition=complete job/benchmark-ada -n inference --timeout=30m
kubectl logs -n inference job/benchmark-ada

# Extract results from the completed pod
kubectl cp inference/$(kubectl get pod -n inference -l job-name=benchmark-ada \
    -o jsonpath='{.items[0].metadata.name}'):/results ./results/ada/
```

Repeat for `benchmark-job-blackwell.yaml`, then run `generate_report.py`.

---

## Cost model

The cost model converts GPU-hour pricing to per-token, per-request, and
per-million-token costs.  It does not hardcode any prices.

Fill in `gpu_hourly_usd` in the config file, then run:

```bash
python -m harness.run_benchmark --config configs/rtx4000ada.yaml
# CSVs written to results/
```

Or call the cost model directly on existing summaries:

```python
from harness.cost_model import compute_cost, write_csvs
# See harness/cost_model.py docstring for the full API.
```

---

## Success criteria

- [ ] `python -m pytest tests/ -v` passes (no GPU required).
- [ ] `run_benchmark.py` completes on both GPU targets without error.
- [ ] `generate_report.py` produces a valid REPORT.md from results.
- [ ] CSVs contain non-zero cost figures (requires `gpu_hourly_usd` set).
- [ ] REPORT.md latency table matches raw JSON files.
- [ ] No performance claims appear without a corresponding measured result.
- [ ] All PLACEHOLDER values in configs and K8s manifests are replaced before
  production deployment.
