# Phase 4 — Multi-GPU Benchmarking and Cost Model

## Goal

Produce a reproducible head-to-head throughput and cost comparison between
RTX 4000 Ada and RTX PRO 6000 Blackwell on Akamai Cloud, and build a
cost model that lets operators estimate the right GPU tier for their workload.

## Inputs and outputs

| | Detail |
|---|---|
| Input | Benchmark configuration (model, batch sizes, concurrency levels) |
| Output | Throughput (tokens/sec), latency (P50/P95/P99), GPU utilisation |
| Output | Cost-per-request and cost-per-million-tokens for each GPU tier |
| Output | Markdown report and raw JSON results |

## Key technologies

- Python 3.11+ benchmarking harness
- **RTX 4000 Ada** on Akamai LKE GPU node pool
- **RTX PRO 6000 Blackwell** on Akamai LKE GPU node pool
- PyTorch and/or vLLM as inference backend (reused from Phase 1/2)
- Akamai Cloud pricing API or published price list for cost calculations

## What the benchmark measures

1. **Throughput** — tokens generated per second at batch sizes 1, 4, 16, 64.
2. **Latency** — time-to-first-token (TTFT) and inter-token latency (ITL)
   at each batch size and concurrency level.
3. **Memory utilisation** — GPU VRAM used at peak load.
4. **Cost per token** — compute cost (Akamai GPU node price) divided by
   tokens generated per hour.

## File layout (target)

```
phases/phase4-benchmarks/
├── README.md
├── requirements.txt
├── harness/
│   ├── run_benchmark.py      # Main benchmark runner
│   ├── metrics.py            # Throughput, latency, memory collection
│   └── cost_model.py         # GPU cost -> cost-per-token calculation
├── configs/
│   ├── rtx4000ada.yaml       # Benchmark config for RTX 4000 Ada
│   └── rtxpro6000.yaml       # Benchmark config for RTX PRO 6000 Blackwell
├── k8s/
│   ├── benchmark-job-ada.yaml
│   └── benchmark-job-blackwell.yaml
├── results/                  # gitignored — generated outputs go here
└── report/
    └── generate_report.py    # Read results/, write REPORT.md
```

## Success criteria

- [ ] Benchmark runs end-to-end on both GPU targets without error.
- [ ] Results are deterministic (same config -> same output within noise).
- [ ] Cost model uses only published Akamai pricing (no fabricated numbers).
- [ ] `generate_report.py` produces a valid Markdown report from raw results.
- [ ] Report clearly labels any PLACEHOLDER values not yet filled in.

## Open questions

> TODO: Confirm Akamai Cloud GPU node pricing source and whether it is
> available via API or must be read from a published price list.

> TODO: Decide cost model methodology — per-token, per-request, or
> per-GPU-hour (see CLAUDE.md open questions log, item 5).

> TODO: Confirm whether both GPU types are available in the same LKE
> region, or whether cross-region testing is required.
