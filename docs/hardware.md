# Hardware Targets

All benchmark claims in this project are measured on the hardware described
here. Do not substitute other GPUs and present results as equivalent.

> PLACEHOLDER: All specifications and benchmark figures in this file are
> stubs. Fill in from official NVIDIA data sheets and from measured results
> once hardware is available. Never copy unverified numbers from third-party
> sites.

## RTX 4000 Ada Generation

| Attribute | Value |
|-----------|-------|
| Architecture | Ada Lovelace |
| CUDA Cores | PLACEHOLDER |
| Tensor Cores | PLACEHOLDER |
| VRAM | PLACEHOLDER GB GDDR6 |
| Memory Bandwidth | PLACEHOLDER GB/s |
| TDP | PLACEHOLDER W |
| FP16 Tensor (peak) | PLACEHOLDER TFLOPS |
| INT8 Tensor (peak) | PLACEHOLDER TOPS |
| NVLink | PLACEHOLDER |
| Form factor | PLACEHOLDER |

### Notes

> PLACEHOLDER: Add notes about driver version, CUDA version, and any
> Akamai LKE-specific configuration required for this GPU.

---

## RTX PRO 6000 Blackwell

| Attribute | Value |
|-----------|-------|
| Architecture | Blackwell |
| CUDA Cores | PLACEHOLDER |
| Tensor Cores | PLACEHOLDER |
| VRAM | PLACEHOLDER GB GDDR7 |
| Memory Bandwidth | PLACEHOLDER GB/s |
| TDP | PLACEHOLDER W |
| FP16 Tensor (peak) | PLACEHOLDER TFLOPS |
| INT8 Tensor (peak) | PLACEHOLDER TOPS |
| FP8 Tensor (peak) | PLACEHOLDER TFLOPS |
| NVLink | PLACEHOLDER |
| Form factor | PLACEHOLDER |

### Notes

> PLACEHOLDER: Add notes about driver version, CUDA version, Blackwell-
> specific features used (e.g. FP8 quantization, transformer engine), and
> any Akamai LKE-specific configuration required for this GPU.

---

## Comparison summary

> PLACEHOLDER: Fill in once both GPUs are benchmarked in Phase 4.

| Metric | RTX 4000 Ada | RTX PRO 6000 Blackwell | Ratio |
|--------|-------------|------------------------|-------|
| Tokens/sec (FP16, batch 1) | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| Tokens/sec (FP16, batch 32) | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| Cost per 1M tokens (USD) | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| Time-to-first-token (ms) | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| Memory capacity (effective) | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |

---

## Driver and software versions

> PLACEHOLDER: Record exact versions used during benchmarking so results
> are reproducible.

| Software | Version |
|----------|---------|
| NVIDIA Driver | PLACEHOLDER |
| CUDA | PLACEHOLDER |
| PyTorch | PLACEHOLDER |
| vLLM | PLACEHOLDER |
| OS / kernel | PLACEHOLDER |
