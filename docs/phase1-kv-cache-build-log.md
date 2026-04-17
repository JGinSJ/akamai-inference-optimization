# Phase 1 KV Cache Build Log

*2026-04-16*

---

## 1. What was originally intended

Phase 1 was conceived as a standalone educational demo: a from-scratch PyTorch implementation of the key-value cache mechanism inside a decoder-only transformer, written with no dependencies beyond PyTorch itself. The intent was to make the KV cache visible and measurable — showing exactly where computation is saved, how the cache grows, and that cached and no-cache generation produce identical outputs.

The project sits below the infrastructure stack. It does not deploy to a cluster, does not call an API, and does not use vLLM or HuggingFace. It exists to establish the foundational concept that the rest of the project builds on: attention K/V tensors for past positions are fixed once computed, so they can be stored and reused rather than recomputed at every decode step.

Originally planned deliverables:

- `attention.py` — `MultiHeadAttention` with an explicit, appendable KV cache tuple
- `kv_cache.py` — a standalone cache management module
- `model.py` — a minimal decoder-only transformer stack
- `benchmark.py` — timing harness comparing cached vs no-cache generation
- Unit tests covering correctness, shapes, and causal masking
- An LKE Kubernetes Job manifest to run the benchmark on GPU hardware

---

## 2. What was actually built

### Module layout

The implementation lives under `phases/phase1-kv-cache/kv_cache/` as a Python package:

```
kv_cache/
├── __init__.py      # re-exports DecoderTransformer, CharTokenizer, generate
├── attention.py     # MultiHeadAttention with KV cache
├── model.py         # TransformerBlock + DecoderTransformer
├── tokenizer.py     # CharTokenizer (no external deps)
└── generate.py      # prefill + decode loop, cached and no-cache modes
```

A top-level `demo.py` drives both generation modes on the same model and prompt, compares timing, verifies output correctness, and writes results to `results/phase1_timing.json`.

### attention.py

`MultiHeadAttention` accepts an optional `kv_cache: Optional[Tuple[Tensor, Tensor]]`. Cache tensors have shape `[batch, seq_len, n_heads, d_head]`. On each forward call:

- If `kv_cache` is `None` (prefill or full-recompute): computes fresh K and V for all input tokens, applies a causal mask when `seq_len > 1`, returns output and a new `(K, V)` cache.
- If `kv_cache` is provided (cached decode): concatenates the cached K/V tensors with the new single-token K/V along the sequence dimension, then attends. No causal mask is applied when `seq_len == 1` — a single query is always the latest position and correctly attends to all prior keys without masking.

The causal mask is built position-by-position using `q_idx <= past_len + k_idx` rather than a fixed upper-triangular matrix, which handles the case where new queries attend into a mixed past-and-current key range.

### model.py

`TransformerBlock` wraps `MultiHeadAttention` with a two-sublayer pre-norm residual structure: layer norm → attention, then layer norm → feed-forward (Linear → GELU → Linear). Each block returns its updated `(K, V)` cache alongside the output tensor.

`DecoderTransformer` adds token and positional embeddings, stacks N `TransformerBlock`s, applies a final layer norm, and projects to vocabulary logits. The positional offset is computed from the existing cache length so that cached decode steps continue numbering positions from where prefill left off — not from 0.

The model accepts `kv_caches: Optional[List[Optional[KVCache]]]`. Passing `None` runs a full prefill or recompute pass; passing the list from a prior step runs a single-token cached decode.

### generate.py

`generate()` implements both modes in one function, selected by `use_cache: bool`:

**With cache:** one forward pass over the full prompt (prefill), then one forward pass per new token, each receiving only the single new token ID and the accumulated cache list. The first generated token comes from the last position of the prefill output; all subsequent ones come from position 0 of the single-token decode output.

**Without cache:** at every step the full growing sequence is fed through `model()` with `kv_caches=None`. This recomputes all K/V tensors from scratch at each step.

Both paths return a dict including per-step timing (`time.perf_counter()`), the generated token ID list, and the final cache size in bytes.

### demo.py / benchmark

`demo.py` is the combined benchmark and correctness check. It runs both generation modes on the same randomly-seeded model (same weights, same prompt), measures per-step decode times, prints a comparison table, and asserts that both modes produce identical token sequences. If the sequences differ, it exits with code 1.

Results from the committed `results/phase1_timing.json` (CPU, Python 3.11.15, PyTorch 2.2.2, Intel Mac):

- Prompt: 129 tokens, generating 64 new tokens
- With cache — prefill: 10.6 ms; decode avg: 1.89 ms/step; cache at end: 1.5 MB
- Without cache — decode avg: 6.98 ms/step
- Measured speedup: **3.69×**
- Output correctness: PASS

These are real measured numbers on the dev machine. They are not PLACEHOLDER values. They represent CPU timing on a small toy model (d_model=256, 4 layers, 4 heads) — the intent is to show the O(n²) vs O(n) difference, not to benchmark inference throughput.

### Tests

`tests/test_attention.py` — 10 tests across 4 classes:

`TestCacheCorrectness` (3 tests): logit-level correctness at every decode step using `torch.allclose(atol=1e-5)`, not just argmax token IDs; end-to-end token sequence equality; exact output length for varying `max_new_tokens`.

`TestKVCacheShape` (3 tests): K/V tensor shape after prefill; cache grows by exactly 1 per decode step; full model cache covers exactly `len(prompt)` positions after prefill.

`TestCausalMask` (1 test): two sequences differing only at position `k` — output at positions `0..k-1` must be identical in both, verifying that future tokens do not leak into earlier attention positions.

`TestTokenizer` (3 tests): encode/decode roundtrip; vocab size is exactly 98; BOS/EOS/PAD token IDs do not produce output characters when decoded.

### LKE job manifest

No Kubernetes manifest was written for Phase 1. There is no `infrastructure/kubernetes/` directory. This was a deliberate scoping decision — see section 3.

---

## 3. Decisions made and why

### The `kv_cache.py` module became a package

**Originally assumed:** a single file `kv_cache.py` containing all cache logic.

**Discovered:** splitting into separate modules (`attention.py`, `model.py`, `tokenizer.py`, `generate.py`) made the code considerably easier to follow and test in isolation — each file has one clear responsibility. The planned `kv_cache.py` became `kv_cache/` (a package), with `kv_cache/__init__.py` re-exporting the three public symbols (`DecoderTransformer`, `CharTokenizer`, `generate`). The public API from `demo.py`'s perspective is unchanged: `from kv_cache import DecoderTransformer, CharTokenizer, generate`.

### `benchmark.py` became `demo.py`

**Originally assumed:** a separate `benchmark.py` file focused on timing.

**Outcome:** the benchmark and the demo are the same thing. Separating them would mean running the model twice and splitting the output/JSON writing logic across two files with no gain. `demo.py` covers both: it runs both generation modes, prints timing, checks correctness, and persists results to JSON. No deviation in functionality — just a naming difference.

### PyTorch version constraint on Intel Mac

**Originally assumed:** `torch>=2.4` could be used on the dev machine.

**Discovered:** PyTorch 2.4+ does not ship a wheel for Intel Mac (x86_64 darwin). The maximum installable version on this machine is PyTorch 2.2.2. `requirements.txt` therefore pins `torch>=2.1` rather than `>=2.4`, and the implementation avoids any API introduced after 2.1. The CLAUDE.md session note records this as: "torch>=2.4 required but unavailable on Intel Mac — local dev limited to infrastructure tests only."

In practice, Phase 1 uses only stable pre-2.4 PyTorch primitives (`nn.Module`, `torch.matmul`, `torch.softmax`, `torch.cat`, `torch.no_grad`) and is unaffected by this constraint. The 2.4+ requirement exists only for Phase 3 (Qwen-Image with `transformers` 4.x), which requires `torch>=2.4` and cannot run locally.

### Benchmark numbers are real, not PLACEHOLDER

**Originally assumed:** timing results from the dev machine would need to be labeled PLACEHOLDER since the "real" numbers would come from GPU hardware on LKE.

**Outcome:** the CPU timing numbers in `results/phase1_timing.json` are real measured values and are committed. They are not labeled PLACEHOLDER because Phase 1 is a CPU demo by design — the point is to show the compute scaling behaviour, not to produce production throughput numbers. The 3.69× speedup at 129 prompt tokens + 64 decode steps on a d_model=256 toy model is a valid illustration of the O(n²) vs O(n) difference, even on CPU.

### LKE job manifest: not written

**Originally assumed:** a Kubernetes Job manifest would be produced so the Phase 1 benchmark could run on the cluster's GPU node.

**Decided:** not written, for two reasons:

1. Phase 1 has no dependency on GPU hardware. The model is intentionally small and runs cleanly on CPU in ~2 seconds. Running it on a GPU node adds deployment complexity without meaningfully improving the demonstration.

2. Phase 1 is an educational building block, not a deployed service. Phases 2, 3, and 4 are where the cluster manifests live. The architecture doc (`docs/architecture.md`) explicitly classifies Phase 1 as "Developer laptop / CI" in its phase map.

If the benchmark were ever to be run at larger scale (larger model, longer sequences, batch sizes > 1) to produce GPU timing numbers for the cost model, a Job manifest would be straightforward to write — it would be a single-container Job with `gpu-type=rtx4000ada` nodeSelector, `resources.limits.nvidia.com/gpu: 1`, and the Phase 1 package installed via the `requirements.txt`.

### Test design: logit-level vs token-level correctness

**Considered:** testing only that the final generated token sequences are identical (token-level equality).

**Decided:** the gold-standard test (`test_logits_match_no_cache_step_by_step`) compares raw logit tensors at every decode step using `torch.allclose(atol=1e-5)`. This catches bugs where the cache returns slightly wrong attention values that still happen to argmax to the same token — a silent correctness failure that a token-level equality check would miss. The logit-level test is kept alongside the token-level test, not replacing it.

### No `benchmark.py` / no `kv_cache.py` in top-level layout

No deviation in intent — both the benchmark harness and the cache implementation exist and work. The file naming follows what emerged naturally from the split-into-package decision.

---

## 4. Current state

### Test results

```
10 passed in 2.04s
```

All 10 tests pass on the dev machine (Python 3.11.15, PyTorch 2.2.2, Intel Mac, CPU only). No skipped tests, no failures.

```bash
cd phases/phase1-kv-cache
python -m pytest tests/ -v
```

### Dev machine behaviour

`python demo.py` runs cleanly in approximately 2 seconds on CPU. The most recently committed timing run produced a 3.69× decode speedup with outputs matching between cached and no-cache modes.

### Deployment status

Phase 1 has never been deployed to LKE and there is no Kubernetes manifest for it. Deploying it would require:

1. Write a Kubernetes Job manifest targeting the GPU node pool (`gpu-type=rtx4000ada` nodeSelector, or CPU node with `workload-type=cpu`).
2. Package the phase into a container image with Python 3.11 and PyTorch installed.
3. Mount or bake in the `phases/phase1-kv-cache/` directory.
4. Set `command: ["python", "demo.py"]`.

This has not been done because the demo runs in under 2 seconds on CPU and there is no operational need to run it on the cluster. GPU hardware is reserved for vLLM (Phase 2) and Qwen-Image (Phase 3).

### Relationship to the rest of the project

Phase 1 is the conceptual foundation. It demonstrates, at the level of tensor operations, why KV caching reduces the per-token compute cost from O(n²) to O(n). Phases 2 and 3 implement the same idea at infrastructure level:

- Phase 2 (Fermyon + Valkey + vLLM) caches complete responses for identical prompts across requests, using LMCache to offload vLLM's internal KV states to Valkey. The Fermyon proxy short-circuits identical chat completion requests before they reach the GPU.
- Phase 3 (Qwen-Image) benefits from vLLM's internal prefix caching on the GPU node, which is the production realization of the exact mechanism Phase 1 demonstrates from scratch.

Phase 1 is intentionally self-contained. It imports nothing from Phases 2–4 and has no runtime dependency on the cluster. It can be read, run, and understood independently of the rest of the project as a standalone illustration of the KV cache concept.
