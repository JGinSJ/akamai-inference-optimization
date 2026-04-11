# Phase 1 — KV Cache from Scratch

A minimal PyTorch decoder-only transformer that makes the key-value cache
visible and measurable. No HuggingFace, no vLLM — just PyTorch and the
math.

## Why KV caching reduces compute

To generate one token, a transformer must compute attention over every
prior position. Without a cache, generating a sequence of length `n`
requires recomputing the full attention matrix at every step:

```
Step 1: attend over 1 position   → O(1)   work
Step 2: attend over 2 positions  → O(2)   work
Step 3: attend over 3 positions  → O(3)   work
...
Step n: attend over n positions  → O(n)   work
Total: O(1 + 2 + ... + n) = O(n²) attention operations
```

The key observation: the key (K) and value (V) vectors for every past
token are **fixed** once that token has been processed. They depend only
on the token's embedding and the model weights — both of which are
constant. There is no reason to recompute them.

The KV cache stores each layer's K and V tensors after they are first
computed. On subsequent steps, new K and V vectors are computed only
for the single new token, then concatenated onto the cache:

```
Step 1 (prefill): compute K, V for all prompt tokens. Store in cache.
Step 2 (decode):  compute K, V for 1 new token. Append to cache.
                  Attention query is 1 vector; keys are n+1 entries.
Step 3 (decode):  same — 1 new K/V pair, cache grows by 1.
...
```

This reduces the work per decode step from `O(n)` to `O(1)` in terms
of matrix multiplications performed on new input, at the cost of
`O(n · d_model)` memory to hold the cache.

**The trade-off in one line:** KV caching trades memory for compute.

### Where the savings come from

In the attention equation:

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d ) V
```

Without cache: Q, K, and V are all recomputed from the full sequence
at every step. The dominant cost is the `Q Kᵀ` matmul, which is
`O(seq_len²)` per layer.

With cache: only Q for the new token is computed fresh. K and V are
retrieved from the cache (a memory read, not a matmul). The `Q Kᵀ`
cost drops to `O(seq_len)` per decode step — one query dot-producted
against each cached key.

### Memory cost

A KV cache for one layer holds two tensors of shape
`[seq_len, n_heads, d_head]`. For the model in this demo
(4 layers, 4 heads, d_head=64, float32) and a 200-token sequence:

```
4 layers × 2 tensors × 200 tokens × 4 heads × 64 dims × 4 bytes
= 1,638,400 bytes ≈ 1.6 MB
```

For production models (e.g. 32 layers, 32 heads, d_head=128, batch>1)
this grows quickly, which is why prefix caching (Phase 2) and
careful cache eviction are important at scale.

---

## What this demo shows

| Concept | Where to look |
|---|---|
| KV cache data structure (K, V per layer) | `kv_cache/attention.py` |
| Prefill vs decode separation | `kv_cache/generate.py` |
| Full-recompute ablation | `kv_cache/generate.py` — `use_cache=False` |
| Timing comparison | `demo.py` output / `results/phase1_timing.json` |

## Model configuration

| Hyperparameter | Value |
|---|---|
| Layers | 4 |
| Attention heads | 4 |
| d_model | 256 |
| d_ff | 1024 |
| Tokenizer | Character-level (vocab size 98) |
| Max sequence length | 512 |

Weights are randomly initialised. The demo measures compute cost,
not generation quality.

## Setup

```bash
cd phases/phase1-kv-cache
pip install -r requirements.txt
```

Requires Python 3.11+ and PyTorch 2.1+. Runs on CPU; uses CUDA automatically
if available.

## Run the demo

```bash
python demo.py
```

Expected output (timings will vary by hardware):

```
=== Phase 1: KV Cache from Scratch ===
Device  : cpu
Model   : 4 layers, 4 heads, d_model=256, d_ff=1024
Prompt  : 138 tokens | Generating: 64 new tokens

--- WITH KV CACHE ---
  Prefill (138 tokens)   :   XX.XX ms
  Decode steps 1–63 avg  :    X.XX ms  [min: X.XX  max: X.XX]
  Cache memory at end    :    X.XX MB  (seq_len=202)

--- WITHOUT KV CACHE (full recompute) ---
  Decode steps 1–64 avg  :   XX.XX ms  [min: X.XX  max: X.XX]

--- COMPARISON ---
  Avg decode speedup (no-cache / cached) : X.Xx
  Output correctness                     : PASS
```

JSON results are written to `results/phase1_timing.json` (gitignored).

## Run the tests

```bash
python -m pytest tests/ -v
```

Tests cover:
- Logit-level correctness: cached and no-cache forward passes produce
  identical logit tensors at every decode step (`torch.allclose`, atol=1e-5)
- Token-level correctness: cached and no-cache produce identical generated sequences
- KV cache shape after prefill and after N decode steps
- Causal mask: position i output is unaffected by tokens at positions > i
- Tokenizer encode/decode roundtrip

## File layout

```
phase1-kv-cache/
├── kv_cache/
│   ├── __init__.py
│   ├── tokenizer.py   # Character-level tokenizer, vocab size 98
│   ├── attention.py   # MultiHeadAttention with explicit KV cache
│   ├── model.py       # TransformerBlock + DecoderTransformer
│   └── generate.py    # Prefill + decode loop, cached and no-cache modes
├── tests/
│   └── test_attention.py
├── demo.py
├── requirements.txt
└── README.md
```
