# Phase 1 — KV Cache from Scratch

## Goal

Build a minimal PyTorch transformer that makes the key-value cache
visible, measurable, and understandable — without HuggingFace or other
inference framework dependencies.

The purpose is not to build a production inference engine. It is to
demonstrate exactly what is being reused and what is being recomputed
on each autoregressive forward pass, so that the savings from Phases 2–4
can be clearly motivated.

## Inputs and outputs

| | Detail |
|---|---|
| Input | A short prompt string (e.g. 64 tokens) |
| Output | Generated continuation (e.g. 64 tokens) |
| Side output | Per-step timing breakdown: prefill vs decode latency |
| Side output | KV cache memory footprint as a function of sequence length |

## Key technologies

- Python 3.11+
- PyTorch (CPU or CUDA, no framework requirement)
- No HuggingFace `transformers` or `accelerate`
- No vLLM

## What the demo shows

1. **Prefill pass** — the full prompt is processed once; KV states are
   stored for every layer.
2. **Decode loop** — each new token attends over its own Q against the
   cached K and V from all prior positions. No recomputation.
3. **Ablation** — the same model run *without* the cache (full recompute
   at every step) to make the speedup measurable.
4. **Memory plot** — KV cache size (bytes) vs sequence length, per layer.

## File layout

```
phases/phase1-kv-cache/
├── README.md
├── requirements.txt
├── demo.py               # Runnable end-to-end demo script
├── kv_cache/
│   ├── __init__.py
│   ├── tokenizer.py      # Character-level tokenizer, vocab size 98
│   ├── model.py          # TransformerBlock + DecoderTransformer
│   ├── attention.py      # MultiHeadAttention with explicit KV cache
│   └── generate.py       # Prefill + decode loop, cached and no-cache modes
└── tests/
    ├── __init__.py
    └── test_attention.py # Logit-level and token-level correctness tests
```

## Success criteria

- [ ] `python demo.py` runs to completion on CPU without errors.
- [ ] Output with cache and output without cache are identical (correctness).
- [ ] Per-step timing shows sublinear growth with cache enabled.
- [ ] Tests pass: `python -m pytest phases/phase1-kv-cache/tests/`.

## Decisions

| Decision | Resolution |
|---|---|
| Model size | 4 layers, 4 heads, d_model=256, d_ff=1024 — runs on CPU in CI, shows meaningful timing difference |
| Tokenizer | Character-level, 95 printable ASCII + PAD/BOS/EOS = 98 tokens, no external dependencies |
