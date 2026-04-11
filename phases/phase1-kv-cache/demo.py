"""
Phase 1 demo: KV Cache from Scratch

Runs autoregressive generation in two modes on the same randomly-initialised
model and prompt, then compares timing and verifies output correctness.

Usage
-----
    cd phases/phase1-kv-cache
    pip install -r requirements.txt
    python demo.py

Output
------
  - Timing table printed to stdout
  - results/phase1_timing.json written with raw numbers
"""

import json
import os
import sys

import torch

from kv_cache import CharTokenizer, DecoderTransformer, generate

# ---------------------------------------------------------------------------
# Model configuration — small enough for CPU, large enough to show O(n) growth
# ---------------------------------------------------------------------------
D_MODEL = 256
N_HEADS = 4
D_FF = 1024
N_LAYERS = 4
MAX_SEQ_LEN = 512
MAX_NEW_TOKENS = 64

PROMPT = (
    "Transformer models reuse cached key-value states during autoregressive "
    "decoding to avoid redundant computation on each new token."
)

SEED = 42


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:7.2f} ms"


def _cache_size_str(n_bytes: int) -> str:
    if n_bytes >= 1024 * 1024:
        return f"{n_bytes / 1024 / 1024:.2f} MB"
    return f"{n_bytes / 1024:.1f} KB"


def main() -> None:
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Phase 1: KV Cache from Scratch ===")
    print(f"Device  : {device}")
    print(f"Model   : {N_LAYERS} layers, {N_HEADS} heads, d_model={D_MODEL}, d_ff={D_FF}")

    tokenizer = CharTokenizer()
    model = DecoderTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        n_layers=N_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
    ).to(device)

    prompt_ids = tokenizer.encode(PROMPT)
    print(f"Prompt  : {len(prompt_ids)} tokens | Generating: {MAX_NEW_TOKENS} new tokens")
    print(f"Prompt text: \"{PROMPT[:60]}...\"")

    # ------------------------------------------------------------------
    # Run WITH cache
    # ------------------------------------------------------------------
    result_cached = generate(
        model, prompt_ids, max_new_tokens=MAX_NEW_TOKENS, use_cache=True
    )

    # ------------------------------------------------------------------
    # Run WITHOUT cache (same model, same prompt, same seed — weights unchanged)
    # ------------------------------------------------------------------
    result_nocache = generate(
        model, prompt_ids, max_new_tokens=MAX_NEW_TOKENS, use_cache=False
    )

    # ------------------------------------------------------------------
    # Correctness check
    # ------------------------------------------------------------------
    outputs_match = result_cached["generated_ids"] == result_nocache["generated_ids"]

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    cached_steps = result_cached["decode_step_times_s"]
    nocache_steps = result_nocache["decode_step_times_s"]
    cached_avg = sum(cached_steps) / len(cached_steps) if cached_steps else 0.0
    nocache_avg = sum(nocache_steps) / len(nocache_steps)
    speedup = nocache_avg / cached_avg if cached_avg > 0 else float("nan")

    print()
    print("--- WITH KV CACHE ---")
    print(f"  Prefill ({len(prompt_ids)} tokens)   : {_fmt_ms(result_cached['prefill_time_s'])}")
    if cached_steps:
        print(f"  Decode steps 1–{len(cached_steps)} avg  : {_fmt_ms(cached_avg)}"
              f"  [min: {_fmt_ms(min(cached_steps))}  max: {_fmt_ms(max(cached_steps))}]")
    seq_len_final = len(prompt_ids) + MAX_NEW_TOKENS
    print(f"  Cache memory at end       : {_cache_size_str(result_cached['cache_bytes_final'])}"
          f"  (seq_len={seq_len_final})")

    print()
    print("--- WITHOUT KV CACHE (full recompute) ---")
    print(f"  Decode steps 1–{len(nocache_steps)} avg  : {_fmt_ms(nocache_avg)}"
          f"  [min: {_fmt_ms(min(nocache_steps))}  max: {_fmt_ms(max(nocache_steps))}]")

    print()
    print("--- COMPARISON ---")
    print(f"  Avg decode speedup (no-cache / cached) : {speedup:.1f}x")
    match_label = "PASS" if outputs_match else "FAIL"
    print(f"  Output correctness                     : {match_label}"
          f"  (both modes produced {'identical' if outputs_match else 'DIFFERENT'} token sequences)")

    if not outputs_match:
        print("\nERROR: cached and no-cache outputs differ. Check attention/model logic.",
              file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Write JSON results
    # ------------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", "phase1_timing.json")
    payload = {
        "model_config": {
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "d_ff": D_FF,
            "n_layers": N_LAYERS,
            "vocab_size": tokenizer.vocab_size,
        },
        "prompt_len_tokens": len(prompt_ids),
        "max_new_tokens": MAX_NEW_TOKENS,
        "device": str(device),
        "seed": SEED,
        "with_cache": {
            "prefill_time_s": result_cached["prefill_time_s"],
            "decode_step_times_s": result_cached["decode_step_times_s"],
            "decode_avg_s": cached_avg,
            "cache_bytes_final": result_cached["cache_bytes_final"],
        },
        "without_cache": {
            "decode_step_times_s": result_nocache["decode_step_times_s"],
            "decode_avg_s": nocache_avg,
        },
        "speedup": speedup,
        "outputs_match": outputs_match,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults written to: {output_path}")


if __name__ == "__main__":
    main()
