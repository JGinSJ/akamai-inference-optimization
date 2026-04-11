"""
Autoregressive generation with and without KV cache.

Both modes produce *identical* token sequences for the same model and prompt —
the only difference is how much computation is performed at each decode step.

With cache
----------
  Prefill  : one forward pass over the full prompt (O(n²) attention).
             KV states for all prompt positions are stored.
  Decode   : one forward pass per new token, attending over 1 query
             against a growing K/V cache — O(n) per step, O(1) if n is
             treated as the cached sequence length.

Without cache
-------------
  Each decode step re-runs the full forward pass over the entire
  sequence so far — O(n²) attention growing at every step.

Return value
------------
A dict with:
  use_cache            : bool
  prompt_ids           : List[int]
  generated_ids        : List[int]   (length == max_new_tokens)
  prefill_time_s       : float | None  (None in no-cache mode)
  decode_step_times_s  : List[float]
  cache_bytes_final    : int  (0 in no-cache mode)
"""

import time
from typing import Dict, List, Optional

import torch

from .model import DecoderTransformer


def generate(
    model: DecoderTransformer,
    prompt_ids: List[int],
    max_new_tokens: int,
    use_cache: bool,
) -> Dict:
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        generated_ids: List[int] = []
        decode_step_times_s: List[float] = []
        prefill_time_s: Optional[float] = None
        cache_bytes_final: int = 0

        if use_cache:
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            # --- Prefill: process full prompt once ---
            t0 = time.perf_counter()
            logits, kv_caches = model(prompt_tensor, kv_caches=None)
            prefill_time_s = time.perf_counter() - t0

            # First generated token comes from the last position of the prefill output
            next_id = int(logits[0, -1].argmax())
            generated_ids.append(next_id)

            # --- Decode: one token at a time, reusing cached K/V ---
            for _ in range(max_new_tokens - 1):
                input_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
                t0 = time.perf_counter()
                logits, kv_caches = model(input_tensor, kv_caches=kv_caches)
                decode_step_times_s.append(time.perf_counter() - t0)
                next_id = int(logits[0, -1].argmax())
                generated_ids.append(next_id)

            # Compute final cache size in bytes
            cache_bytes_final = sum(
                k.element_size() * k.numel() + v.element_size() * v.numel()
                for k, v in kv_caches
            )

        else:
            # --- No-cache: full recompute at every step ---
            all_ids: List[int] = list(prompt_ids)

            for _ in range(max_new_tokens):
                input_tensor = torch.tensor([all_ids], dtype=torch.long, device=device)
                t0 = time.perf_counter()
                logits, _ = model(input_tensor, kv_caches=None)
                decode_step_times_s.append(time.perf_counter() - t0)
                next_id = int(logits[0, -1].argmax())
                generated_ids.append(next_id)
                all_ids.append(next_id)

    return {
        "use_cache": use_cache,
        "prompt_ids": prompt_ids,
        "generated_ids": generated_ids,
        "prefill_time_s": prefill_time_s,
        "decode_step_times_s": decode_step_times_s,
        "cache_bytes_final": cache_bytes_final,
    }
