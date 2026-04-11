"""
Multi-head self-attention with an explicit, inspectable KV cache.

Cache format: a tuple (K, V) where each tensor has shape
    [batch, seq_len, n_heads, d_head]

During prefill (kv_cache=None, seq_len > 1):
  - Full causal mask is applied.
  - Returns output + a fresh (K, V) cache for every position.

During cached decode (kv_cache provided, seq_len=1):
  - New K/V for the single input token are appended to the cache.
  - No causal mask needed: a single query always attends to all past.

During no-cache decode (kv_cache=None, seq_len > 1):
  - Full causal mask applied over the entire growing sequence.
  - Identical numerical output to the cached path.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Type alias for one layer's cache entry
KVCache = Tuple[torch.Tensor, torch.Tensor]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,                    # [batch, seq_len, d_model]
        kv_cache: Optional[KVCache] = None, # (K_past, V_past) or None
    ) -> Tuple[torch.Tensor, KVCache]:
        batch, seq_len, _ = x.shape

        # Project queries, keys, values for the current input tokens
        Q = self.W_q(x).view(batch, seq_len, self.n_heads, self.d_head)
        K = self.W_k(x).view(batch, seq_len, self.n_heads, self.d_head)
        V = self.W_v(x).view(batch, seq_len, self.n_heads, self.d_head)

        # Extend with cached keys/values from previous steps
        past_len = 0
        if kv_cache is not None:
            K_past, V_past = kv_cache
            past_len = K_past.shape[1]
            K = torch.cat([K_past, K], dim=1)  # [batch, past+seq, n_heads, d_head]
            V = torch.cat([V_past, V], dim=1)

        # Save the updated cache before transposing
        new_cache: KVCache = (K, V)

        # Transpose to [batch, n_heads, seq, d_head] for attention matmul
        Q = Q.transpose(1, 2)               # [batch, n_heads, seq_len, d_head]
        K_t = K.transpose(1, 2)             # [batch, n_heads, past+seq, d_head]
        V_t = V.transpose(1, 2)             # [batch, n_heads, past+seq, d_head]

        # Scaled dot-product attention
        scale = math.sqrt(self.d_head)
        # scores: [batch, n_heads, seq_len, past+seq_len]
        scores = torch.matmul(Q, K_t.transpose(-2, -1)) / scale

        # Causal mask: applied when seq_len > 1 (prefill or full-recompute decode).
        # Skipped for single-token cached decode — the single query is always the
        # latest position and correctly attends to all prior keys.
        if seq_len > 1:
            total_len = past_len + seq_len
            # query index q attends to key index k iff k <= past_len + q
            q_idx = torch.arange(seq_len, device=x.device).unsqueeze(1)   # [seq_len, 1]
            k_idx = torch.arange(total_len, device=x.device).unsqueeze(0) # [1, total_len]
            future_mask = k_idx > (past_len + q_idx)                       # [seq_len, total_len]
            scores = scores.masked_fill(
                future_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V_t)  # [batch, n_heads, seq_len, d_head]

        # Merge heads and project
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        out = self.W_o(out)

        return out, new_cache
