"""
Minimal decoder-only transformer.

Components
----------
TransformerBlock  — pre-norm attention + pre-norm feed-forward
DecoderTransformer — token embedding + positional embedding + N blocks + LM head

The model accepts an optional list of per-layer KV caches so that the same
forward() call serves both prefill (kv_caches=None) and cached decode
(kv_caches=list of (K,V) tuples from the previous step).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .attention import KVCache, MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        # Pre-norm residual attention
        attn_out, new_cache = self.attn(self.norm1(x), kv_cache)
        x = x + attn_out
        # Pre-norm residual feed-forward
        x = x + self.ff(self.norm2(x))
        return x, new_cache


class DecoderTransformer(nn.Module):
    """
    Parameters
    ----------
    vocab_size  : number of token ids (e.g. 98 for CharTokenizer)
    d_model     : embedding / hidden dimension
    n_heads     : number of attention heads (must divide d_model)
    d_ff        : feed-forward inner dimension
    n_layers    : number of transformer blocks
    max_seq_len : maximum supported sequence length (for positional embeddings)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        token_ids: torch.Tensor,                      # [batch, seq_len]
        kv_caches: Optional[List[Optional[KVCache]]] = None,
    ) -> Tuple[torch.Tensor, List[KVCache]]:
        """
        Returns
        -------
        logits     : [batch, seq_len, vocab_size]
        new_caches : list of (K, V) per layer, each K/V shaped
                     [batch, past_len+seq_len, n_heads, d_head]
        """
        batch, seq_len = token_ids.shape

        # Positional offset: if there is an existing cache, new tokens continue
        # from the position after the last cached token.
        past_len = 0
        if kv_caches is not None and kv_caches[0] is not None:
            past_len = kv_caches[0][0].shape[1]  # K_past.shape[1]

        positions = torch.arange(past_len, past_len + seq_len, device=token_ids.device)
        x = self.token_emb(token_ids) + self.pos_emb(positions)

        new_caches: List[KVCache] = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, layer_cache)
            new_caches.append(new_cache)

        x = self.norm(x)
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
        return logits, new_caches
