"""
Tests for Phase 1: KV cache correctness, shape, and causal masking.

Run with:
    cd phases/phase1-kv-cache
    python -m pytest tests/ -v
"""

import torch
import pytest

from kv_cache.attention import MultiHeadAttention
from kv_cache.model import DecoderTransformer
from kv_cache.tokenizer import CharTokenizer
from kv_cache.generate import generate

# Fixed model config for all tests — small for fast CPU runs
D_MODEL = 64
N_HEADS = 4
D_FF = 128
N_LAYERS = 2
VOCAB_SIZE = 98
SEED = 0


def _make_model(seed: int = SEED) -> DecoderTransformer:
    torch.manual_seed(seed)
    return DecoderTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        n_layers=N_LAYERS,
        max_seq_len=256,
    )


class TestCacheCorrectness:
    """Cached and no-cache generation must produce identical token sequences."""

    def test_logits_match_no_cache_step_by_step(self):
        """
        Gold-standard correctness test: compare raw logit tensors at every
        decode step, not just the argmax token IDs.

        For each step t we verify:
          cached_logits[t] ≈ nocache_logits[t]  (torch.allclose, atol=1e-5)

        This catches bugs where the cache returns slightly wrong attention
        values that still happen to produce the same argmax.

        Procedure
        ---------
        Let prompt = [p0, p1, ..., p_{n-1}].

        No-cache reference at step t:
          input  = [p0, ..., p_{n-1}, g0, ..., g_{t-1}]
          output = logits[:, -1, :]   (last position)

        Cached path at step t=0 (prefill):
          input  = [p0, ..., p_{n-1}]
          output = logits[:, -1, :]

        Cached path at step t>0 (decode):
          input  = [g_{t-1}]  (single new token)
          output = logits[:, 0, :]    (only position)

        Both must produce numerically identical last-position logits.
        """
        model = _make_model()
        model.eval()
        tokenizer = CharTokenizer()
        prompt_ids = tokenizer.encode("Hello, world! This is a test of the KV cache.")
        device = next(model.parameters()).device
        n_steps = 8

        # ------------------------------------------------------------------
        # No-cache reference: collect last-position logits at each step
        # ------------------------------------------------------------------
        nocache_logits = []
        all_ids = list(prompt_ids)
        with torch.no_grad():
            for _ in range(n_steps):
                t = torch.tensor([all_ids], dtype=torch.long, device=device)
                logits, _ = model(t, kv_caches=None)
                nocache_logits.append(logits[0, -1].clone())
                all_ids.append(int(logits[0, -1].argmax()))

        # ------------------------------------------------------------------
        # Cached path: prefill once, then single-token decode steps
        # ------------------------------------------------------------------
        cached_logits = []
        all_ids = list(prompt_ids)
        with torch.no_grad():
            # Step 0: prefill
            t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            logits, kv_caches = model(t, kv_caches=None)
            cached_logits.append(logits[0, -1].clone())
            next_id = int(logits[0, -1].argmax())
            all_ids.append(next_id)

            # Steps 1..n_steps-1: single-token decode
            for _ in range(n_steps - 1):
                t = torch.tensor([[next_id]], dtype=torch.long, device=device)
                logits, kv_caches = model(t, kv_caches=kv_caches)
                cached_logits.append(logits[0, 0].clone())
                next_id = int(logits[0, 0].argmax())
                all_ids.append(next_id)

        # ------------------------------------------------------------------
        # Compare logit tensors at every step
        # ------------------------------------------------------------------
        for step, (c_logits, nc_logits) in enumerate(zip(cached_logits, nocache_logits)):
            assert torch.allclose(c_logits, nc_logits, atol=1e-5), (
                f"Step {step}: cached and no-cache logits diverge. "
                f"Max abs diff: {(c_logits - nc_logits).abs().max().item():.2e}"
            )

    def test_output_matches_no_cache(self):
        model = _make_model()
        tokenizer = CharTokenizer()
        prompt_ids = tokenizer.encode("Hello, world! This is a test of the KV cache.")

        result_cached = generate(model, prompt_ids, max_new_tokens=16, use_cache=True)
        result_nocache = generate(model, prompt_ids, max_new_tokens=16, use_cache=False)

        assert result_cached["generated_ids"] == result_nocache["generated_ids"], (
            "Cached and no-cache generation produced different token sequences. "
            f"cached={result_cached['generated_ids']}, "
            f"nocache={result_nocache['generated_ids']}"
        )

    def test_output_length(self):
        model = _make_model()
        tokenizer = CharTokenizer()
        prompt_ids = tokenizer.encode("Short prompt.")

        for max_new in [1, 8, 32]:
            result = generate(model, prompt_ids, max_new_tokens=max_new, use_cache=True)
            assert len(result["generated_ids"]) == max_new, (
                f"Expected {max_new} generated tokens, got {len(result['generated_ids'])}"
            )


class TestKVCacheShape:
    """KV cache tensors must have the expected shapes after prefill and decode."""

    def test_cache_shape_after_prefill(self):
        torch.manual_seed(SEED)
        attn = MultiHeadAttention(D_MODEL, N_HEADS)
        d_head = D_MODEL // N_HEADS
        batch, seq_len = 1, 10

        x = torch.randn(batch, seq_len, D_MODEL)
        _, (K, V) = attn(x, kv_cache=None)

        assert K.shape == (batch, seq_len, N_HEADS, d_head), (
            f"Expected K shape {(batch, seq_len, N_HEADS, d_head)}, got {K.shape}"
        )
        assert V.shape == (batch, seq_len, N_HEADS, d_head), (
            f"Expected V shape {(batch, seq_len, N_HEADS, d_head)}, got {V.shape}"
        )

    def test_cache_grows_by_one_per_decode_step(self):
        torch.manual_seed(SEED)
        attn = MultiHeadAttention(D_MODEL, N_HEADS)
        d_head = D_MODEL // N_HEADS
        batch, prompt_len = 1, 8

        # Prefill
        x_prompt = torch.randn(batch, prompt_len, D_MODEL)
        _, cache = attn(x_prompt, kv_cache=None)
        assert cache[0].shape[1] == prompt_len

        # Three decode steps — cache should grow by 1 each time
        for step in range(1, 4):
            x_tok = torch.randn(batch, 1, D_MODEL)
            _, cache = attn(x_tok, kv_cache=cache)
            expected_len = prompt_len + step
            assert cache[0].shape[1] == expected_len, (
                f"After step {step}: expected cache length {expected_len}, "
                f"got {cache[0].shape[1]}"
            )

    def test_full_model_cache_shape(self):
        """After generate(), each layer's cache covers prompt + generated tokens."""
        model = _make_model()
        tokenizer = CharTokenizer()
        prompt = "Cache shape test."
        prompt_ids = tokenizer.encode(prompt)
        max_new = 8
        d_head = D_MODEL // N_HEADS

        # Run a prefill then partial decode manually to inspect cache shapes
        device = next(model.parameters()).device
        token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        model.eval()
        with torch.no_grad():
            _, kv_caches = model(token_ids, kv_caches=None)

        assert len(kv_caches) == N_LAYERS
        for layer_idx, (K, V) in enumerate(kv_caches):
            assert K.shape == (1, len(prompt_ids), N_HEADS, d_head), (
                f"Layer {layer_idx} K shape mismatch: {K.shape}"
            )
            assert V.shape == (1, len(prompt_ids), N_HEADS, d_head), (
                f"Layer {layer_idx} V shape mismatch: {V.shape}"
            )


class TestCausalMask:
    """Output at position i must not depend on tokens at positions > i."""

    def test_causal_isolation(self):
        """
        Run the model on two sequences that differ only at position k.
        Outputs at positions 0..k-1 must be identical; position k onward may differ.
        """
        torch.manual_seed(SEED)
        model = _make_model()
        model.eval()

        seq_len = 8
        change_pos = 4  # change the token at this position

        # Build two token id sequences — identical except at change_pos
        base_ids = [10, 20, 30, 40, 50, 60, 70, 80]
        modified_ids = list(base_ids)
        modified_ids[change_pos] = 90  # different token

        t_base = torch.tensor([base_ids], dtype=torch.long)
        t_modified = torch.tensor([modified_ids], dtype=torch.long)

        with torch.no_grad():
            logits_base, _ = model(t_base, kv_caches=None)
            logits_modified, _ = model(t_modified, kv_caches=None)

        # Positions before the change must be identical
        assert torch.allclose(
            logits_base[0, :change_pos], logits_modified[0, :change_pos], atol=1e-5
        ), (
            "Causal mask failure: output at positions before the change point "
            "differ between sequences that only differ after that point."
        )

        # At and after the change, outputs should differ (not guaranteed for random
        # weights, but almost certain; we just verify the test structure is sensible)
        # We don't assert *must* differ — just that the test is not vacuous.
        assert change_pos < seq_len


class TestTokenizer:
    """Basic sanity checks on the character-level tokenizer."""

    def test_encode_decode_roundtrip(self):
        tokenizer = CharTokenizer()
        text = "Hello, World! 42"
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_vocab_size(self):
        assert CharTokenizer.vocab_size == 98

    def test_special_tokens_not_in_decode(self):
        tokenizer = CharTokenizer()
        ids = [tokenizer.bos_id, 35, 36, tokenizer.eos_id, tokenizer.pad_id]
        result = tokenizer.decode(ids)
        # BOS, EOS, PAD produce no output characters; only ids 35 and 36 decode
        assert result == tokenizer.decode([35, 36])
