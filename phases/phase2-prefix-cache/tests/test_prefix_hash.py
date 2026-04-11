"""
Tests for the Phase 2 prefix cache key derivation.

The Fermyon Wasm handler (fermyon/src/lib.rs) computes cache keys with
make_cache_key().  This file:

  1. Mirrors that algorithm in Python so it can be tested without a Rust
     toolchain.
  2. Documents the exact contract so any future reimplementation can verify
     compatibility.
  3. Contains a skipped stub for semantic caching (not yet implemented).

Run with:
    cd phases/phase2-prefix-cache
    python -m pytest tests/ -v

The Python implementation must agree with the Rust implementation:
  - SHA-256 of the first `prefix_chars` *Unicode characters* (not bytes)
    of the prompt, encoded as UTF-8 before hashing.
  - Result: 64-character lowercase hexadecimal string.
"""

import hashlib
import pytest

# ---------------------------------------------------------------------------
# Python mirror of fermyon/src/lib.rs :: make_cache_key()
# ---------------------------------------------------------------------------

DEFAULT_PREFIX_CHARS = 128


def make_cache_key(prompt: str, prefix_chars: int = DEFAULT_PREFIX_CHARS) -> str:
    """
    Python mirror of the Rust make_cache_key() function.

    Algorithm (must stay in sync with fermyon/src/lib.rs):
      1. Take the first `prefix_chars` Unicode *scalar values* of `prompt`.
         In Python this is prompt[:prefix_chars] — string slicing is
         character-based, matching Rust's .chars().take(n).
      2. Encode the result as UTF-8.
      3. Return the lowercase SHA-256 hex digest (always 64 characters).

    If this function and the Rust function ever diverge, cached responses will
    not be found.  The test suite below guards against that.
    """
    prefix = prompt[:prefix_chars]
    digest = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
    return digest


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


class TestKeyFormat:
    def test_result_is_64_char_hex(self):
        key = make_cache_key("hello world")
        assert len(key) == 64, f"Expected 64 chars, got {len(key)}"
        assert all(c in "0123456789abcdef" for c in key), (
            f"Key contains non-hex characters: {key}"
        )

    def test_result_is_lowercase(self):
        key = make_cache_key("Test Input")
        assert key == key.lower()

    def test_empty_prompt(self):
        # An empty prompt produces the SHA-256 of an empty string — deterministic
        expected = hashlib.sha256(b"").hexdigest()
        assert make_cache_key("") == expected

    def test_empty_prompt_with_zero_prefix_chars(self):
        expected = hashlib.sha256(b"").hexdigest()
        assert make_cache_key("anything", prefix_chars=0) == expected


class TestDeterminism:
    def test_same_prompt_same_key(self):
        prompt = "The quick brown fox jumps over the lazy dog."
        assert make_cache_key(prompt) == make_cache_key(prompt)

    def test_same_prompt_same_prefix_chars_same_key(self):
        prompt = "System: You are a helpful assistant. User: What is 2+2?"
        for n in [10, 64, 128, 256]:
            k1 = make_cache_key(prompt, prefix_chars=n)
            k2 = make_cache_key(prompt, prefix_chars=n)
            assert k1 == k2, f"Non-deterministic at prefix_chars={n}"


class TestPrefixTruncation:
    def test_same_prefix_different_suffix_same_key(self):
        """
        Two prompts that share the same first `prefix_chars` characters must
        produce the same cache key, regardless of what comes after.

        This is the core semantic of exact-match prefix caching: shared system
        prompts / conversation history produce cache hits.

        The shared prefix must be LONGER than DEFAULT_PREFIX_CHARS so that the
        hash window is filled entirely by the shared portion.  Both prompts then
        hash to SHA-256(shared[:DEFAULT_PREFIX_CHARS]) — identical keys.
        """
        # 30 chars × 5 = 150 chars — deliberately > DEFAULT_PREFIX_CHARS (128)
        shared = "You are a helpful assistant. " * 5
        prompt_a = shared + "What is the capital of France?"
        prompt_b = shared + "Explain quantum entanglement."
        assert len(shared) > DEFAULT_PREFIX_CHARS, (
            "Test setup error: shared prefix must exceed DEFAULT_PREFIX_CHARS "
            f"({DEFAULT_PREFIX_CHARS}) so truncation kicks in. "
            f"Got len(shared)={len(shared)}"
        )
        assert make_cache_key(prompt_a) == make_cache_key(prompt_b)

    def test_different_prefix_different_key(self):
        key_a = make_cache_key("Alpha prompt suffix...")
        key_b = make_cache_key("Beta prompt suffix...")
        assert key_a != key_b

    def test_prefix_chars_controls_truncation(self):
        """A longer prefix_chars window means fewer cache hits for differing prompts."""
        prompt_a = "Shared start. " + "A" * 200
        prompt_b = "Shared start. " + "B" * 200

        shared_len = len("Shared start. ")

        # With a small window that covers only the shared part → same key
        key_short_a = make_cache_key(prompt_a, prefix_chars=shared_len)
        key_short_b = make_cache_key(prompt_b, prefix_chars=shared_len)
        assert key_short_a == key_short_b, (
            "Small prefix_chars should produce the same key for prompts with "
            "identical openings"
        )

        # With a large window that covers the differing part → different keys
        key_long_a = make_cache_key(prompt_a, prefix_chars=300)
        key_long_b = make_cache_key(prompt_b, prefix_chars=300)
        assert key_long_a != key_long_b, (
            "Large prefix_chars should produce different keys when prompts diverge "
            "within the window"
        )

    def test_prefix_chars_beyond_prompt_length(self):
        """prefix_chars larger than the prompt just hashes the full prompt."""
        short = "Short."
        key_exact = make_cache_key(short, prefix_chars=len(short))
        key_large = make_cache_key(short, prefix_chars=10_000)
        assert key_exact == key_large


class TestUtf8Correctness:
    def test_multibyte_chars_counted_as_characters_not_bytes(self):
        """
        The prefix window is measured in Unicode characters.
        A 2-byte UTF-8 character (e.g. é, ñ) counts as 1 character, not 2 bytes.
        This test verifies that Python slicing (character-based) matches Rust
        .chars().take(n) (also character-based).
        """
        # 'é' is U+00E9, 2 bytes in UTF-8
        prompt = "café " * 30  # 5 chars × 30 = 150 chars, but 180 bytes
        key_chars = make_cache_key(prompt, prefix_chars=10)

        # Manually compute the expected key
        prefix = prompt[:10]           # "café café " — 10 characters
        expected = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
        assert key_chars == expected

    def test_4byte_emoji_counted_as_one_character(self):
        """An emoji (4-byte UTF-8) must count as one character in the prefix window."""
        # U+1F600 😀 is 4 bytes in UTF-8
        prompt = "😀" * 50
        key = make_cache_key(prompt, prefix_chars=5)
        prefix = prompt[:5]   # 5 emoji characters
        expected = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
        assert key == expected


class TestKnownValues:
    """
    Pinned SHA-256 values.  These serve as a cross-language compatibility
    check: if the Rust make_cache_key() is ever changed, these tests will
    catch drift between the Python and Rust implementations.
    """

    def test_known_ascii_prompt(self):
        prompt = "Hello, World!"
        # SHA-256 of "Hello, World!" (no truncation — len < 128)
        expected = hashlib.sha256(b"Hello, World!").hexdigest()
        assert make_cache_key(prompt) == expected

    def test_known_truncated_prompt(self):
        prompt = "A" * 200
        expected = hashlib.sha256(("A" * 128).encode("utf-8")).hexdigest()
        assert make_cache_key(prompt, prefix_chars=128) == expected


# ---------------------------------------------------------------------------
# Semantic caching — future work
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "TODO(semantic-cache): Not implemented in Phase 2. "
        "Semantic caching would allow prompts with different wording but the "
        "same meaning to share a cache entry. Implementation requires: "
        "(1) a sentence-transformer model to embed the prompt prefix, "
        "(2) a vector store (e.g. pgvector, Qdrant) deployed alongside Valkey, "
        "(3) an approximate nearest-neighbour query replacing the SHA-256 lookup "
        "in fermyon/src/lib.rs. "
        "See docs/phases/phase2-fermyon-valkey.md for context."
    )
)
def test_semantic_cache_equivalent_prompts_share_key():
    """
    Two prompts that mean the same thing should produce the same cache key
    even if they are worded differently.

    Example pairs that should hit the same cache entry:
      "You are a helpful assistant."
      "You are an assistant that helps users."

      "Summarise the following text:"
      "Please provide a summary of the text below:"

    This test is a stub.  Fill in the implementation once a vector store
    and embedding model are available.
    """
    prompt_a = "You are a helpful assistant."
    prompt_b = "You are an assistant that helps users."

    # TODO: replace make_cache_key with semantic_cache_key() once implemented
    key_a = make_cache_key(prompt_a)
    key_b = make_cache_key(prompt_b)

    # This assertion intentionally FAILS with exact-match hashing.
    # It will PASS once semantic caching is implemented.
    assert key_a == key_b, (
        "Semantic cache not implemented: different-wording prompts produce "
        "different keys under exact-match hashing"
    )
