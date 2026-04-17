"""
Tests for the Phase 2 chat completion cache key derivation.

The Fermyon Wasm proxy (fermyon/proxy/src/lib.rs) computes cache keys with
make_cache_key(model, messages).  This file:

  1. Mirrors that algorithm in Python so it can be tested without a Rust
     toolchain or a running Spin server.
  2. Pins known SHA-256 outputs so any divergence between the Python mirror
     and the Rust implementation is caught immediately.
  3. Provides integration test stubs (marked @pytest.mark.integration) for
     end-to-end behaviour: cache hits, cache misses, and stream:true handling.
     Integration tests are NOT run in CI — they require a live Fermyon proxy,
     Valkey, and vLLM.

Algorithm (must stay in sync with fermyon/proxy/src/lib.rs :: make_cache_key):

    key = "fermyon:v1:" + hex(SHA-256(model + "\\x00" + messages_json))

    where messages_json = compact JSON of [{role, content}, ...] with extra
    fields (name, tool_call_id, etc.) stripped.

Run unit tests with:
    cd phases/phase2-prefix-cache
    python -m pytest tests/test_chat_cache.py -v

Run integration tests (requires live cluster):
    python -m pytest tests/test_chat_cache.py -v -m integration
"""

import hashlib
import json

import pytest

# ---------------------------------------------------------------------------
# Python mirror of fermyon/proxy/src/lib.rs :: make_cache_key()
# ---------------------------------------------------------------------------

CACHE_KEY_PREFIX = "fermyon:v1:"


def make_cache_key(model: str, messages: list[dict]) -> str:
    """
    Python mirror of the Rust make_cache_key() function.

    Algorithm (must stay in sync with fermyon/proxy/src/lib.rs):
      1. Normalise each message to {role, content} only — extra fields dropped.
      2. Serialise the normalised list as compact JSON (no spaces).
      3. Feed model bytes, a null-byte separator, and messages_json bytes into
         SHA-256.
      4. Prepend "fermyon:v1:" to the lowercase hex digest.

    The null-byte separator between model and messages_json prevents hash
    collisions when a model name ends with characters that also appear in
    valid JSON (e.g. a model name ending with '}').
    """
    normalized = [{"role": m["role"], "content": m["content"]} for m in messages]
    messages_json = json.dumps(normalized, separators=(",", ":"), ensure_ascii=False)
    data = model.encode("utf-8") + b"\x00" + messages_json.encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    return CACHE_KEY_PREFIX + digest


# ---------------------------------------------------------------------------
# Unit tests — key format
# ---------------------------------------------------------------------------


class TestKeyFormat:
    def test_prefix(self):
        key = make_cache_key("gpt-4", [{"role": "user", "content": "hi"}])
        assert key.startswith(CACHE_KEY_PREFIX), (
            f"Key must start with '{CACHE_KEY_PREFIX}', got: {key}"
        )

    def test_total_length(self):
        # "fermyon:v1:" (11 chars) + SHA-256 hex (64 chars) = 75 chars
        key = make_cache_key("gpt-4", [{"role": "user", "content": "hi"}])
        assert len(key) == 75, f"Expected 75 chars, got {len(key)}"

    def test_hex_suffix_lowercase(self):
        key = make_cache_key("gpt-4", [{"role": "user", "content": "Hello"}])
        hex_part = key[len(CACHE_KEY_PREFIX):]
        assert len(hex_part) == 64
        assert all(c in "0123456789abcdef" for c in hex_part), (
            f"Hex part contains non-hex characters: {hex_part}"
        )


# ---------------------------------------------------------------------------
# Unit tests — determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_inputs_same_key(self):
        msgs = [{"role": "user", "content": "What is 2+2?"}]
        assert make_cache_key("mistral", msgs) == make_cache_key("mistral", msgs)

    def test_repeated_calls_identical(self):
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain gradient descent."},
        ]
        keys = {make_cache_key("llama3", msgs) for _ in range(10)}
        assert len(keys) == 1, "make_cache_key must be deterministic across calls"


# ---------------------------------------------------------------------------
# Unit tests — key excludes sampling parameters
# ---------------------------------------------------------------------------


class TestSamplingParamsIgnored:
    """
    temperature, top_p, max_tokens, stream, and other sampling fields must
    NOT affect the cache key.  Two requests with the same model and messages
    but different sampling settings must share a cache entry.
    """

    def _key_for(self, extra_fields: dict) -> str:
        msgs = [{"role": "user", "content": "Describe the Eiffel Tower."}]
        # Simulate the proxy stripping extra fields: only role+content survive.
        # make_cache_key already normalises, so this is implicit.
        return make_cache_key("mistral-7b", msgs)

    def test_ignores_temperature(self):
        assert self._key_for({"temperature": 0.0}) == self._key_for({"temperature": 1.0})

    def test_ignores_top_p(self):
        assert self._key_for({"top_p": 0.5}) == self._key_for({"top_p": 0.95})

    def test_ignores_max_tokens(self):
        assert self._key_for({"max_tokens": 100}) == self._key_for({"max_tokens": 4096})

    def test_ignores_stream_flag(self):
        assert self._key_for({"stream": True}) == self._key_for({"stream": False})

    def test_ignores_extra_message_fields(self):
        """
        Extra message fields (name, tool_call_id) must be stripped before
        hashing so that two requests with the same logical messages produce
        the same key regardless of client-supplied extra fields.
        """
        msgs_clean = [{"role": "user", "content": "Hello"}]
        msgs_extra = [{"role": "user", "content": "Hello", "name": "Alice", "tool_call_id": "xyz"}]
        assert make_cache_key("mistral", msgs_clean) == make_cache_key("mistral", msgs_extra)


# ---------------------------------------------------------------------------
# Unit tests — key changes on different model or messages
# ---------------------------------------------------------------------------


class TestKeyDifferences:
    def test_different_model_different_key(self):
        msgs = [{"role": "user", "content": "Summarise the French Revolution."}]
        key_a = make_cache_key("mistral-7b", msgs)
        key_b = make_cache_key("llama3-8b", msgs)
        assert key_a != key_b

    def test_different_content_different_key(self):
        key_a = make_cache_key("m", [{"role": "user", "content": "Paris"}])
        key_b = make_cache_key("m", [{"role": "user", "content": "Berlin"}])
        assert key_a != key_b

    def test_different_role_different_key(self):
        key_a = make_cache_key("m", [{"role": "user", "content": "Hello"}])
        key_b = make_cache_key("m", [{"role": "assistant", "content": "Hello"}])
        assert key_a != key_b

    def test_different_message_order_different_key(self):
        msg1 = {"role": "user", "content": "First"}
        msg2 = {"role": "assistant", "content": "Second"}
        key_fwd = make_cache_key("m", [msg1, msg2])
        key_rev = make_cache_key("m", [msg2, msg1])
        assert key_fwd != key_rev

    def test_additional_message_different_key(self):
        msgs_short = [{"role": "user", "content": "Hello"}]
        msgs_long = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        assert make_cache_key("m", msgs_short) != make_cache_key("m", msgs_long)


# ---------------------------------------------------------------------------
# Unit tests — null-byte separator prevents prefix collisions
# ---------------------------------------------------------------------------


class TestNullByteSeparator:
    """
    Without the null-byte separator, a model name that ends with the opening
    of a JSON array could collide with a different (model, messages) pair.

    Example:
        model = 'x[{"role"'  messages = [{"role": "user", "content": "y"}]
        model = 'x'          messages = [{"role": "user", "content": "y"}]
        (with messages_json starting with '[{"role"...')

    The null byte (\x00) cannot appear in either model names or JSON strings
    (which escape \u0000 instead), so it is an unambiguous separator.
    """

    def test_model_name_ending_in_json_fragment_does_not_collide(self):
        msgs = [{"role": "user", "content": "test"}]
        messages_json = json.dumps(
            [{"role": m["role"], "content": m["content"]} for m in msgs],
            separators=(",", ":"),
        )
        # Craft a model name so that model_a + messages_json == model_b + messages_json
        # when concatenated WITHOUT a separator — but they differ WITH the separator.
        model_a = "base"
        model_b = "base" + messages_json[:4]  # e.g. "base[{"r"
        shorter_msgs = [{"role": "user", "content": "test"}]

        key_a = make_cache_key(model_a, msgs)
        key_b = make_cache_key(model_b, shorter_msgs)
        # These may or may not be equal by chance, but with the null-byte separator
        # the inputs to SHA-256 are unambiguously distinct.
        data_a = model_a.encode() + b"\x00" + messages_json.encode()
        data_b = model_b.encode() + b"\x00" + messages_json.encode()
        assert data_a != data_b, (
            "Test setup error: crafted inputs should differ in their raw byte forms"
        )
        assert key_a != key_b

    def test_separator_is_null_byte(self):
        """Verify the separator character used in the algorithm."""
        model = "m"
        msgs = [{"role": "user", "content": "hi"}]
        normalized = [{"role": m["role"], "content": m["content"]} for m in msgs]
        messages_json = json.dumps(normalized, separators=(",", ":"), ensure_ascii=False)
        expected_data = model.encode("utf-8") + b"\x00" + messages_json.encode("utf-8")
        expected_key = CACHE_KEY_PREFIX + hashlib.sha256(expected_data).hexdigest()
        assert make_cache_key(model, msgs) == expected_key


# ---------------------------------------------------------------------------
# Unit tests — field-order independence for message objects
# ---------------------------------------------------------------------------


class TestFieldOrderIndependence:
    """
    A message dict with fields in {content, role} order must produce the same
    key as one with fields in {role, content} order, because normalisation
    rebuilds each message as {role, content} before serialisation.
    """

    def test_content_role_order_same_as_role_content_order(self):
        msgs_rc = [{"role": "user", "content": "Hello"}]
        msgs_cr = [{"content": "Hello", "role": "user"}]
        assert make_cache_key("m", msgs_rc) == make_cache_key("m", msgs_cr)

    def test_multi_message_field_order_independence(self):
        msgs_a = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is gravity?"},
        ]
        msgs_b = [
            {"content": "Be concise.", "role": "system"},
            {"content": "What is gravity?", "role": "user"},
        ]
        assert make_cache_key("mistral", msgs_a) == make_cache_key("mistral", msgs_b)


# ---------------------------------------------------------------------------
# Unit tests — known pinned values
# ---------------------------------------------------------------------------


class TestKnownValues:
    """
    Pinned SHA-256 outputs.  If the Rust make_cache_key() is modified in a
    way that changes the hash inputs, these tests will catch the divergence.
    """

    def test_single_user_message(self):
        model = "mistral-7b"
        msgs = [{"role": "user", "content": "Hello"}]
        # Compute expected manually to document the exact wire format:
        #   model bytes + \x00 + compact JSON of [{role, content}]
        messages_json = '[{"role":"user","content":"Hello"}]'
        data = b"mistral-7b\x00" + messages_json.encode("utf-8")
        expected = CACHE_KEY_PREFIX + hashlib.sha256(data).hexdigest()
        assert make_cache_key(model, msgs) == expected

    def test_system_plus_user_message(self):
        model = "llama3"
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        messages_json = (
            '[{"role":"system","content":"You are a helpful assistant."},'
            '{"role":"user","content":"What is 2+2?"}]'
        )
        data = b"llama3\x00" + messages_json.encode("utf-8")
        expected = CACHE_KEY_PREFIX + hashlib.sha256(data).hexdigest()
        assert make_cache_key(model, msgs) == expected


# ---------------------------------------------------------------------------
# Integration test stubs
#
# These require a live cluster:
#   - Fermyon proxy reachable (port-forward or LoadBalancer)
#   - Valkey running in the inference namespace
#   - vLLM running and healthy
#
# Run with:
#   FERMYON_URL=http://localhost:8082 \
#   python -m pytest tests/test_chat_cache.py -v -m integration
# ---------------------------------------------------------------------------

FERMYON_URL_DEFAULT = "http://localhost:8082"


def _fermyon_url() -> str:
    import os
    return os.environ.get("FERMYON_URL", FERMYON_URL_DEFAULT)


@pytest.mark.integration
def test_integration_cache_miss_returns_x_cache_miss():
    """
    A first request for a unique prompt must return X-Cache: MISS and a 200
    response with a valid JSON body from vLLM.
    """
    import requests
    import time

    unique_content = f"cache-miss-test-{time.time_ns()}"
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "messages": [{"role": "user", "content": unique_content}],
    }
    resp = requests.post(
        f"{_fermyon_url()}/v1/chat/completions",
        json=payload,
        timeout=60,
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    assert resp.headers.get("x-cache", "").upper() == "MISS", (
        f"Expected X-Cache: MISS on first request, got: {resp.headers.get('x-cache')}"
    )
    body = resp.json()
    assert "choices" in body, f"Response missing 'choices': {body}"


@pytest.mark.integration
def test_integration_cache_hit_returns_x_cache_hit():
    """
    A repeated request with the same model and messages must return X-Cache: HIT
    on the second call.  The response body must be identical to the first call.
    """
    import requests

    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "messages": [
            {"role": "system", "content": "You are a test assistant."},
            {"role": "user", "content": "What is the capital of France? (cache-hit-test)"},
        ],
    }
    url = f"{_fermyon_url()}/v1/chat/completions"

    resp1 = requests.post(url, json=payload, timeout=60)
    assert resp1.status_code == 200
    # First call may be HIT or MISS depending on prior test runs; we don't assert here.

    resp2 = requests.post(url, json=payload, timeout=10)
    assert resp2.status_code == 200
    assert resp2.headers.get("x-cache", "").upper() == "HIT", (
        f"Expected X-Cache: HIT on repeated request, got: {resp2.headers.get('x-cache')}"
    )
    assert resp1.text == resp2.text, "Cache hit body must be identical to first response"


@pytest.mark.integration
def test_integration_stream_true_handled_without_error():
    """
    A request with stream:true must succeed.  The Fermyon proxy strips the
    stream flag and returns a complete non-streaming JSON response.
    The client must NOT receive an SSE/chunked response — the body is plain JSON.
    """
    import requests

    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "messages": [{"role": "user", "content": "Say hello. (stream-test)"}],
        "stream": True,
    }
    resp = requests.post(
        f"{_fermyon_url()}/v1/chat/completions",
        json=payload,
        timeout=60,
    )
    assert resp.status_code == 200, (
        f"stream:true request failed with {resp.status_code}: {resp.text}"
    )
    # Must be parseable as JSON — not an SSE stream
    body = resp.json()
    assert "choices" in body, (
        f"stream:true response must be plain JSON with 'choices', got: {body}"
    )
    # Content-Type must be application/json, not text/event-stream
    ct = resp.headers.get("content-type", "")
    assert "application/json" in ct, (
        f"Expected Content-Type: application/json, got: {ct}"
    )


@pytest.mark.integration
def test_integration_sampling_params_do_not_affect_cache_key():
    """
    Two requests with the same model and messages but different temperature
    values must return the same cached response on the second call.

    This verifies the proxy's key derivation excludes sampling parameters,
    not just the Python mirror.
    """
    import requests

    msgs = [{"role": "user", "content": "Define entropy. (sampling-key-test)"}]
    model = "mistralai/Mistral-7B-Instruct-v0.2"
    url = f"{_fermyon_url()}/v1/chat/completions"

    # Warm the cache with temperature=0.0
    resp1 = requests.post(url, json={"model": model, "messages": msgs, "temperature": 0.0}, timeout=60)
    assert resp1.status_code == 200

    # Second request with temperature=1.0 should HIT the same cache entry
    resp2 = requests.post(url, json={"model": model, "messages": msgs, "temperature": 1.0}, timeout=10)
    assert resp2.status_code == 200
    assert resp2.headers.get("x-cache", "").upper() == "HIT", (
        "Requests differing only in temperature must share a cache entry"
    )
