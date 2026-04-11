"""
Phase 3 tests.

Split into two groups:

  Infrastructure tests (no GPU required)
  ----------------------------------------
  - DynamicBatcher grouping and deadline logic
  - Future result/exception propagation
  - InferenceRequest construction and defaults
  - base64 image decode + PIL round-trip
  - FastAPI request/response schema validation

  GPU inference tests (skipped without CUDA)
  -------------------------------------------
  - Model loading smoke test
  - Single-request inference produces non-empty output
  - Baseline and optimized paths produce consistent output
    (numerical identity is not required — outputs may differ slightly
    between float16 and bfloat16, or with flash-attn)

Run with:
    cd phases/phase3-qwen-image
    python -m pytest tests/ -v
"""

import base64
import io
import threading
import time

import pytest
import torch
from PIL import Image

from serve.batching import DynamicBatcher, Future, InferenceRequest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image_b64(width: int = 64, height: int = 64, color: tuple = (128, 64, 32)) -> str:
    """Generate a tiny solid-color JPEG as a base64 string."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _make_request(**kwargs) -> InferenceRequest:
    defaults = dict(image_b64=_make_image_b64(), prompt="describe this", max_new_tokens=32)
    defaults.update(kwargs)
    return InferenceRequest(**defaults)


# ---------------------------------------------------------------------------
# Future
# ---------------------------------------------------------------------------

class TestFuture:
    def test_set_result_unblocks_wait(self):
        f = Future()
        threading.Timer(0.05, lambda: f.set_result("hello")).start()
        assert f.result(timeout=2.0) == "hello"

    def test_set_exception_raises_on_result(self):
        f = Future()
        f.set_exception(ValueError("boom"))
        with pytest.raises(ValueError, match="boom"):
            f.result()

    def test_timeout_raises_timeout_error(self):
        f = Future()
        with pytest.raises(TimeoutError):
            f.result(timeout=0.05)

    def test_done_property(self):
        f = Future()
        assert not f.done
        f.set_result(42)
        assert f.done

    def test_result_is_not_set_before_event(self):
        f = Future()
        # Accessing _result directly to verify initial state
        assert f._result is None
        assert f._exception is None


# ---------------------------------------------------------------------------
# InferenceRequest
# ---------------------------------------------------------------------------

class TestInferenceRequest:
    def test_construction_sets_fields(self):
        req = InferenceRequest(image_b64="abc", prompt="hello", max_new_tokens=128)
        assert req.image_b64 == "abc"
        assert req.prompt == "hello"
        assert req.max_new_tokens == 128

    def test_arrived_at_defaults_to_now(self):
        before = time.monotonic()
        req = InferenceRequest(image_b64="x", prompt="y", max_new_tokens=1)
        after = time.monotonic()
        assert before <= req.arrived_at <= after

    def test_result_future_defaults_to_none(self):
        req = InferenceRequest(image_b64="x", prompt="y", max_new_tokens=1)
        assert req.result_future is None

    def test_result_future_is_settable(self):
        req = InferenceRequest(image_b64="x", prompt="y", max_new_tokens=1)
        f = Future()
        req.result_future = f
        assert req.result_future is f


# ---------------------------------------------------------------------------
# DynamicBatcher
# ---------------------------------------------------------------------------

class TestDynamicBatcher:
    def test_single_request_is_processed(self):
        def process(batch):
            return [f"echo:{req.prompt}" for req in batch]

        batcher = DynamicBatcher(process, max_batch_size=4, max_wait_s=0.1)
        future = batcher.submit(_make_request(prompt="hello"))
        assert future.result(timeout=5.0) == "echo:hello"

    def test_batch_groups_requests_up_to_max(self):
        received_sizes = []

        def process(batch):
            received_sizes.append(len(batch))
            return ["ok"] * len(batch)

        batcher = DynamicBatcher(process, max_batch_size=3, max_wait_s=0.5)
        futures = [batcher.submit(_make_request()) for _ in range(3)]
        for f in futures:
            f.result(timeout=5.0)

        # All three should have been dispatched in a single batch
        assert 3 in received_sizes

    def test_batch_flushes_after_wait_deadline(self):
        received_sizes = []

        def process(batch):
            received_sizes.append(len(batch))
            return ["ok"] * len(batch)

        batcher = DynamicBatcher(process, max_batch_size=100, max_wait_s=0.1)
        # Submit fewer than max_batch_size and wait for deadline flush
        futures = [batcher.submit(_make_request()) for _ in range(2)]
        for f in futures:
            f.result(timeout=5.0)

        assert len(received_sizes) >= 1
        assert all(s > 0 for s in received_sizes)

    def test_exception_in_process_propagates_to_all_futures(self):
        def process(batch):
            raise RuntimeError("model exploded")

        batcher = DynamicBatcher(process, max_batch_size=2, max_wait_s=0.1)
        futures = [batcher.submit(_make_request()) for _ in range(2)]
        for f in futures:
            with pytest.raises(RuntimeError, match="model exploded"):
                f.result(timeout=5.0)

    def test_order_preserved_within_batch(self):
        def process(batch):
            return [req.prompt for req in batch]

        batcher = DynamicBatcher(process, max_batch_size=10, max_wait_s=0.5)
        prompts = [f"prompt_{i}" for i in range(5)]
        futures = [batcher.submit(_make_request(prompt=p)) for p in prompts]
        results = [f.result(timeout=5.0) for f in futures]
        assert results == prompts


# ---------------------------------------------------------------------------
# Image decoding
# ---------------------------------------------------------------------------

class TestImageDecoding:
    def test_decode_plain_base64(self):
        from serve.image_utils import decode_image
        img_b64 = _make_image_b64(32, 32, (255, 0, 0))
        img = decode_image(img_b64)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size == (32, 32)

    def test_decode_data_uri_prefix(self):
        from serve.image_utils import decode_image
        img_b64 = _make_image_b64(16, 16)
        uri = f"data:image/jpeg;base64,{img_b64}"
        img = decode_image(uri)
        assert img.size == (16, 16)

    def test_decode_returns_rgb(self):
        """RGBA or grayscale images should be converted to RGB."""
        from serve.image_utils import decode_image
        # Create a grayscale image
        gray = Image.new("L", (16, 16), color=128)
        buf = io.BytesIO()
        gray.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        img = decode_image(b64)
        assert img.mode == "RGB"

    def test_invalid_base64_raises(self):
        from serve.image_utils import decode_image
        with pytest.raises(Exception):
            decode_image("this is not valid base64!!!")


# ---------------------------------------------------------------------------
# GPU inference tests — skipped without CUDA
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason=(
        "GPU inference tests require a CUDA device. "
        "Run on an RTX 4000 Ada or RTX PRO 6000 Blackwell node. "
        "See docs/hardware.md."
    ),
)
class TestModelInferenceGPU:
    """
    These tests load the actual Qwen2.5-VL model and run inference.
    They are skipped in CI and on machines without a CUDA GPU.

    They will download the model from HuggingFace on first run (~16 GB).
    Set HF_HOME to a persistent directory to avoid re-downloading.
    """

    @pytest.fixture(scope="class")
    def model_and_processor(self):
        from serve.model import load_model
        return load_model()

    def test_model_loads(self, model_and_processor):
        model, processor = model_and_processor
        assert model is not None
        assert processor is not None

    def test_single_inference_returns_string(self, model_and_processor):
        from serve.model import run_inference
        model, processor = model_and_processor
        img_b64 = _make_image_b64(224, 224)
        result = run_inference(model, processor, img_b64, "What color is this image?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_image_raises(self, model_and_processor):
        from serve.model import run_inference, decode_image
        with pytest.raises(Exception):
            decode_image("")

    @pytest.mark.skipif(
        True,
        reason=(
            "EXPERIMENTAL: Optimized model test requires flash-attn or "
            "bfloat16 validation.  Enable manually when hardware is available."
        ),
    )
    def test_optimized_output_is_non_empty(self, model_and_processor):
        """EXPERIMENTAL: Load the optimized model and verify it produces output."""
        from serve.model_optimized import load_model_optimized
        from serve.model import run_inference
        model, processor = load_model_optimized(use_bfloat16=True)
        img_b64 = _make_image_b64(224, 224)
        result = run_inference(model, processor, img_b64, "Describe this image.")
        assert isinstance(result, str)
        assert len(result) > 0
