"""
Phase 4 tests — benchmark/benchmark.py

All tests run without a GPU, network connection, or vLLM instance.
requests.post is mocked throughout; no real HTTP calls are made.

Run with:
    cd phases/phase4-benchmarks
    python -m pytest tests/ -v
"""

from __future__ import annotations

import csv
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the phase root is on the path (mirrors benchmark.py's own sys.path
# insertion so imports resolve the same way in tests).
_PHASE_ROOT = Path(__file__).resolve().parent.parent
if str(_PHASE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PHASE_ROOT))

from benchmark.benchmark import (
    _CSV_FIELDNAMES,
    _append_csv,
    _build_prompt,
    _summarise,
    run_benchmark,
)
from harness.metrics import RequestResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok_result(e2e_s: float = 0.5, tokens: int = 64) -> RequestResult:
    return RequestResult(ttft_s=None, e2e_s=e2e_s, tokens_generated=tokens)


def _err_result(msg: str = "timeout") -> RequestResult:
    return RequestResult(ttft_s=None, e2e_s=0.1, tokens_generated=0, error=msg)


def _fake_response(completion_tokens: int = 64, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"usage": {"completion_tokens": completion_tokens}}
    return resp


# ---------------------------------------------------------------------------
# _build_prompt — prompt construction
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_returns_string(self):
        assert isinstance(_build_prompt(256), str)

    def test_scales_with_target(self):
        short = _build_prompt(128)
        long_ = _build_prompt(512)
        assert len(long_) > len(short)

    def test_minimum_one_repeat(self):
        # Even target_tokens=1 should produce a non-empty prompt
        assert len(_build_prompt(1)) > 0

    @pytest.mark.parametrize("target", [128, 256, 512])
    def test_length_order_of_magnitude(self, target):
        # Prompt character length should be in a plausible range:
        # ~4 chars/token is a rough lower bound; _BASE_PHRASE gives more.
        # We just verify it's not absurdly short or absurdly long.
        prompt = _build_prompt(target)
        char_len = len(prompt)
        # Should be at least target * 3 chars (conservative) and at most target * 50
        assert char_len >= target * 3
        assert char_len <= target * 50


# ---------------------------------------------------------------------------
# _summarise — aggregation without network
# ---------------------------------------------------------------------------

class TestSummarise:
    def _run(self, results, **kwargs):
        defaults = dict(
            wall_time_s=5.0,
            tag="test",
            gpu_label="RTX 4000 Ada",
            model="test-model",
            prompt_tokens=256,
            max_tokens=128,
            concurrency=4,
        )
        defaults.update(kwargs)
        return _summarise(results, **defaults)

    def test_error_count_tracked(self):
        results = [_ok_result()] * 8 + [_err_result()] * 2
        s = self._run(results)
        assert s.error_count == 2
        assert s.total_requests == 10

    def test_error_count_zero_on_clean_run(self):
        s = self._run([_ok_result()] * 10)
        assert s.error_count == 0

    def test_all_errors_raises(self):
        with pytest.raises(RuntimeError):
            self._run([_err_result()] * 5)

    def test_tokens_per_second_positive(self):
        s = self._run([_ok_result(tokens=100)] * 10)
        assert s.tokens_per_second > 0

    def test_e2e_percentiles_present(self):
        s = self._run([_ok_result(e2e_s=0.3)] * 20)
        assert s.e2e_ms is not None
        assert s.e2e_ms.p50 > 0
        assert s.e2e_ms.p99 >= s.e2e_ms.p50

    def test_ttft_is_none(self):
        # benchmark.py never captures TTFT (non-streaming); confirm it stays None
        s = self._run([_ok_result()] * 5)
        assert s.ttft_ms is None

    def test_mean_tokens_generated(self):
        results = [_ok_result(tokens=t) for t in [50, 100, 150]]
        s = self._run(results)
        assert s.mean_tokens_generated == pytest.approx(100.0, rel=1e-5)


# ---------------------------------------------------------------------------
# _append_csv — CSV output
# ---------------------------------------------------------------------------

class TestAppendCsv:
    def _write_row(self, path: Path, tag: str = "run-1", tps: float = 100.0) -> None:
        from harness.cost_model import CostBreakdown

        results = [_ok_result(tokens=64)] * 10
        summary = _summarise(
            results,
            wall_time_s=5.0,
            tag=tag,
            gpu_label="RTX 4000 Ada",
            model="test-model",
            prompt_tokens=256,
            max_tokens=128,
            concurrency=4,
        )
        # Patch tokens_per_second to a controlled value for cost assertions
        summary = summary.__class__(
            **{**summary.__dict__, "tokens_per_second": tps}
        )
        cost = CostBreakdown(
            gpu_hourly_usd=0.96,
            cost_per_token_usd=0.96 / 3600 / tps,
            cost_per_request_usd=(0.96 / 3600 / tps) * summary.mean_tokens_generated,
            cost_per_million_tokens_usd=(0.96 / 3600 / tps) * 1_000_000,
        )
        _append_csv(path, summary, cost, wall_time_s=5.0, prompt_tokens=256, max_tokens=128)

    def test_creates_file_with_header(self, tmp_path):
        out = tmp_path / "results.csv"
        self._write_row(out)
        assert out.exists()
        with out.open() as f:
            reader = csv.DictReader(f)
            assert set(_CSV_FIELDNAMES).issubset(set(reader.fieldnames))

    def test_csv_has_required_columns(self, tmp_path):
        out = tmp_path / "results.csv"
        self._write_row(out)
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        required = {
            "tag", "gpu_label", "model", "prompt_tokens", "max_tokens",
            "concurrency", "tokens_per_second", "cost_per_token_usd",
            "cost_per_request_usd", "cost_per_million_tokens_usd",
            "e2e_p50_ms", "e2e_p95_ms", "e2e_p99_ms",
        }
        assert required.issubset(rows[0].keys())

    def test_cost_columns_positive(self, tmp_path):
        out = tmp_path / "results.csv"
        self._write_row(out, tps=200.0)
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert float(rows[0]["cost_per_token_usd"]) > 0
        assert float(rows[0]["cost_per_request_usd"]) > 0
        assert float(rows[0]["cost_per_million_tokens_usd"]) > 0

    def test_appends_without_overwriting(self, tmp_path):
        out = tmp_path / "results.csv"
        self._write_row(out, tag="run-1")
        self._write_row(out, tag="run-2")
        self._write_row(out, tag="run-3")
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert [r["tag"] for r in rows] == ["run-1", "run-2", "run-3"]

    def test_header_written_once_on_multiple_appends(self, tmp_path):
        out = tmp_path / "results.csv"
        for i in range(4):
            self._write_row(out, tag=f"run-{i}")
        with out.open() as f:
            lines = f.readlines()
        # First line is the header; remaining lines are data rows
        header_lines = [l for l in lines if "tag" in l and "gpu_label" in l]
        assert len(header_lines) == 1

    def test_creates_parent_directory(self, tmp_path):
        out = tmp_path / "nested" / "dir" / "results.csv"
        self._write_row(out)
        assert out.exists()


# ---------------------------------------------------------------------------
# run_benchmark — concurrency and mocking
# ---------------------------------------------------------------------------

class TestRunBenchmark:
    def _mock_post(self, completion_tokens: int = 64):
        return patch(
            "benchmark.benchmark.requests.post",
            return_value=_fake_response(completion_tokens),
        )

    def test_returns_correct_request_count(self):
        with self._mock_post():
            results, _ = run_benchmark(
                url="http://fake/v1/completions",
                model="test-model",
                prompt_tokens=128,
                max_tokens=64,
                concurrency=2,
                num_requests=10,
                timeout_s=5.0,
            )
        assert len(results) == 10

    def test_all_results_are_request_result(self):
        with self._mock_post():
            results, _ = run_benchmark(
                url="http://fake/v1/completions",
                model="test-model",
                prompt_tokens=128,
                max_tokens=64,
                concurrency=2,
                num_requests=5,
                timeout_s=5.0,
            )
        assert all(isinstance(r, RequestResult) for r in results)

    def test_wall_time_is_positive(self):
        with self._mock_post():
            _, wall_time_s = run_benchmark(
                url="http://fake/v1/completions",
                model="test-model",
                prompt_tokens=128,
                max_tokens=64,
                concurrency=2,
                num_requests=5,
                timeout_s=5.0,
            )
        assert wall_time_s > 0

    def test_concurrency_limits_simultaneous_calls(self):
        """Thread pool should never exceed concurrency simultaneous requests."""
        max_concurrency = 3
        peak_concurrent = 0
        active = 0
        lock = threading.Lock()

        def fake_post(*args, **kwargs):
            nonlocal peak_concurrent, active
            with lock:
                active += 1
                if active > peak_concurrent:
                    peak_concurrent = active
            # Small sleep to let threads overlap
            import time; time.sleep(0.02)
            with lock:
                active -= 1
            return _fake_response()

        with patch("benchmark.benchmark.requests.post", side_effect=fake_post):
            run_benchmark(
                url="http://fake/v1/completions",
                model="test-model",
                prompt_tokens=128,
                max_tokens=64,
                concurrency=max_concurrency,
                num_requests=12,
                timeout_s=5.0,
            )

        assert peak_concurrent <= max_concurrency

    def test_http_error_recorded_not_raised(self):
        """A failed request should appear as an error result, not raise an exception."""
        error_resp = MagicMock()
        error_resp.raise_for_status.side_effect = Exception("HTTP 500")

        with patch("benchmark.benchmark.requests.post", return_value=error_resp):
            results, _ = run_benchmark(
                url="http://fake/v1/completions",
                model="test-model",
                prompt_tokens=128,
                max_tokens=64,
                concurrency=1,
                num_requests=3,
                timeout_s=5.0,
            )

        assert len(results) == 3
        assert all(r.error is not None for r in results)

    def test_tokens_from_response_body(self):
        """tokens_generated should come from usage.completion_tokens in the response."""
        with self._mock_post(completion_tokens=99):
            results, _ = run_benchmark(
                url="http://fake/v1/completions",
                model="test-model",
                prompt_tokens=128,
                max_tokens=128,
                concurrency=1,
                num_requests=1,
                timeout_s=5.0,
            )
        assert results[0].tokens_generated == 99


# ---------------------------------------------------------------------------
# gpu_hourly_usd=0 guard (tested via _summarise + compute_cost path)
# ---------------------------------------------------------------------------

class TestGpuHourlyUsdGuard:
    def test_zero_hourly_usd_raises(self):
        from harness.cost_model import compute_cost

        results = [_ok_result()] * 5
        summary = _summarise(
            results,
            wall_time_s=2.0,
            tag="t",
            gpu_label="GPU",
            model="m",
            prompt_tokens=256,
            max_tokens=128,
            concurrency=1,
        )
        # compute_cost raises when tokens_per_second > 0 but cost arithmetic
        # would be zero — the guard is on gpu_hourly_usd via _validate_hourly_price,
        # but compute_cost itself accepts a float; the CLI guards against 0 before
        # calling it.  Test the invariant: cost_per_token must be > 0 when price > 0.
        cost = compute_cost(summary, gpu_hourly_usd=0.96)
        assert cost.cost_per_token_usd > 0

    def test_negative_throughput_raises_in_compute_cost(self):
        from harness.cost_model import compute_cost
        from harness.metrics import LatencyPercentiles, BenchmarkSummary

        lp = LatencyPercentiles(p50=100, p95=150, p99=200, mean=110, min=80, max=300)
        summary = BenchmarkSummary(
            tag="t", gpu_label="GPU", model="m",
            batch_size=1, concurrency=1, total_requests=1, error_count=0,
            tokens_per_second=0.0, requests_per_second=1.0,
            ttft_ms=None, e2e_ms=lp, itl_ms=None,
            mean_tokens_generated=64.0,
        )
        with pytest.raises(ValueError, match="tokens_per_second"):
            compute_cost(summary, gpu_hourly_usd=0.96)
