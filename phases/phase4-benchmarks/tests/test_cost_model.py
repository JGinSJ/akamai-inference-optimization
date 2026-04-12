"""
Phase 4 tests — cost model, metrics, and CSV output.

All tests run without a GPU, network connection, or vLLM instance.

Run with:
    cd phases/phase4-benchmarks
    python -m pytest tests/ -v
"""

from __future__ import annotations

import csv
import math
import tempfile
from pathlib import Path

import pytest

from harness.cost_model import (
    CostBreakdown,
    _validate_hourly_price,
    compute_cost,
    write_csvs,
)
from harness.metrics import (
    BenchmarkSummary,
    LatencyPercentiles,
    RequestResult,
    error_rate,
    summarise,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_latency(value: float = 100.0) -> LatencyPercentiles:
    return LatencyPercentiles(
        p50=value, p95=value * 1.5, p99=value * 2.0,
        mean=value, min=value * 0.5, max=value * 3.0,
    )


def _make_summary(
    *,
    tag: str = "test-run",
    gpu_label: str = "RTX 4000 Ada",
    tokens_per_second: float = 100.0,
    mean_tokens_generated: float = 50.0,
    batch_size: int = 4,
    concurrency: int = 4,
    total_requests: int = 10,
    error_count: int = 0,
) -> BenchmarkSummary:
    return BenchmarkSummary(
        tag=tag,
        gpu_label=gpu_label,
        model="test-model",
        batch_size=batch_size,
        concurrency=concurrency,
        total_requests=total_requests,
        error_count=error_count,
        tokens_per_second=tokens_per_second,
        requests_per_second=2.0,
        ttft_ms=_make_latency(50.0),
        e2e_ms=_make_latency(100.0),
        itl_ms=_make_latency(10.0),
        mean_tokens_generated=mean_tokens_generated,
    )


# ---------------------------------------------------------------------------
# _validate_hourly_price
# ---------------------------------------------------------------------------

class TestValidateHourlyPrice:
    def test_valid_float(self):
        assert _validate_hourly_price(1.5, "config.yaml") == pytest.approx(1.5)

    def test_valid_int(self):
        assert _validate_hourly_price(2, "config.yaml") == pytest.approx(2.0)

    def test_valid_string_float(self):
        assert _validate_hourly_price("3.50", "config.yaml") == pytest.approx(3.5)

    def test_placeholder_raises(self):
        with pytest.raises(ValueError, match="PLACEHOLDER"):
            _validate_hourly_price("PLACEHOLDER", "config.yaml")

    def test_none_raises(self):
        with pytest.raises(ValueError):
            _validate_hourly_price(None, "config.yaml")

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="> 0"):
            _validate_hourly_price(0, "config.yaml")

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="> 0"):
            _validate_hourly_price(-1.0, "config.yaml")


# ---------------------------------------------------------------------------
# compute_cost
# ---------------------------------------------------------------------------

class TestComputeCost:
    def test_basic_arithmetic(self):
        """
        At 100 tok/s and $1.00/hr the cost per token should be
        1.00 / 3600 / 100 = 2.778e-6 USD.
        """
        summary = _make_summary(tokens_per_second=100.0, mean_tokens_generated=50.0)
        cost = compute_cost(summary, gpu_hourly_usd=1.0)

        expected_per_token = 1.0 / 3600 / 100
        assert cost.cost_per_token_usd == pytest.approx(expected_per_token, rel=1e-6)

    def test_cost_per_request(self):
        """cost_per_request = cost_per_token * mean_tokens_generated"""
        summary = _make_summary(tokens_per_second=100.0, mean_tokens_generated=200.0)
        cost = compute_cost(summary, gpu_hourly_usd=1.0)
        expected_per_token = 1.0 / 3600 / 100
        assert cost.cost_per_request_usd == pytest.approx(expected_per_token * 200.0, rel=1e-6)

    def test_cost_per_million_tokens(self):
        summary = _make_summary(tokens_per_second=100.0)
        cost = compute_cost(summary, gpu_hourly_usd=1.0)
        assert cost.cost_per_million_tokens_usd == pytest.approx(
            cost.cost_per_token_usd * 1_000_000, rel=1e-6
        )

    def test_higher_throughput_lower_cost(self):
        """Double the throughput should halve the cost per token."""
        s1 = _make_summary(tokens_per_second=100.0)
        s2 = _make_summary(tokens_per_second=200.0)
        c1 = compute_cost(s1, gpu_hourly_usd=1.0)
        c2 = compute_cost(s2, gpu_hourly_usd=1.0)
        assert c1.cost_per_token_usd == pytest.approx(c2.cost_per_token_usd * 2, rel=1e-6)

    def test_higher_hourly_higher_cost(self):
        """Double the hourly price should double the cost per token."""
        summary = _make_summary(tokens_per_second=100.0)
        c1 = compute_cost(summary, gpu_hourly_usd=1.0)
        c2 = compute_cost(summary, gpu_hourly_usd=2.0)
        assert c2.cost_per_token_usd == pytest.approx(c1.cost_per_token_usd * 2, rel=1e-6)

    def test_zero_throughput_raises(self):
        summary = _make_summary(tokens_per_second=0.0)
        with pytest.raises(ValueError, match="tokens_per_second"):
            compute_cost(summary, gpu_hourly_usd=1.0)

    def test_returns_cost_breakdown(self):
        summary = _make_summary()
        result = compute_cost(summary, gpu_hourly_usd=1.0)
        assert isinstance(result, CostBreakdown)
        assert result.gpu_hourly_usd == 1.0


# ---------------------------------------------------------------------------
# write_csvs
# ---------------------------------------------------------------------------

class TestWriteCsvs:
    def _read_csv(self, path: Path) -> list[dict]:
        with path.open() as f:
            return list(csv.DictReader(f))

    def test_creates_three_csv_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            summaries = [
                _make_summary(tag="run-a", batch_size=4, concurrency=1),
                _make_summary(tag="run-b", batch_size=8, concurrency=4),
            ]
            write_csvs(summaries, 1.0, out)
            assert (out / "comparison.csv").exists()
            assert (out / "cost_by_batch.csv").exists()
            assert (out / "cost_by_concurrency.csv").exists()

    def test_comparison_csv_row_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            summaries = [_make_summary(tag=f"run-{i}") for i in range(5)]
            write_csvs(summaries, 1.0, out)
            rows = self._read_csv(out / "comparison.csv")
            assert len(rows) == 5

    def test_csv_has_required_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_csvs([_make_summary()], 1.0, out)
            rows = self._read_csv(out / "comparison.csv")
            required = {
                "tag", "gpu_label", "tokens_per_second",
                "cost_per_token_usd", "cost_per_million_tokens_usd",
                "e2e_p50_ms", "e2e_p95_ms", "e2e_p99_ms",
            }
            assert required.issubset(rows[0].keys())

    def test_cost_values_are_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_csvs([_make_summary(tokens_per_second=500.0)], 2.0, out)
            rows = self._read_csv(out / "comparison.csv")
            assert float(rows[0]["cost_per_token_usd"]) > 0
            assert float(rows[0]["cost_per_million_tokens_usd"]) > 0

    def test_empty_summaries_produces_no_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_csvs([], 1.0, out)
            assert not (out / "comparison.csv").exists()

    def test_cost_by_batch_sorted_by_batch_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            summaries = [
                _make_summary(tag="run-c16", batch_size=16),
                _make_summary(tag="run-c4", batch_size=4),
                _make_summary(tag="run-c8", batch_size=8),
            ]
            write_csvs(summaries, 1.0, out)
            rows = self._read_csv(out / "cost_by_batch.csv")
            sizes = [int(r["batch_size"]) for r in rows]
            assert sizes == sorted(sizes)

    def test_cost_by_concurrency_sorted_by_concurrency(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            summaries = [
                _make_summary(tag="run-c16", concurrency=16),
                _make_summary(tag="run-c1", concurrency=1),
                _make_summary(tag="run-c4", concurrency=4),
            ]
            write_csvs(summaries, 1.0, out)
            rows = self._read_csv(out / "cost_by_concurrency.csv")
            levels = [int(r["concurrency"]) for r in rows]
            assert levels == sorted(levels)


# ---------------------------------------------------------------------------
# metrics.summarise
# ---------------------------------------------------------------------------

class TestSummarise:
    def _make_results(self, n: int, tokens: int = 50) -> list[RequestResult]:
        return [
            RequestResult(ttft_s=0.05, e2e_s=0.5, tokens_generated=tokens)
            for _ in range(n)
        ]

    def test_basic_summary(self):
        results = self._make_results(10, tokens=100)
        s = summarise(
            results,
            tag="t",
            gpu_label="GPU",
            model="m",
            batch_size=4,
            concurrency=2,
            wall_time_s=5.0,
        )
        assert s.total_requests == 10
        assert s.error_count == 0
        assert s.tokens_per_second == pytest.approx(1000 / 5.0, rel=1e-5)

    def test_error_count(self):
        results = self._make_results(8)
        results += [RequestResult(ttft_s=None, e2e_s=0.1, tokens_generated=0, error="timeout")]
        results += [RequestResult(ttft_s=None, e2e_s=0.1, tokens_generated=0, error="HTTP 500")]
        s = summarise(
            results,
            tag="t",
            gpu_label="GPU",
            model="m",
            batch_size=1,
            concurrency=1,
            wall_time_s=1.0,
        )
        assert s.error_count == 2
        assert s.total_requests == 10

    def test_all_errors_raises(self):
        results = [
            RequestResult(ttft_s=None, e2e_s=0.1, tokens_generated=0, error="fail")
            for _ in range(5)
        ]
        with pytest.raises(ValueError, match="No successful requests"):
            summarise(
                results,
                tag="t",
                gpu_label="GPU",
                model="m",
                batch_size=1,
                concurrency=1,
                wall_time_s=1.0,
            )

    def test_ttft_percentiles_present_when_ttft_recorded(self):
        results = self._make_results(10)
        s = summarise(
            results,
            tag="t",
            gpu_label="GPU",
            model="m",
            batch_size=1,
            concurrency=1,
            wall_time_s=5.0,
        )
        assert s.ttft_ms is not None
        assert s.ttft_ms.p50 > 0

    def test_ttft_percentiles_none_when_ttft_not_recorded(self):
        results = [
            RequestResult(ttft_s=None, e2e_s=0.5, tokens_generated=50)
            for _ in range(5)
        ]
        s = summarise(
            results,
            tag="t",
            gpu_label="GPU",
            model="m",
            batch_size=1,
            concurrency=1,
            wall_time_s=2.5,
        )
        assert s.ttft_ms is None


# ---------------------------------------------------------------------------
# error_rate
# ---------------------------------------------------------------------------

class TestErrorRate:
    def test_no_errors(self):
        s = _make_summary(total_requests=10, error_count=0)
        assert error_rate(s) == pytest.approx(0.0)

    def test_all_errors(self):
        s = _make_summary(total_requests=10, error_count=10)
        assert error_rate(s) == pytest.approx(1.0)

    def test_partial_errors(self):
        s = _make_summary(total_requests=10, error_count=3)
        assert error_rate(s) == pytest.approx(0.3)

    def test_zero_requests(self):
        s = _make_summary(total_requests=0, error_count=0)
        assert error_rate(s) == pytest.approx(0.0)
