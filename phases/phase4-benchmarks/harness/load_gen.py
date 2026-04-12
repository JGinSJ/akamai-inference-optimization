"""
Async load generator for vLLM OpenAI-compatible endpoints.

Sends concurrent requests to POST /v1/completions using Server-Sent Events
(SSE) streaming so that time-to-first-token can be measured per request.

Usage (standalone)
------------------
    python -m harness.load_gen \
        --url http://localhost:8000/v1/completions \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --concurrency 4 \
        --num-requests 40 \
        --max-tokens 128 \
        --output results/run_01.json \
        --tag ada-c4-b128

Output JSON schema
------------------
{
  "tag": "ada-c4-b128",
  "model": "...",
  "concurrency": 4,
  "num_requests": 40,
  "max_tokens": 128,
  "wall_time_s": 12.34,
  "results": [
    {
      "ttft_s": 0.082,
      "e2e_s": 1.234,
      "tokens_generated": 128,
      "prompt_tokens": 32,
      "error": null
    },
    ...
  ]
}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from .metrics import RequestResult

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic prompt pool
# ---------------------------------------------------------------------------

_PROMPTS = [
    "Explain the difference between latency and throughput in distributed systems.",
    "What are the trade-offs between model quantization and inference accuracy?",
    "Describe how KV caching reduces compute in autoregressive language models.",
    "What is tensor parallelism and how does it differ from pipeline parallelism?",
    "Explain the prefill and decode phases of transformer inference.",
    "What factors determine the optimal batch size for GPU inference?",
    "How does flash attention reduce memory usage during self-attention computation?",
    "Describe the relationship between GPU memory bandwidth and token generation speed.",
]


def _get_prompt(index: int) -> str:
    return _PROMPTS[index % len(_PROMPTS)]


# ---------------------------------------------------------------------------
# Single-request coroutine
# ---------------------------------------------------------------------------

async def _send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    request_timeout_s: float,
) -> RequestResult:
    """
    Send one streaming request and record TTFT + E2E latency.

    The vLLM streaming response is a sequence of SSE lines:
        data: {"choices": [{"text": "...", ...}], "usage": {...}}
        ...
        data: [DONE]

    We record the wall time at the first non-empty "text" chunk as TTFT.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,  # greedy — deterministic output
    }

    t_start = time.perf_counter()
    ttft_s: Optional[float] = None
    tokens_generated = 0
    prompt_tokens: Optional[int] = None

    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=request_timeout_s),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return RequestResult(
                    ttft_s=None,
                    e2e_s=time.perf_counter() - t_start,
                    tokens_generated=0,
                    error=f"HTTP {resp.status}: {body[:200]}",
                )

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload_str = line[len("data:"):].strip()
                if payload_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue

                text = chunk.get("choices", [{}])[0].get("text", "")
                if text and ttft_s is None:
                    ttft_s = time.perf_counter() - t_start

                # vLLM reports usage on the final chunk
                usage = chunk.get("usage")
                if usage:
                    tokens_generated = usage.get("completion_tokens", tokens_generated)
                    prompt_tokens = usage.get("prompt_tokens")
                elif text:
                    # Approximate: count non-empty chunks as tokens
                    tokens_generated += 1

    except asyncio.TimeoutError:
        return RequestResult(
            ttft_s=ttft_s,
            e2e_s=time.perf_counter() - t_start,
            tokens_generated=tokens_generated,
            error="timeout",
        )
    except aiohttp.ClientError as exc:
        return RequestResult(
            ttft_s=ttft_s,
            e2e_s=time.perf_counter() - t_start,
            tokens_generated=tokens_generated,
            error=str(exc),
        )

    e2e_s = time.perf_counter() - t_start
    return RequestResult(
        ttft_s=ttft_s,
        e2e_s=e2e_s,
        tokens_generated=tokens_generated,
        prompt_tokens=prompt_tokens,
    )


# ---------------------------------------------------------------------------
# Concurrency harness
# ---------------------------------------------------------------------------

async def run(
    url: str,
    model: str,
    num_requests: int,
    concurrency: int,
    max_tokens: int,
    request_timeout_s: float = 120.0,
) -> tuple[List[RequestResult], float]:
    """
    Drive `num_requests` total requests at `concurrency` simultaneous workers.

    Returns (results, wall_time_s).
    """
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 4)

    async def bounded(index: int) -> RequestResult:
        async with semaphore:
            prompt = _get_prompt(index)
            return await _send_request(
                session, url, model, prompt, max_tokens, request_timeout_s
            )

    t_wall_start = time.perf_counter()
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [asyncio.create_task(bounded(i)) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
    wall_time_s = time.perf_counter() - t_wall_start

    return list(results), wall_time_s


# ---------------------------------------------------------------------------
# Result serialisation
# ---------------------------------------------------------------------------

def _result_to_dict(r: RequestResult) -> dict:
    return {
        "ttft_s": r.ttft_s,
        "e2e_s": r.e2e_s,
        "tokens_generated": r.tokens_generated,
        "prompt_tokens": r.prompt_tokens,
        "error": r.error,
    }


def save_results(
    output_path: Path,
    *,
    tag: str,
    model: str,
    concurrency: int,
    num_requests: int,
    max_tokens: int,
    wall_time_s: float,
    results: List[RequestResult],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag": tag,
        "model": model,
        "concurrency": concurrency,
        "num_requests": num_requests,
        "max_tokens": max_tokens,
        "wall_time_s": wall_time_s,
        "results": [_result_to_dict(r) for r in results],
    }
    output_path.write_text(json.dumps(payload, indent=2))
    log.info("Results saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="vLLM async load generator")
    p.add_argument("--url", required=True, help="vLLM completions endpoint URL")
    p.add_argument("--model", required=True, help="Model ID (must match vLLM --model)")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--num-requests", type=int, default=40)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--request-timeout", type=float, default=120.0)
    p.add_argument("--output", required=True, help="Path to write results JSON")
    p.add_argument("--tag", required=True, help="Run label embedded in the output")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    log.info(
        "Starting load gen: url=%s concurrency=%d requests=%d max_tokens=%d",
        args.url,
        args.concurrency,
        args.num_requests,
        args.max_tokens,
    )

    results, wall_time_s = asyncio.run(
        run(
            url=args.url,
            model=args.model,
            num_requests=args.num_requests,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            request_timeout_s=args.request_timeout,
        )
    )

    errors = sum(1 for r in results if r.error)
    log.info(
        "Done. wall_time=%.2fs  requests=%d  errors=%d",
        wall_time_s,
        len(results),
        errors,
    )

    save_results(
        Path(args.output),
        tag=args.tag,
        model=args.model,
        concurrency=args.concurrency,
        num_requests=args.num_requests,
        max_tokens=args.max_tokens,
        wall_time_s=wall_time_s,
        results=results,
    )


if __name__ == "__main__":
    main()
