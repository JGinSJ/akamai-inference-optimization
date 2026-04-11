"""
Phase 3 — FastAPI serving wrapper for Qwen-Image.

POST /v1/generate   {"image": "<base64>", "prompt": "..."}
GET  /health

Environment variables
---------------------
MODEL_NAME        HuggingFace model ID or local path.
                  Default: Qwen/Qwen2.5-VL-7B-Instruct
                  TODO: Update once VRAM is confirmed in docs/hardware.md.

USE_OPTIMIZED     "1" to load the EXPERIMENTAL optimized model.
                  Default: "0" (baseline).
USE_FLASH_ATTN    "1" to enable EXPERIMENTAL Flash Attention 2.
USE_BFLOAT16      "1" to enable EXPERIMENTAL bfloat16 dtype.
USE_TORCH_COMPILE "1" to enable EXPERIMENTAL torch.compile.

MAX_NEW_TOKENS    Maximum tokens to generate per request. Default: 256.
MAX_BATCH_SIZE    Dynamic batching: max requests per batch. Default: 1.
MAX_WAIT_S        Dynamic batching: max seconds to wait for a full batch.
                  Default: 0.05.

Usage
-----
    cd phases/phase3-qwen-image
    pip install -r requirements.txt
    uvicorn serve.app:app --host 0.0.0.0 --port 8080

To run the optimized (EXPERIMENTAL) path:
    USE_OPTIMIZED=1 USE_FLASH_ATTN=1 uvicorn serve.app:app --port 8080
"""

import logging
import os
import time

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .batching import DynamicBatcher, InferenceRequest
from .model import load_model, run_inference
from .model_optimized import load_model_optimized

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")
USE_OPTIMIZED = os.environ.get("USE_OPTIMIZED", "0") == "1"
USE_FLASH_ATTN = os.environ.get("USE_FLASH_ATTN", "0") == "1"
USE_BFLOAT16 = os.environ.get("USE_BFLOAT16", "0") == "1"
USE_TORCH_COMPILE = os.environ.get("USE_TORCH_COMPILE", "0") == "1"
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "1"))
MAX_WAIT_S = float(os.environ.get("MAX_WAIT_S", "0.05"))

# Populated at startup
_model = None
_processor = None
_batcher: DynamicBatcher = None

GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Qwen-Image Inference — Phase 3",
    description="Akamai LKE deployment of Qwen2.5-VL for image-language inference.",
    version="0.1.0",
)


@app.on_event("startup")
def _startup() -> None:
    global _model, _processor, _batcher

    if USE_OPTIMIZED:
        log.info(
            "EXPERIMENTAL optimized model requested. "
            "flash_attn=%s bfloat16=%s torch_compile=%s",
            USE_FLASH_ATTN,
            USE_BFLOAT16,
            USE_TORCH_COMPILE,
        )
        _model, _processor = load_model_optimized(
            model_name=MODEL_NAME,
            use_flash_attention=USE_FLASH_ATTN,
            use_bfloat16=USE_BFLOAT16,
            use_torch_compile=USE_TORCH_COMPILE,
        )
    else:
        _model, _processor = load_model(model_name=MODEL_NAME)

    def _process_batch(batch: list) -> list:
        # NOTE: requests within a batch are processed sequentially.
        # True multi-image batched inference (padding variable-size vision
        # tokens to a uniform length) is a TODO.
        return [
            run_inference(
                _model, _processor,
                req.image_b64, req.prompt, req.max_new_tokens,
            )
            for req in batch
        ]

    _batcher = DynamicBatcher(
        process_batch=_process_batch,
        max_batch_size=MAX_BATCH_SIZE,
        max_wait_s=MAX_WAIT_S,
    )
    log.info(
        "Server ready. GPU=%s | model=%s | optimized=%s | "
        "max_batch=%d | max_wait=%.3fs",
        GPU_NAME, MODEL_NAME, USE_OPTIMIZED, MAX_BATCH_SIZE, MAX_WAIT_S,
    )


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    image: str          # base64-encoded image (JPEG or PNG)
    prompt: str
    max_new_tokens: int = MAX_NEW_TOKENS


class GenerateResponse(BaseModel):
    response: str
    latency_ms: float
    gpu: str
    optimized: bool     # True if loaded via load_model_optimized()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/v1/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    if not req.image:
        raise HTTPException(status_code=400, detail="'image' must not be empty")
    if not req.prompt:
        raise HTTPException(status_code=400, detail="'prompt' must not be empty")
    if req.max_new_tokens < 1:
        raise HTTPException(status_code=400, detail="'max_new_tokens' must be >= 1")

    inference_req = InferenceRequest(
        image_b64=req.image,
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
    )

    t0 = time.perf_counter()
    future = _batcher.submit(inference_req)

    try:
        response_text = future.result(timeout=120.0)
    except TimeoutError:
        raise HTTPException(status_code=504, detail="inference timed out after 120s")
    except Exception as exc:
        log.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"inference error: {exc}")

    latency_ms = (time.perf_counter() - t0) * 1000

    return GenerateResponse(
        response=response_text,
        latency_ms=round(latency_ms, 2),
        gpu=GPU_NAME,
        optimized=USE_OPTIMIZED,
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "gpu": GPU_NAME,
        "model": MODEL_NAME,
        "optimized": USE_OPTIMIZED,
    }
