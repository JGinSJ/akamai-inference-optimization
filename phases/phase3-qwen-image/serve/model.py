"""
Baseline Qwen-Image inference.

This is the reference implementation: clean, unmodified defaults with no
optimisation flags set.  Every benchmark in Phase 3 uses this as the
starting point.  Do not add optimisations here — put them in
model_optimized.py so the two code paths stay clearly separated.

Model family
------------
Qwen2.5-VL (vision-language).  The default variant is 7B; change
MODEL_NAME to use a smaller or larger checkpoint.  VRAM requirements
are model-dependent — see docs/hardware.md once GPU specs are confirmed.

    Variant                              Approx VRAM (FP16)
    Qwen/Qwen2.5-VL-3B-Instruct         ~  8 GB
    Qwen/Qwen2.5-VL-7B-Instruct         ~ 16 GB   ← default
    Qwen/Qwen2.5-VL-72B-Instruct        ~144 GB

TODO: Confirm variant once docs/hardware.md GPU VRAM figures are filled in.
"""

import logging
import os
from typing import Optional

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from .image_utils import decode_image  # noqa: F401 — re-exported for callers

log = logging.getLogger(__name__)

# TODO: Update once GPU VRAM is confirmed in docs/hardware.md.
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"


def load_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> tuple:
    """
    Load the Qwen2.5-VL model and processor.

    Uses float16 precision, standard (eager) attention, no torch.compile.
    This is the unmodified baseline — suitable for correctness testing and
    as a performance reference.

    Parameters
    ----------
    model_name : HuggingFace hub ID or local directory path.
                 Falls back to MODEL_NAME env var, then DEFAULT_MODEL_NAME.
    device     : "cuda", "cpu", or None (auto-selects cuda if available).

    Returns
    -------
    (model, processor) — both ready for run_inference().
    """
    model_name = model_name or os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info(
        "Loading baseline model: %s | device=%s | dtype=float16",
        model_name,
        device,
    )

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    log.info("Model loaded. Parameters: %.2fB | device: %s", param_count, device)

    return model, processor


def run_inference(
    model,
    processor,
    image_b64: str,
    prompt: str,
    max_new_tokens: int = 256,
) -> str:
    """
    Run a single image + text inference request.

    This function is shared by both the baseline and optimized serving
    paths — the only difference between the two paths is in how the model
    was loaded (load_model vs load_model_optimized).

    Parameters
    ----------
    model          : loaded model (from load_model or load_model_optimized)
    processor      : loaded AutoProcessor
    image_b64      : base64-encoded image bytes (JPEG or PNG).
                     A data-URI prefix (data:image/jpeg;base64,...) is
                     stripped automatically.
    prompt         : text instruction for the model
    max_new_tokens : maximum number of tokens to generate

    Returns
    -------
    str — generated text, with the input prompt tokens stripped.
    """
    image = decode_image(image_b64)

    # Build the chat-format message expected by Qwen2.5-VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decoding — deterministic output
        )

    # Decode only the newly generated tokens, not the input
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][input_len:]
    return processor.decode(generated_ids, skip_special_tokens=True)
