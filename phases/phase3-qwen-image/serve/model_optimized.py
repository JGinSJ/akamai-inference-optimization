"""
EXPERIMENTAL — Qwen-Image optimized model loading.

Every optimization in this file is explicitly labelled EXPERIMENTAL.

What "EXPERIMENTAL" means here
-------------------------------
- The effect is hardware- and workload-dependent.  An optimization that
  helps on RTX PRO 6000 Blackwell may be neutral or harmful on RTX 4000 Ada.
- No throughput or latency improvement is claimed or implied.
- Some combinations may produce numerically different (but semantically
  equivalent) outputs compared to the baseline.
- flash-attn requires a separate installation step not in requirements.txt.

How to measure
--------------
Run benchmark/load_gen.py with USE_OPTIMIZED=0, record
results/phase3_baseline.json, then run again with USE_OPTIMIZED=1 and the
relevant flags, record results/phase3_optimized.json, then compare with
benchmark/report.py.  Let measured numbers speak — do not assume.

run_inference() contract
------------------------
The run_inference() function from model.py is used unchanged with both the
baseline and optimized models.  This ensures the two paths are compared
fairly and that output correctness can be verified by comparing generations.
"""

import logging
import os
from typing import Optional

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from .model import DEFAULT_MODEL_NAME

log = logging.getLogger(__name__)


def load_model_optimized(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    use_flash_attention: bool = False,
    use_bfloat16: bool = False,
    use_torch_compile: bool = False,
) -> tuple:
    """
    Load the Qwen2.5-VL model with EXPERIMENTAL optimizations.

    Each optimization is independently opt-in.  Start with all flags False
    and enable one at a time to isolate the effect of each on your hardware.

    Parameters
    ----------
    model_name         : HuggingFace hub ID or local path.
    device             : "cuda", "cpu", or None (auto-detect).

    use_flash_attention : EXPERIMENTAL
        Enable Flash Attention 2 (attn_implementation="flash_attention_2").

        Mechanism: replaces the standard O(n²) attention kernel with a
        memory-efficient fused CUDA kernel that avoids materialising the
        full attention matrix.

        Requirements:
          pip install flash-attn --no-build-isolation
          CUDA compute capability ≥ 8.0 (Ampere / Ada / Hopper / Blackwell)

        Possible effects: reduced GPU memory usage for long sequences; may
        change throughput.  Measure before concluding.

    use_bfloat16 : EXPERIMENTAL
        Use bfloat16 instead of float16 as the model dtype.

        Mechanism: bfloat16 has the same exponent range as float32 but fewer
        mantissa bits than float16.  This can reduce overflow/underflow on
        operations with large dynamic range.

        Possible effects: Blackwell GPUs have native bfloat16 tensor cores;
        behaviour on Ada is hardware-dependent.  Measure before concluding.

    use_torch_compile : EXPERIMENTAL
        Apply torch.compile(model, mode="reduce-overhead") after loading.

        Mechanism: traces the model's compute graph and emits optimised
        machine code for repeated same-shape inputs.  Typical in batch
        serving where request shapes are stable.

        Possible effects: reduced per-call overhead after a warm-up period.
        First call incurs compilation time (can be tens of seconds).
        Not all Qwen2.5-VL operators may be compilable; graph breaks are
        logged as warnings.  CPU devices are unsupported.

    Returns
    -------
    (model, processor)
    """
    model_name = model_name or os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # EXPERIMENTAL: dtype selection
    # ------------------------------------------------------------------
    if use_bfloat16:
        dtype = torch.bfloat16
        log.info(
            "EXPERIMENTAL: dtype=bfloat16. "
            "Verify that outputs match the float16 baseline before deploying."
        )
    else:
        dtype = torch.float16

    # ------------------------------------------------------------------
    # EXPERIMENTAL: Flash Attention 2
    # ------------------------------------------------------------------
    attn_impl = "eager"
    if use_flash_attention:
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
            log.info(
                "EXPERIMENTAL: Flash Attention 2 enabled (flash-attn %s). "
                "Effect depends on sequence length and GPU architecture. "
                "Measure before drawing conclusions.",
                flash_attn.__version__,
            )
        except ImportError:
            log.warning(
                "EXPERIMENTAL: Flash Attention 2 requested but the flash-attn "
                "package is not installed.  Falling back to standard attention. "
                "To install: pip install flash-attn --no-build-isolation"
            )

    log.info(
        "Loading optimized model: %s | device=%s | dtype=%s | attn=%s",
        model_name,
        device,
        dtype,
        attn_impl,
    )

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    model.eval()

    # ------------------------------------------------------------------
    # EXPERIMENTAL: torch.compile
    # ------------------------------------------------------------------
    if use_torch_compile:
        if device == "cpu":
            log.warning(
                "EXPERIMENTAL: torch.compile requested but device=cpu — skipping. "
                "torch.compile CUDA kernels are not generated for CPU execution."
            )
        else:
            log.info(
                "EXPERIMENTAL: applying torch.compile(mode='reduce-overhead'). "
                "The first inference call will be slow due to kernel compilation. "
                "Graph break warnings are expected and non-fatal."
            )
            try:
                model = torch.compile(model, mode="reduce-overhead")
                log.info("EXPERIMENTAL: torch.compile applied.")
            except Exception as exc:
                log.warning(
                    "EXPERIMENTAL: torch.compile failed (%s) — "
                    "continuing with uncompiled model.",
                    exc,
                )

    param_count = sum(
        p.numel() for p in model.parameters()
        if hasattr(p, "numel")
    ) / 1e9
    log.info("Optimized model ready. Parameters: %.2fB", param_count)

    return model, processor
