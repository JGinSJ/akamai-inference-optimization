# serve/__init__.py intentionally imports only the infrastructure layer.
# Model loading (load_model, load_model_optimized) requires PyTorch and
# transformers, which are not available in all environments (e.g. CI without
# a GPU or the heavy ML deps installed).  Import those directly from their
# submodules when needed:
#
#   from serve.model import load_model, run_inference
#   from serve.model_optimized import load_model_optimized

from .batching import DynamicBatcher, Future, InferenceRequest

__all__ = [
    "DynamicBatcher",
    "Future",
    "InferenceRequest",
]
