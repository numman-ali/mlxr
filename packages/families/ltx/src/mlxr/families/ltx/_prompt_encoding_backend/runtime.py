from __future__ import annotations

_RUNTIME_IMPORT_ERROR: Exception | None
try:
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlxr.core.mlx_models import (
        Gemma3Model,
        TextConfig,
        create_attention_mask,
        create_causal_mask,
    )
    from safetensors import safe_open
    from transformers import AutoTokenizer
except Exception as exc:  # pragma: no cover - exercised via error path tests
    _RUNTIME_IMPORT_ERROR = exc
else:
    _RUNTIME_IMPORT_ERROR = None

__all__ = [
    "_RUNTIME_IMPORT_ERROR",
    "AutoTokenizer",
    "Gemma3Model",
    "TextConfig",
    "create_attention_mask",
    "create_causal_mask",
    "mx",
    "nn",
    "np",
    "safe_open",
]
