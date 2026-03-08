from __future__ import annotations

_RUNTIME_IMPORT_ERROR: Exception | None
try:
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlxr.core.mlx_models import (
        EncodedText,
        Gemma3Model,
        TextConfig,
        TextTokenizer,
        create_attention_mask,
        create_causal_mask,
        load_local_text_tokenizer,
    )
    from safetensors import safe_open
except Exception as exc:  # pragma: no cover - exercised via error path tests
    _RUNTIME_IMPORT_ERROR = exc
else:
    _RUNTIME_IMPORT_ERROR = None

__all__ = [
    "_RUNTIME_IMPORT_ERROR",
    "EncodedText",
    "Gemma3Model",
    "TextConfig",
    "TextTokenizer",
    "create_attention_mask",
    "create_causal_mask",
    "load_local_text_tokenizer",
    "mx",
    "nn",
    "np",
    "safe_open",
]
