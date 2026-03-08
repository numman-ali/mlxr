from __future__ import annotations

_RUNTIME_IMPORT_ERROR: Exception | None
try:
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx_lm.models.base import (
        create_causal_mask,
    )
    from mlx_vlm.models.gemma3.config import (  # type: ignore[import-untyped]
        TextConfig,
    )
    from mlx_vlm.models.gemma3.language import (  # type: ignore[import-untyped]
        Gemma3Model,
        create_attention_mask,
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
