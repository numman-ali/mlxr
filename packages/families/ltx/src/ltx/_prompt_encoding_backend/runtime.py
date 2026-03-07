# mypy: ignore-errors
# ruff: noqa: F401, I001
from __future__ import annotations

_RUNTIME_IMPORT_ERROR: Exception | None
try:
    import mlx.core as mx  # type: ignore[import-not-found, import-untyped]
    import mlx.nn as nn  # type: ignore[import-not-found, import-untyped]
    import numpy as np
    from mlx_lm.models.base import (  # type: ignore[import-not-found, import-untyped]
        create_causal_mask,
    )
    from mlx_vlm.models.gemma3.config import (  # type: ignore[import-not-found, import-untyped]
        TextConfig,
    )
    from mlx_vlm.models.gemma3.language import (  # type: ignore[import-not-found, import-untyped]
        Gemma3Model,
        create_attention_mask,
    )
    from safetensors import safe_open  # type: ignore[import-not-found, import-untyped]
    from transformers import AutoTokenizer  # type: ignore[import-not-found, import-untyped]
except Exception as exc:  # pragma: no cover - exercised via error path tests
    _RUNTIME_IMPORT_ERROR = exc
else:
    _RUNTIME_IMPORT_ERROR = None
