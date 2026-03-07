# mypy: ignore-errors
from __future__ import annotations

from . import runtime
from .components import _to_binary_mask, _TransformerConfigResolution
from .encoder import create_prompt_encoder
from .masks import _connector_precomputed_freqs, _gemma_attention_masks
from .runtime import _RUNTIME_IMPORT_ERROR

if _RUNTIME_IMPORT_ERROR is None:
    from .encoder import _MLXLTXPromptEncoder

    mx = runtime.mx

    __all__ = [
        "_MLXLTXPromptEncoder",
        "_RUNTIME_IMPORT_ERROR",
        "_TransformerConfigResolution",
        "_connector_precomputed_freqs",
        "_gemma_attention_masks",
        "_to_binary_mask",
        "create_prompt_encoder",
        "mx",
    ]
else:
    mx = None
    __all__ = [
        "_RUNTIME_IMPORT_ERROR",
        "_TransformerConfigResolution",
        "_connector_precomputed_freqs",
        "_gemma_attention_masks",
        "_to_binary_mask",
        "create_prompt_encoder",
        "mx",
    ]
