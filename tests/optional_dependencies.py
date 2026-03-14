"""Optional third-party dependency probes for repo test gating."""

from __future__ import annotations

import importlib.util


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


HAS_TORCH = _has_module("torch")
HAS_DIFFUSERS = _has_module("diffusers")
HAS_TRANSFORMERS = _has_module("transformers")

HAS_QWEN_TEXT_REFERENCE = HAS_TORCH and HAS_TRANSFORMERS
HAS_QWEN_VISION_REFERENCE = HAS_TORCH and HAS_TRANSFORMERS
HAS_QWEN_IMAGE_REFERENCE = HAS_TORCH and HAS_DIFFUSERS
