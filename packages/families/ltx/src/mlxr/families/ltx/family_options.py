from __future__ import annotations

import secrets
from collections.abc import Mapping
from typing import Literal

DistilledGuidanceMode = Literal["positive_only", "cfg"]


def effective_seed(*, seed: int | None) -> int:
    if seed is not None:
        return seed
    return secrets.randbelow(2**31)


def family_extensions(raw_extensions: object) -> Mapping[str, object]:
    if raw_extensions is None:
        return {}
    if not isinstance(raw_extensions, Mapping):
        raise ValueError("LTX extensions must be an object when provided")
    return raw_extensions


def style_family_from_extensions(raw_extensions: object) -> str | None:
    style_family = family_extensions(raw_extensions).get("style_family")
    if style_family is None:
        return None
    if not isinstance(style_family, str):
        raise ValueError("LTX extensions.style_family must be a string when provided")
    normalized = style_family.strip()
    return normalized or None


def distilled_guidance_mode_from_extensions(
    raw_extensions: object,
) -> DistilledGuidanceMode:
    raw_mode = family_extensions(raw_extensions).get("distilled_guidance_mode")
    if raw_mode is None:
        return "positive_only"
    if not isinstance(raw_mode, str):
        raise ValueError(
            "LTX extensions.distilled_guidance_mode must be a string when provided"
        )
    normalized = raw_mode.strip().lower()
    if normalized == "positive_only":
        return "positive_only"
    if normalized == "cfg":
        return "cfg"
    raise ValueError(
        "LTX extensions.distilled_guidance_mode must be 'positive_only' or 'cfg'"
    )
