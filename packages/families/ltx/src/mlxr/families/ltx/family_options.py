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
