from __future__ import annotations

import secrets
from collections.abc import Mapping
from typing import Literal

DistilledGuidanceMode = Literal["positive_only", "cfg"]
LTXWorkflowVariant = Literal[
    "distilled_two_stage",
    "two_stage",
    "two_stage_hq",
    "one_stage",
]
LTXControlVariant = Literal[
    "ic_lora",
    "union_ic_lora",
    "motion_track_control",
    "distilled_lora",
]


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


def workflow_variant_from_extensions(
    raw_extensions: object,
    *,
    supported_variants: tuple[str, ...],
    default: str | None = None,
) -> str | None:
    raw_variant = family_extensions(raw_extensions).get("workflow_variant")
    if raw_variant is None:
        return default
    if not isinstance(raw_variant, str):
        raise ValueError(
            "LTX extensions.workflow_variant must be a string when provided"
        )
    normalized = raw_variant.strip().lower()
    if normalized not in {
        "distilled_two_stage",
        "two_stage",
        "two_stage_hq",
        "one_stage",
    }:
        raise ValueError(
            "LTX extensions.workflow_variant must be one of "
            "'distilled_two_stage', 'two_stage', 'two_stage_hq', or 'one_stage'"
        )
    if normalized not in supported_variants:
        supported = ", ".join(supported_variants)
        raise ValueError(
            f"LTX workflow variant '{normalized}' is not supported by this model; "
            f"supported variants: {supported}"
        )
    return normalized


def control_variant_from_extensions(raw_extensions: object) -> LTXControlVariant | None:
    raw_variant = family_extensions(raw_extensions).get("control_variant")
    if raw_variant is None:
        return None
    if not isinstance(raw_variant, str):
        raise ValueError(
            "LTX extensions.control_variant must be a string when provided"
        )
    normalized = raw_variant.strip().lower()
    if normalized not in {
        "ic_lora",
        "union_ic_lora",
        "motion_track_control",
        "distilled_lora",
    }:
        raise ValueError(
            "LTX extensions.control_variant must be 'ic_lora', "
            "'union_ic_lora', 'motion_track_control', or 'distilled_lora'"
        )
    if normalized == "ic_lora":
        return "ic_lora"
    if normalized == "union_ic_lora":
        return "union_ic_lora"
    if normalized == "motion_track_control":
        return "motion_track_control"
    return "distilled_lora"


def conditioning_attention_strength_from_extensions(
    raw_extensions: object,
) -> float | None:
    raw_strength = family_extensions(raw_extensions).get(
        "conditioning_attention_strength"
    )
    if raw_strength is None:
        return None
    if not isinstance(raw_strength, (int, float)):
        raise ValueError(
            "LTX extensions.conditioning_attention_strength must be numeric when provided"
        )
    strength = float(raw_strength)
    if not (0.0 <= strength <= 1.0):
        raise ValueError(
            "LTX extensions.conditioning_attention_strength must be between 0.0 and 1.0"
        )
    return strength


def _hq_distilled_lora_strength_from_extensions(
    raw_extensions: object,
    *,
    key: str,
    default: float,
) -> float:
    raw_strength = family_extensions(raw_extensions).get(key, default)
    if not isinstance(raw_strength, (int, float)):
        raise ValueError(f"LTX extensions.{key} must be numeric when provided")
    strength = float(raw_strength)
    if strength < 0.0:
        raise ValueError(f"LTX extensions.{key} must be >= 0.0")
    return strength


def hq_stage_1_distilled_lora_strength_from_extensions(
    raw_extensions: object,
) -> float:
    return _hq_distilled_lora_strength_from_extensions(
        raw_extensions,
        key="hq_distilled_lora_strength_stage_1",
        default=0.25,
    )


def hq_stage_2_distilled_lora_strength_from_extensions(
    raw_extensions: object,
) -> float:
    return _hq_distilled_lora_strength_from_extensions(
        raw_extensions,
        key="hq_distilled_lora_strength_stage_2",
        default=0.5,
    )


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
