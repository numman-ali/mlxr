from __future__ import annotations

from collections.abc import Mapping


def family_extensions(raw_extensions: object) -> Mapping[str, object]:
    if raw_extensions is None:
        return {}
    if not isinstance(raw_extensions, Mapping):
        raise ValueError("Qwen-Image extensions must be an object when provided")
    return raw_extensions


def prompt_enhance_mode_from_extensions(raw_extensions: object) -> str:
    raw_value = family_extensions(raw_extensions).get("prompt_enhance_mode")
    if raw_value is None:
        return "none"
    if not isinstance(raw_value, str):
        raise ValueError(
            "Qwen-Image extensions.prompt_enhance_mode must be a string when provided"
        )
    value = raw_value.strip().lower()
    if value not in {"none", "dashscope"}:
        raise ValueError(
            "Qwen-Image extensions.prompt_enhance_mode must be 'none' or 'dashscope'"
        )
    return value


def quantize_bits_from_extensions(raw_extensions: object) -> int | None:
    raw_value = family_extensions(raw_extensions).get("quantize_bits")
    if raw_value is None:
        return None
    if raw_value == 0:
        return None
    if not isinstance(raw_value, int):
        raise ValueError(
            "Qwen-Image extensions.quantize_bits must be an integer when provided"
        )
    if raw_value not in {2, 3, 4, 6, 8}:
        raise ValueError(
            "Qwen-Image extensions.quantize_bits must be one of 2, 3, 4, 6, or 8"
        )
    return raw_value


def scheduler_preset_from_extensions(raw_extensions: object) -> str:
    raw_value = family_extensions(raw_extensions).get("scheduler_preset")
    if raw_value is None:
        return "default"
    if not isinstance(raw_value, str):
        raise ValueError(
            "Qwen-Image extensions.scheduler_preset must be a string when provided"
        )
    value = raw_value.strip().lower()
    if value not in {"default", "lightning", "turbo_wuli"}:
        raise ValueError(
            "Qwen-Image extensions.scheduler_preset must be 'default', 'lightning', or 'turbo_wuli'"
        )
    return value
