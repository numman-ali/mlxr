from __future__ import annotations

from collections.abc import Mapping


def family_extensions(raw_extensions: object) -> Mapping[str, object]:
    if raw_extensions is None:
        return {}
    if not isinstance(raw_extensions, Mapping):
        raise ValueError("Z-Image extensions must be an object when provided")
    return raw_extensions


def cfg_normalization_from_extensions(raw_extensions: object) -> float:
    raw_value = family_extensions(raw_extensions).get("cfg_normalization")
    if raw_value is None:
        return 0.0
    if isinstance(raw_value, bool):
        return 1.0 if raw_value else 0.0
    if isinstance(raw_value, (int, float)):
        value = float(raw_value)
        if value < 0.0:
            raise ValueError(
                "Z-Image extensions.cfg_normalization must be >= 0.0 when provided"
            )
        return value
    raise ValueError(
        "Z-Image extensions.cfg_normalization must be a boolean or number when provided"
    )


def cfg_truncation_from_extensions(raw_extensions: object) -> float:
    raw_value = family_extensions(raw_extensions).get("cfg_truncation")
    if raw_value is None:
        return 1.0
    if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
        raise ValueError(
            "Z-Image extensions.cfg_truncation must be a number when provided"
        )
    value = float(raw_value)
    if value < 0.0:
        raise ValueError(
            "Z-Image extensions.cfg_truncation must be >= 0.0 when provided"
        )
    return value


def max_sequence_length_from_extensions(raw_extensions: object) -> int:
    raw_value = family_extensions(raw_extensions).get("max_sequence_length")
    if raw_value is None:
        return 512
    if not isinstance(raw_value, int) or isinstance(raw_value, bool):
        raise ValueError(
            "Z-Image extensions.max_sequence_length must be an integer when provided"
        )
    if raw_value <= 0:
        raise ValueError(
            "Z-Image extensions.max_sequence_length must be > 0 when provided"
        )
    return raw_value
