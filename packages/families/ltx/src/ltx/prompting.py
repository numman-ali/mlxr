from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Orientation = Literal["portrait", "landscape", "square"]


@dataclass(frozen=True, slots=True)
class PromptShapingOptions:
    video_prompt: str | None = None
    audio_prompt: str | None = None
    natural_audio: bool = False
    no_music: bool = False
    duration_seconds: float | None = None
    orientation: Orientation | None = None


def shape_text_first_prompt(
    prompt: str,
    *,
    options: PromptShapingOptions | None = None,
) -> str:
    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        raise ValueError("LTX prompt shaping requires a non-empty prompt")

    active_options = options or PromptShapingOptions()
    parts = [normalized_prompt]

    if active_options.video_prompt:
        video_prompt = active_options.video_prompt.strip()
        if video_prompt:
            parts.append(f"Video details: {video_prompt}")

    if active_options.audio_prompt:
        audio_prompt = active_options.audio_prompt.strip()
        if audio_prompt:
            parts.append(f"Audio details: {audio_prompt}")

    parts.extend(_preference_lines(active_options))
    return "\n".join(parts)


def _preference_lines(options: PromptShapingOptions) -> tuple[str, ...]:
    lines: list[str] = []

    if options.natural_audio:
        lines.append(
            "Audio direction: use only natural diegetic environmental sound and subject-produced sound grounded in the scene."
        )
    if options.no_music:
        lines.append(
            "Audio prohibition: no soundtrack, no score, no background music, no instruments, no piano, and no melody."
        )
    if options.duration_seconds is not None:
        lines.append(f"Target duration: about {options.duration_seconds:.1f} seconds.")
    if options.orientation is not None:
        lines.append(f"Framing preference: {options.orientation} composition.")

    return tuple(lines)
