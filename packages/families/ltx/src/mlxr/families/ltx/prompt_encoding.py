from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(slots=True)
class PromptEncodingResult:
    video_context: object
    audio_context: object | None
    attention_mask: object
    prompt_text: str
    token_count: int
    sequence_length: int
    video_context_shape: tuple[int, ...]
    attention_mask_shape: tuple[int, ...]
    audio_context_shape: tuple[int, ...] | None = None
    context_representation: str = "post_connector"
    caption_proj_before_connector: bool = False
    rope_type: str = "interleaved"
    double_precision_rope: bool = False
    connector_apply_gated_attention: bool = False
    transformer_context_dim: int | None = None
    transformer_apply_gated_attention: bool | None = None
    transformer_cross_attention_adaln: bool | None = None
    config_source: str | None = None
    negative_prompt_text: str | None = None
    negative_video_context: object | None = None
    negative_audio_context: object | None = None
    negative_video_context_shape: tuple[int, ...] | None = None
    negative_audio_context_shape: tuple[int, ...] | None = None


class PromptEncoder(Protocol):
    def encode(
        self,
        prompt: str,
        *,
        max_length: int = 1024,
        return_audio_context: bool = True,
        negative_prompt: str | None = None,
    ) -> PromptEncodingResult: ...

    def close(self) -> None: ...


def create_prompt_encoder(
    checkpoint_path: Path,
    text_encoder_path: Path,
) -> PromptEncoder:
    from ._prompt_encoding_backend import (
        create_prompt_encoder as create_backend_prompt_encoder,
    )

    return create_backend_prompt_encoder(
        checkpoint_path=checkpoint_path,
        text_encoder_path=text_encoder_path,
    )
