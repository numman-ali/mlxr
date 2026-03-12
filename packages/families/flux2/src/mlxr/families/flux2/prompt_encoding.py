from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import mlx.core as mx


@dataclass(slots=True)
class PromptEncodingResult:
    prompt_embeddings: mx.array
    prompt_text: str
    token_count: int
    sequence_length: int
    hidden_size: int
    attention_mask: mx.array


@runtime_checkable
class PromptEncoder(Protocol):
    def encode(
        self,
        prompt: str,
        *,
        max_length: int = 512,
    ) -> PromptEncodingResult: ...

    def encode_many(
        self,
        prompts: tuple[str, ...],
        *,
        max_length: int = 512,
    ) -> tuple[PromptEncodingResult, ...]: ...

    def close(self) -> None: ...


def create_prompt_encoder(
    *,
    text_encoder_path: Path,
    tokenizer_path: Path,
) -> PromptEncoder:
    from ._prompt_encoding_backend import (
        create_prompt_encoder as create_backend_prompt_encoder,
    )

    return create_backend_prompt_encoder(
        text_encoder_path=text_encoder_path,
        tokenizer_path=tokenizer_path,
    )
