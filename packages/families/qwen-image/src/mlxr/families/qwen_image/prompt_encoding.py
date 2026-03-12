from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(slots=True)
class PromptEncodingResult:
    prompt_embeddings: object
    prompt_attention_mask: object
    prompt_text: str
    token_count: int
    sequence_length: int
    hidden_size: int
    config_source: str | None = None
    negative_prompt_text: str | None = None
    negative_prompt_embeddings: object | None = None
    negative_prompt_attention_mask: object | None = None


@runtime_checkable
class PromptEncoder(Protocol):
    def encode(
        self,
        prompt: str,
        *,
        max_length: int = 512,
        negative_prompt: str | None = None,
        image_paths: tuple[Path, ...] = (),
    ) -> PromptEncodingResult: ...

    def close(self) -> None: ...


def create_prompt_encoder(
    *,
    text_encoder_path: Path,
    tokenizer_path: Path,
    processor_path: Path | None = None,
    task: str = "image.generate",
) -> PromptEncoder:
    from ._prompt_encoding_backend import (
        create_prompt_encoder as create_backend_prompt_encoder,
    )

    return create_backend_prompt_encoder(
        text_encoder_path=text_encoder_path,
        tokenizer_path=tokenizer_path,
        processor_path=processor_path,
        task=task,
    )
