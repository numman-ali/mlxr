from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass(frozen=True)
class EncodedText:
    input_ids: np.ndarray
    attention_mask: np.ndarray


class TextTokenizer(Protocol):
    @property
    def model_max_length(self) -> int: ...

    @property
    def padding_side(self) -> str: ...

    @property
    def pad_token(self) -> str | None: ...

    @property
    def eos_token(self) -> str | None: ...

    def encode_text(
        self,
        prompt: str,
        *,
        max_length: int,
        truncation: bool = True,
        padding: str = "max_length",
    ) -> EncodedText: ...

    def decode_tokens(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str: ...


def _require_numpy_array(
    value: object,
    *,
    context: str,
) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    raise RuntimeError(context)


class TransformersTextTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer

    @property
    def model_max_length(self) -> int:
        return int(self._tokenizer.model_max_length)

    @property
    def padding_side(self) -> str:
        return str(self._tokenizer.padding_side)

    @padding_side.setter
    def padding_side(self, value: str) -> None:
        self._tokenizer.padding_side = value

    @property
    def pad_token(self) -> str | None:
        value = self._tokenizer.pad_token
        return None if value is None else str(value)

    @pad_token.setter
    def pad_token(self, value: str | None) -> None:
        self._tokenizer.pad_token = value

    @property
    def eos_token(self) -> str | None:
        value = self._tokenizer.eos_token
        return None if value is None else str(value)

    def encode_text(
        self,
        prompt: str,
        *,
        max_length: int,
        truncation: bool = True,
        padding: str = "max_length",
    ) -> EncodedText:
        encoded = self._tokenizer(
            prompt,
            return_tensors="np",
            max_length=max_length,
            truncation=truncation,
            padding=padding,
        )
        input_ids = _require_numpy_array(
            encoded.get("input_ids"),
            context="Tokenizer did not return input_ids",
        )
        attention_mask = _require_numpy_array(
            encoded.get("attention_mask"),
            context="Tokenizer did not return attention_mask",
        )
        return EncodedText(input_ids=input_ids, attention_mask=attention_mask)

    def decode_tokens(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        return str(
            self._tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
            )
        )


def load_local_text_tokenizer(
    model_path: Path,
    *,
    model_max_length: int,
    padding_side: str = "left",
) -> TextTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        model_max_length=model_max_length,
    )
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        eos_token = tokenizer.eos_token
        if eos_token is None:
            raise ValueError("Tokenizer must define pad_token or eos_token")
        tokenizer.pad_token = eos_token
    return TransformersTextTokenizer(tokenizer)


__all__ = [
    "EncodedText",
    "TextTokenizer",
    "TransformersTextTokenizer",
    "load_local_text_tokenizer",
]
