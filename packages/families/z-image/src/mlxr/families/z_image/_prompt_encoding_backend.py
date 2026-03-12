from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import mlx.core as mx
    import numpy as np
    from mlxr.core.mlx_models import (
        Qwen3Model,
        TextTokenizer,
        load_local_qwen3_model,
        load_local_text_tokenizer,
    )
except Exception as exc:  # pragma: no cover - exercised via runtime import failure path
    _RUNTIME_IMPORT_ERROR: Exception | None = exc
else:
    _RUNTIME_IMPORT_ERROR = None

from .prompt_encoding import PromptEncoder, PromptEncodingResult

if _RUNTIME_IMPORT_ERROR is None:

    @dataclass(slots=True)
    class _RuntimePromptEncoder(PromptEncoder):
        text_encoder_path: Path
        tokenizer_path: Path
        _model: Qwen3Model
        _tokenizer: TextTokenizer

        def encode(
            self,
            prompt: str,
            *,
            max_length: int = 512,
            negative_prompt: str | None = None,
        ) -> PromptEncodingResult:
            formatted_prompt = self._tokenizer.format_chat_prompt(
                prompt,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            positive = self._encode_formatted_prompt(
                formatted_prompt,
                max_length=max_length,
            )
            negative_embeddings: tuple[object, ...] | None = None
            normalized_negative_prompt: str | None = None
            if negative_prompt is not None:
                normalized_negative_prompt = negative_prompt.strip() or None
                formatted_negative_prompt = self._tokenizer.format_chat_prompt(
                    normalized_negative_prompt or "",
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                negative_embeddings = self._encode_formatted_prompt(
                    formatted_negative_prompt,
                    max_length=max_length,
                )[0]
            return PromptEncodingResult(
                prompt_embeddings=positive[0],
                prompt_text=prompt.strip(),
                token_count=positive[1],
                sequence_length=positive[2],
                hidden_size=positive[3],
                config_source=str(self.text_encoder_path / "config.json"),
                negative_prompt_text=normalized_negative_prompt,
                negative_prompt_embeddings=negative_embeddings,
            )

        def close(self) -> None:
            return None

        def _encode_formatted_prompt(
            self, formatted_prompt: str, *, max_length: int
        ) -> tuple[tuple[object, ...], int, int, int]:
            encoded = self._tokenizer.encode_text(
                formatted_prompt,
                max_length=max_length,
                truncation=True,
                # Mirror the official Z-Image pipeline contract: encode with
                # max-length padding, then slice back to the valid token span.
                padding="max_length",
            )
            input_ids = mx.array(encoded.input_ids, dtype=mx.int32)
            attention_mask = mx.array(encoded.attention_mask, dtype=mx.int32)
            _, hidden_states = self._model(
                input_ids,
                attention_mask=attention_mask,
                return_hidden_states=True,
            )
            prompt_hidden = hidden_states[-2]
            embeddings = _select_valid_tokens(prompt_hidden, encoded.attention_mask)
            hidden_size = int(prompt_hidden.shape[-1])
            sequence_length = int(prompt_hidden.shape[1])
            token_count = int(encoded.attention_mask[0].sum())
            return embeddings, token_count, sequence_length, hidden_size


def create_prompt_encoder(
    *,
    text_encoder_path: Path,
    tokenizer_path: Path,
) -> PromptEncoder:
    if _RUNTIME_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Z-Image prompt encoding runtime dependencies failed to import"
        ) from _RUNTIME_IMPORT_ERROR
    return _RuntimePromptEncoder(
        text_encoder_path=text_encoder_path,
        tokenizer_path=tokenizer_path,
        _model=load_local_qwen3_model(text_encoder_path),
        _tokenizer=load_local_text_tokenizer(
            tokenizer_path,
            model_max_length=131072,
            padding_side="right",
        ),
    )


def _select_valid_tokens(
    hidden: mx.array,
    attention_mask: np.ndarray,
) -> tuple[object, ...]:
    embeddings: list[object] = []
    for batch_index in range(int(hidden.shape[0])):
        valid_indices = np.flatnonzero(attention_mask[batch_index])
        if valid_indices.size <= 0:
            raise ValueError("Prompt encoding produced no valid tokens")
        start = int(valid_indices[0])
        end = int(valid_indices[-1]) + 1
        embeddings.append(hidden[batch_index, start:end, :])
    return tuple(embeddings)
