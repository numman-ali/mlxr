from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlxr.core.mlx_models.qwen3_text import Qwen3Model, load_local_qwen3_model
from mlxr.core.mlx_models.tokenization import TextTokenizer, load_local_text_tokenizer

from .prompt_encoding import PromptEncoder, PromptEncodingResult

_OUTPUT_LAYERS_QWEN3 = (9, 18, 27)


@dataclass(slots=True)
class _RuntimePromptEncoder(PromptEncoder):
    text_encoder_path: Path
    tokenizer_path: Path
    _tokenizer: TextTokenizer
    _model: Qwen3Model

    def encode(
        self,
        prompt: str,
        *,
        max_length: int = 512,
    ) -> PromptEncodingResult:
        prompt_text = prompt.strip()
        formatted_prompt = self._tokenizer.format_chat_prompt(
            prompt_text,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = self._tokenizer.encode_text(
            formatted_prompt,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        input_ids = mx.array(encoded.input_ids.astype(np.int32))
        attention_mask = mx.array(encoded.attention_mask.astype(np.int32))
        hidden, hidden_states = self._model(
            input_ids,
            attention_mask=attention_mask,
            return_hidden_states=True,
        )
        del hidden
        selected_states = []
        for index in _OUTPUT_LAYERS_QWEN3:
            try:
                selected = hidden_states[index]
            except IndexError as exc:  # pragma: no cover - config mismatch guard
                raise ValueError(
                    "FLUX.2 Qwen3 prompt encoder is missing one of the required "
                    f"hidden-state taps {_OUTPUT_LAYERS_QWEN3}"
                ) from exc
            selected_states.append(selected.astype(mx.bfloat16))
        prompt_embeddings = mx.concatenate(selected_states, axis=-1)
        mx.eval(prompt_embeddings)
        return PromptEncodingResult(
            prompt_embeddings=prompt_embeddings,
            prompt_text=prompt_text,
            token_count=int(np.asarray(encoded.attention_mask).sum().item()),
            sequence_length=int(prompt_embeddings.shape[1]),
            hidden_size=int(prompt_embeddings.shape[2]),
            attention_mask=attention_mask,
        )

    def encode_many(
        self,
        prompts: tuple[str, ...],
        *,
        max_length: int = 512,
    ) -> tuple[PromptEncodingResult, ...]:
        return tuple(self.encode(prompt, max_length=max_length) for prompt in prompts)

    def close(self) -> None:
        return None


def create_prompt_encoder(
    *,
    text_encoder_path: Path,
    tokenizer_path: Path,
) -> PromptEncoder:
    tokenizer = load_local_text_tokenizer(
        tokenizer_path,
        model_max_length=512,
        padding_side="right",
    )
    model = load_local_qwen3_model(text_encoder_path)
    mx.eval(model.parameters())
    return _RuntimePromptEncoder(
        text_encoder_path=text_encoder_path,
        tokenizer_path=tokenizer_path,
        _tokenizer=tokenizer,
        _model=model,
    )
