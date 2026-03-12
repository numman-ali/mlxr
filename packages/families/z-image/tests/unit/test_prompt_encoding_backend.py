from __future__ import annotations

import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlxr.families.z_image._prompt_encoding_backend import (
    _RuntimePromptEncoder,
    _select_valid_tokens,
)


class PromptEncodingBackendTests(unittest.TestCase):
    def test_select_valid_tokens_keeps_full_unpadded_sequence(self) -> None:
        hidden = mx.arange(1 * 4 * 3, dtype=mx.float32).reshape(1, 4, 3)
        attention_mask = np.array([[1, 1, 1, 1]], dtype=np.int32)

        embeddings = _select_valid_tokens(hidden, attention_mask)

        np.testing.assert_allclose(np.asarray(embeddings[0]), np.asarray(hidden[0]))

    def test_select_valid_tokens_handles_left_padding(self) -> None:
        hidden = mx.arange(1 * 5 * 2, dtype=mx.float32).reshape(1, 5, 2)
        attention_mask = np.array([[0, 0, 1, 1, 1]], dtype=np.int32)

        embeddings = _select_valid_tokens(hidden, attention_mask)

        np.testing.assert_allclose(
            np.asarray(embeddings[0]),
            np.asarray(hidden[0, 2:5, :]),
        )

    def test_select_valid_tokens_handles_right_padding(self) -> None:
        hidden = mx.arange(1 * 5 * 2, dtype=mx.float32).reshape(1, 5, 2)
        attention_mask = np.array([[1, 1, 1, 0, 0]], dtype=np.int32)

        embeddings = _select_valid_tokens(hidden, attention_mask)

        np.testing.assert_allclose(
            np.asarray(embeddings[0]),
            np.asarray(hidden[0, 0:3, :]),
        )

    def test_runtime_prompt_encoder_uses_official_max_length_padding_contract(
        self,
    ) -> None:
        class FakeTokenizer:
            def __init__(self) -> None:
                self.calls: list[tuple[str, int, bool, str]] = []

            def format_chat_prompt(
                self,
                prompt: str,
                *,
                add_generation_prompt: bool = True,
                enable_thinking: bool | None = None,
            ) -> str:
                self.formatted = (prompt, add_generation_prompt, enable_thinking)
                return f"formatted::{prompt}"

            def encode_text(
                self,
                prompt: str,
                *,
                max_length: int,
                truncation: bool = True,
                padding: str = "max_length",
            ) -> object:
                self.calls.append((prompt, max_length, truncation, padding))
                return type(
                    "Encoded",
                    (),
                    {
                        "input_ids": np.array([[11, 12, 13, 0, 0, 0]], dtype=np.int32),
                        "attention_mask": np.array(
                            [[1, 1, 1, 0, 0, 0]], dtype=np.int32
                        ),
                    },
                )()

        class FakeModel:
            def __call__(
                self,
                inputs: mx.array,
                attention_mask: mx.array | None = None,
                *,
                return_hidden_states: bool = False,
            ) -> tuple[mx.array, list[mx.array]]:
                self.last_attention_mask = attention_mask
                hidden = mx.arange(1 * 6 * 4, dtype=mx.float32).reshape(1, 6, 4)
                return hidden, [hidden, hidden, hidden]

        tokenizer = FakeTokenizer()
        model = FakeModel()
        encoder = _RuntimePromptEncoder(
            text_encoder_path=Path("/tmp/text_encoder"),
            tokenizer_path=Path("/tmp/tokenizer"),
            _model=model,
            _tokenizer=tokenizer,
        )

        result = encoder.encode("storm lighthouse", max_length=6)

        self.assertEqual(
            tokenizer.calls,
            [("formatted::storm lighthouse", 6, True, "max_length")],
        )
        np.testing.assert_array_equal(
            np.asarray(model.last_attention_mask),
            np.array([[1, 1, 1, 0, 0, 0]], dtype=np.int32),
        )
        self.assertEqual(result.token_count, 3)
        self.assertEqual(result.sequence_length, 6)
        self.assertEqual(np.asarray(result.prompt_embeddings[0]).shape, (3, 4))


if __name__ == "__main__":
    unittest.main()
