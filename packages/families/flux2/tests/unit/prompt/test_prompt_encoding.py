from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from mlxr.families.flux2._prompt_encoding_backend import create_prompt_encoder


class _FakeTokenizer:
    def format_chat_prompt(
        self,
        prompt: str,
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool | None = None,
    ) -> str:
        assert add_generation_prompt is True
        assert enable_thinking is False
        return f"CHAT::{prompt}"

    def encode_text(
        self,
        prompt: str,
        *,
        max_length: int,
        truncation: bool = True,
        padding: str = "max_length",
    ):
        del max_length, truncation, padding

        class _Encoded:
            input_ids = np.array([[1, 2, 3]], dtype=np.int32)
            attention_mask = np.array(
                [[1, 1, 0]],
                dtype=np.int32 if prompt == "CHAT::sunrise city" else np.int32,
            )

        assert prompt in {"CHAT::sunrise city", "CHAT::"}
        return _Encoded()


class _FakeQwen3Model:
    def __call__(
        self,
        inputs: mx.array,
        attention_mask: mx.array | None = None,
        *,
        return_hidden_states: bool = False,
    ):
        del inputs, attention_mask
        assert return_hidden_states is True
        hidden_states = [
            mx.full((1, 3, 4), float(index), dtype=mx.float32) for index in range(32)
        ]
        return hidden_states[-1], hidden_states

    def parameters(self) -> list[object]:
        return []


class Flux2PromptEncodingTests(unittest.TestCase):
    def test_encode_stacks_expected_qwen3_hidden_layers(self) -> None:
        with (
            patch(
                "mlxr.families.flux2._prompt_encoding_backend.load_local_text_tokenizer",
                return_value=_FakeTokenizer(),
            ),
            patch(
                "mlxr.families.flux2._prompt_encoding_backend.load_local_qwen3_model",
                return_value=_FakeQwen3Model(),
            ),
        ):
            encoder = create_prompt_encoder(
                text_encoder_path=Path("/tmp/text_encoder"),
                tokenizer_path=Path("/tmp/tokenizer"),
            )
            result = encoder.encode("sunrise city")

        embeddings = np.asarray(result.prompt_embeddings.astype(mx.float32))
        self.assertEqual(embeddings.shape, (1, 3, 12))
        np.testing.assert_allclose(embeddings[..., :4], 9.0)
        np.testing.assert_allclose(embeddings[..., 4:8], 18.0)
        np.testing.assert_allclose(embeddings[..., 8:], 27.0)
        self.assertEqual(result.prompt_text, "sunrise city")
        self.assertEqual(result.token_count, 2)
        self.assertEqual(result.sequence_length, 3)
        self.assertEqual(result.hidden_size, 12)

    def test_encode_many_supports_unconditional_prompt_branch(self) -> None:
        with (
            patch(
                "mlxr.families.flux2._prompt_encoding_backend.load_local_text_tokenizer",
                return_value=_FakeTokenizer(),
            ),
            patch(
                "mlxr.families.flux2._prompt_encoding_backend.load_local_qwen3_model",
                return_value=_FakeQwen3Model(),
            ),
        ):
            encoder = create_prompt_encoder(
                text_encoder_path=Path("/tmp/text_encoder"),
                tokenizer_path=Path("/tmp/tokenizer"),
            )
            empty_result, prompt_result = encoder.encode_many(("", "sunrise city"))

        self.assertEqual(empty_result.prompt_text, "")
        self.assertEqual(prompt_result.prompt_text, "sunrise city")
        self.assertEqual(empty_result.sequence_length, prompt_result.sequence_length)


if __name__ == "__main__":
    unittest.main()
