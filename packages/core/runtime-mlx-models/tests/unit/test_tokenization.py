from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from mlxr.core.mlx_models.tokenization import (
    EncodedText,
    TransformersTextTokenizer,
    load_local_text_tokenizer,
)


class _FakeHFTokenizer:
    def __init__(self) -> None:
        self.model_max_length = 128
        self.padding_side = "right"
        self.pad_token: str | None = None
        self.eos_token: str | None = "</s>"
        self.last_prompt: str | None = None

    def __call__(
        self,
        prompt: str,
        *,
        return_tensors: str,
        max_length: int,
        truncation: bool,
        padding: str,
    ) -> dict[str, object]:
        self.last_prompt = prompt
        del return_tensors, truncation, padding
        return {
            "input_ids": np.zeros((1, max_length), dtype=np.int32),
            "attention_mask": np.ones((1, max_length), dtype=np.int32),
        }

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        del skip_special_tokens
        return ",".join(str(token) for token in token_ids)


class TokenizationTests(unittest.TestCase):
    def test_transformers_wrapper_encodes_and_decodes(self) -> None:
        fake = _FakeHFTokenizer()
        tokenizer = TransformersTextTokenizer(fake)
        encoded = tokenizer.encode_text("  hello  ", max_length=4)
        self.assertIsInstance(encoded, EncodedText)
        self.assertEqual(encoded.input_ids.shape, (1, 4))
        self.assertEqual(encoded.attention_mask.shape, (1, 4))
        self.assertEqual(tokenizer.decode_tokens([1, 2, 3]), "1,2,3")
        self.assertEqual(fake.last_prompt, "hello")

    def test_load_local_text_tokenizer_sets_left_padding_and_pad_token(self) -> None:
        fake_tokenizer = _FakeHFTokenizer()
        with TemporaryDirectory() as tmp_dir:
            with patch(
                "mlxr.core.mlx_models.tokenization.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ) as mocked_loader:
                loaded = load_local_text_tokenizer(Path(tmp_dir), model_max_length=64)
        self.assertIsInstance(loaded, TransformersTextTokenizer)
        self.assertEqual(fake_tokenizer.padding_side, "left")
        self.assertEqual(fake_tokenizer.pad_token, fake_tokenizer.eos_token)
        mocked_loader.assert_called_once()

    def test_load_local_text_tokenizer_rejects_missing_pad_and_eos_token(self) -> None:
        fake_tokenizer = _FakeHFTokenizer()
        fake_tokenizer.eos_token = None
        with TemporaryDirectory() as tmp_dir:
            with patch(
                "mlxr.core.mlx_models.tokenization.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "Tokenizer must define pad_token or eos_token",
                ):
                    load_local_text_tokenizer(Path(tmp_dir), model_max_length=64)


if __name__ == "__main__":
    unittest.main()
