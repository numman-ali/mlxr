from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from mlxr.families.qwen_image._prompt_encoding_backend import (
    _EDIT_PROMPT_TEMPLATE_DROP_TOKENS,
    _PROMPT_TEMPLATE,
    _PROMPT_TEMPLATE_DROP_TOKENS,
    _compute_image_position_ids,
    _select_valid_prompt_tokens,
    create_prompt_encoder,
)
from PIL import Image


class _FakeQwenTokenizer:
    def __init__(self) -> None:
        self.last_prompt: str | None = None

    def encode_text(
        self,
        prompt: str,
        *,
        max_length: int,
        truncation: bool = True,
        padding: str = "max_length",
    ):
        del truncation, padding
        self.last_prompt = prompt
        token_count = 40
        input_ids = np.arange(token_count, dtype=np.int32)[None, :]
        attention_mask = np.ones((1, token_count), dtype=np.int32)
        return type(
            "Encoded",
            (),
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )()


class _FakeQwenModel:
    def __call__(
        self,
        input_ids: mx.array,
        *,
        attention_mask: mx.array | None = None,
        return_hidden_states: bool = False,
    ):
        del attention_mask
        hidden = mx.expand_dims(input_ids.astype(mx.float32), axis=-1)
        if not return_hidden_states:
            return hidden
        return hidden, [hidden, hidden]

    def embed_tokens(self, inputs: mx.array) -> mx.array:
        return mx.expand_dims(inputs.astype(mx.float32), axis=-1)


class _FakeEditTokenizer:
    def __init__(self) -> None:
        self.image_token_id = 999

    def convert_tokens_to_ids(self, token: str) -> int:
        if token != "<|image_pad|>":
            raise AssertionError(f"Unexpected token {token}")
        return self.image_token_id

    def __call__(
        self,
        prompts: list[str],
        *,
        return_tensors: str,
        padding: bool,
    ) -> dict[str, np.ndarray]:
        del prompts, return_tensors, padding
        input_ids = np.arange(70, dtype=np.int32)[None, :]
        input_ids[0, 10:12] = self.image_token_id
        attention_mask = np.ones((1, 70), dtype=np.int32)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _FakeEditModel(_FakeQwenModel):
    def __call__(
        self,
        input_ids: mx.array,
        *,
        input_embeddings: mx.array | None = None,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        return_hidden_states: bool = False,
    ):
        del input_ids, attention_mask, position_ids
        hidden = (
            input_embeddings
            if input_embeddings is not None
            else mx.zeros((1, 1, 1), dtype=mx.float32)
        )
        if not return_hidden_states:
            return hidden
        return hidden, [hidden, hidden]


class _FakeVisionModel:
    def __call__(self, pixel_values: mx.array, *, grid_thw: mx.array) -> mx.array:
        del pixel_values, grid_thw
        return mx.array([[101.0], [102.0]], dtype=mx.float32)


class QwenImagePromptEncodingTests(unittest.TestCase):
    def test_compute_image_position_ids_matches_expected_layout(self) -> None:
        input_ids = np.arange(6, dtype=np.int32)[None, :]
        attention_mask = np.ones((1, 6), dtype=np.int32)
        mm_token_type_ids = np.asarray([[0, 0, 1, 1, 0, 0]], dtype=np.int32)
        image_grid_thw = np.asarray([[1, 2, 4]], dtype=np.int32)

        position_ids = _compute_image_position_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            spatial_merge_size=2,
        )

        expected = np.asarray(
            [
                [[0, 1, 2, 2, 4, 5]],
                [[0, 1, 2, 2, 4, 5]],
                [[0, 1, 2, 3, 4, 5]],
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(position_ids, expected)

    def test_select_valid_prompt_tokens_drops_template_prefix(self) -> None:
        hidden = mx.array(np.arange(40, dtype=np.float32).reshape(1, 40, 1))
        attention_mask = np.ones((1, 40), dtype=np.int32)

        selected, selected_mask = _select_valid_prompt_tokens(
            hidden=hidden,
            attention_mask=attention_mask,
            drop_tokens=_PROMPT_TEMPLATE_DROP_TOKENS,
            max_length=512,
        )

        np.testing.assert_array_equal(
            np.asarray(selected).reshape(-1),
            np.arange(_PROMPT_TEMPLATE_DROP_TOKENS, 40, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            np.asarray(selected_mask),
            np.ones((1, 40 - _PROMPT_TEMPLATE_DROP_TOKENS), dtype=np.int32),
        )

    def test_create_prompt_encoder_uses_official_template_and_drop_logic(self) -> None:
        fake_tokenizer = _FakeQwenTokenizer()
        with TemporaryDirectory() as tmp_dir:
            with (
                patch(
                    "mlxr.families.qwen_image._prompt_encoding_backend.load_local_qwen2_text_model",
                    return_value=_FakeQwenModel(),
                ),
                patch(
                    "mlxr.families.qwen_image._prompt_encoding_backend.load_local_text_tokenizer",
                    return_value=fake_tokenizer,
                ),
            ):
                encoder = create_prompt_encoder(
                    text_encoder_path=Path(tmp_dir),
                    tokenizer_path=Path(tmp_dir),
                )
                result = encoder.encode("heroic fox", max_length=64)

        self.assertEqual(
            fake_tokenizer.last_prompt,
            _PROMPT_TEMPLATE.format("heroic fox"),
        )
        self.assertEqual(result.token_count, 40 - _PROMPT_TEMPLATE_DROP_TOKENS)
        self.assertEqual(result.sequence_length, 40 - _PROMPT_TEMPLATE_DROP_TOKENS)
        self.assertEqual(result.hidden_size, 1)
        np.testing.assert_array_equal(
            np.asarray(result.prompt_embeddings).reshape(-1),
            np.arange(_PROMPT_TEMPLATE_DROP_TOKENS, 40, dtype=np.float32),
        )

    def test_encode_truncates_to_requested_length(self) -> None:
        fake_tokenizer = _FakeQwenTokenizer()
        with TemporaryDirectory() as tmp_dir:
            with (
                patch(
                    "mlxr.families.qwen_image._prompt_encoding_backend.load_local_qwen2_text_model",
                    return_value=_FakeQwenModel(),
                ),
                patch(
                    "mlxr.families.qwen_image._prompt_encoding_backend.load_local_text_tokenizer",
                    return_value=fake_tokenizer,
                ),
            ):
                encoder = create_prompt_encoder(
                    text_encoder_path=Path(tmp_dir),
                    tokenizer_path=Path(tmp_dir),
                )
                result = encoder.encode("heroic fox", max_length=3)

        self.assertEqual(result.token_count, 3)
        self.assertEqual(result.sequence_length, 3)
        np.testing.assert_array_equal(
            np.asarray(result.prompt_embeddings).reshape(-1),
            np.arange(
                _PROMPT_TEMPLATE_DROP_TOKENS,
                _PROMPT_TEMPLATE_DROP_TOKENS + 3,
                dtype=np.float32,
            ),
        )

    def test_create_prompt_encoder_supports_edit_task(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "input.png"
            Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8), mode="RGB").save(
                image_path
            )
            (root / "preprocessor_config.json").write_text(
                (
                    '{"min_pixels":3136,"max_pixels":12845056,"patch_size":14,'
                    '"temporal_patch_size":2,"merge_size":2,"rescale_factor":0.00392156862745098,'
                    '"image_mean":[0.1,0.2,0.3],"image_std":[0.4,0.5,0.6]}'
                ),
                encoding="utf-8",
            )
            with (
                patch(
                    "mlxr.families.qwen_image._prompt_encoding_backend.load_local_qwen2_text_model",
                    return_value=_FakeEditModel(),
                ),
                patch(
                    "mlxr.families.qwen_image._prompt_encoding_backend.load_local_qwen2_vl_vision_model",
                    return_value=_FakeVisionModel(),
                ),
                patch(
                    "mlxr.families.qwen_image._prompt_encoding_backend.AutoTokenizer.from_pretrained",
                    return_value=_FakeEditTokenizer(),
                ),
                patch(
                    "mlxr.families.qwen_image._prompt_encoding_backend.preprocess_qwen2_vl_images",
                    return_value=type(
                        "Processed",
                        (),
                        {
                            "pixel_values": np.zeros((8, 1176), dtype=np.float32),
                            "image_grid_thw": np.asarray([[1, 2, 4]], dtype=np.int32),
                        },
                    )(),
                ),
            ):
                encoder = create_prompt_encoder(
                    text_encoder_path=root,
                    tokenizer_path=root,
                    processor_path=root,
                    task="image.edit",
                )
                result = encoder.encode(
                    "brighten the flames",
                    image_paths=(image_path,),
                )

        self.assertEqual(result.token_count, 70 - _EDIT_PROMPT_TEMPLATE_DROP_TOKENS)
        self.assertEqual(result.sequence_length, 70 - _EDIT_PROMPT_TEMPLATE_DROP_TOKENS)


if __name__ == "__main__":
    unittest.main()
