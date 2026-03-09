from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlxr.families.ltx._prompt_encoding_backend import runtime
from mlxr.families.ltx._prompt_encoding_backend.components import LanguageModel


def _test_text_config() -> runtime.TextConfig:
    base_config = runtime.TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        sliding_window=16,
        _sliding_window_pattern=2,
        rope_parameters=None,
    )
    config_dict = base_config.to_dict()
    config_dict["sliding_window_pattern"] = 2
    return runtime.TextConfig.from_dict(config_dict)


class LanguageModelLoadingTests(unittest.TestCase):
    def _write_model_dir(self, root: Path) -> tuple[Path, dict[str, mx.array]]:
        model_dir = root / "text_encoder"
        model_dir.mkdir(parents=True, exist_ok=True)
        config = _test_text_config()
        (model_dir / "config.json").write_text(
            json.dumps({"text_config": config.to_dict()}),
            encoding="utf-8",
        )
        (model_dir / "model.safetensors").write_bytes(b"ignored")

        seed_model = LanguageModel(config=config)
        parameter_tree = tree_flatten(seed_model.parameters(), destination={})
        self.assertIsInstance(parameter_tree, dict)

        weights: dict[str, mx.array] = {}
        for index, (key, value) in enumerate(sorted(parameter_tree.items()), start=1):
            weights[f"language_model.{key}"] = mx.full(
                value.shape,
                index / 10.0,
                dtype=mx.bfloat16,
            )
        return model_dir, weights

    def test_language_model_loader_accepts_complete_weight_coverage(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            model_dir, weights = self._write_model_dir(Path(tmp_dir))
            with patch(
                "mlxr.families.ltx._prompt_encoding_backend.components.mx.load",
                return_value=weights,
            ):
                loaded_model = LanguageModel.from_pretrained(model_dir)

        loaded_tree = tree_flatten(loaded_model.parameters(), destination={})
        self.assertIsInstance(loaded_tree, dict)
        self.assertEqual(
            set(loaded_tree), {key.removeprefix("language_model.") for key in weights}
        )
        np.testing.assert_allclose(
            np.asarray(loaded_tree["model.embed_tokens.weight"].astype(mx.float32)),
            np.asarray(
                weights["language_model.model.embed_tokens.weight"].astype(mx.float32)
            ),
        )

    def test_language_model_loader_rejects_missing_weight_coverage(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            model_dir, weights = self._write_model_dir(Path(tmp_dir))
            incomplete_weights = dict(weights)
            incomplete_weights.pop("language_model.model.embed_tokens.weight")

            with patch(
                "mlxr.families.ltx._prompt_encoding_backend.components.mx.load",
                return_value=incomplete_weights,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Gemma text encoder weight coverage mismatch",
                ) as raised:
                    LanguageModel.from_pretrained(model_dir)

        self.assertIn("model.embed_tokens.weight", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
