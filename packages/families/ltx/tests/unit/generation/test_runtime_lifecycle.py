from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from mlxr.families.ltx._generation_backend.distilled import LTXDistilledVideoGenerator
from mlxr.families.ltx._generation_backend.runtime_helpers import _release_transformers


class RuntimeLifecycleTests(unittest.TestCase):
    def test_release_transformers_clears_base_and_lora_caches(self) -> None:
        class _FakeHost:
            def __init__(self) -> None:
                self._transformer = object()
                self._transformer_lora_cache = {
                    (("lora-a", 1.0),): object(),
                    (("lora-b", 0.5),): object(),
                }

        host = _FakeHost()

        with patch(
            "mlxr.families.ltx._generation_backend.runtime_helpers.mx.clear_cache"
        ) as clear_cache:
            _release_transformers(host)

        self.assertIsNone(host._transformer)
        self.assertEqual(host._transformer_lora_cache, {})
        clear_cache.assert_called_once()

    def test_close_clears_transformer_lora_cache(self) -> None:
        generator = LTXDistilledVideoGenerator(
            checkpoint_path=Path("/tmp/fake-checkpoint.safetensors"),
            spatial_upsampler_path=Path("/tmp/fake-upsampler.safetensors"),
        )
        generator._transformer_lora_cache[(("lora-a", 1.0),)] = object()
        generator._transformer = object()

        with patch(
            "mlxr.families.ltx._generation_backend.distilled.mx.clear_cache"
        ) as clear_cache:
            generator.close()

        self.assertIsNone(generator._transformer)
        self.assertEqual(generator._transformer_lora_cache, {})
        clear_cache.assert_called_once()


if __name__ == "__main__":
    unittest.main()
