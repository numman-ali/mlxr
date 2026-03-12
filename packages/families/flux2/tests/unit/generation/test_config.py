from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from mlxr.families.flux2._generation_backend.config import (
    Flux2TransformerConfig,
    SchedulerConfig,
)

from ._fixtures import tiny_scheduler_config, tiny_transformer_config


class Flux2ConfigTests(unittest.TestCase):
    def test_transformer_config_from_path_sets_hidden_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(
                json.dumps(asdict(tiny_transformer_config())),
                encoding="utf-8",
            )

            config = Flux2TransformerConfig.from_path(config_path)

        self.assertEqual(config.hidden_size, 8)
        self.assertTrue(config.guidance_embeds)

    def test_scheduler_config_defaults_dynamic_shifting_to_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "scheduler_config.json"
            raw = asdict(tiny_scheduler_config(use_dynamic_shifting=True))
            raw.pop("use_dynamic_shifting")
            config_path.write_text(json.dumps(raw), encoding="utf-8")

            config = SchedulerConfig.from_path(config_path)

        self.assertFalse(config.use_dynamic_shifting)


if __name__ == "__main__":
    unittest.main()
