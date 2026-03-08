from __future__ import annotations

import unittest

from mlxr.families.ltx._generation_backend.model_config import (
    LTXModelConfig,
    LTXModelType,
)
from mlxr.families.ltx._generation_backend.rope_ops import LTXRopeType


class ModelConfigTests(unittest.TestCase):
    def test_from_dict_normalizes_enums_and_filters_unknown_keys(self) -> None:
        config = LTXModelConfig.from_dict(
            {
                "model_type": "ltx av model",
                "rope_type": "split",
                "caption_proj_before_connector": True,
                "ignored": "value",
            }
        )
        self.assertEqual(config.model_type, LTXModelType.AudioVideo)
        self.assertEqual(config.rope_type, LTXRopeType.SPLIT.value)
        self.assertTrue(config.caption_proj_before_connector)

    def test_get_video_and_audio_config_carry_guidance_flags(self) -> None:
        config = LTXModelConfig(
            apply_gated_attention=True,
            cross_attention_adaln=True,
        )
        video_config = config.get_video_config()
        audio_config = config.get_audio_config()
        assert video_config is not None
        assert audio_config is not None
        self.assertTrue(video_config.apply_gated_attention)
        self.assertTrue(video_config.cross_attention_adaln)
        self.assertTrue(audio_config.apply_gated_attention)
        self.assertTrue(audio_config.cross_attention_adaln)

    def test_defaults_fill_positional_limits(self) -> None:
        config = LTXModelConfig()
        self.assertEqual(config.positional_embedding_max_pos, [20, 2048, 2048])
        self.assertEqual(config.audio_positional_embedding_max_pos, [20])


if __name__ == "__main__":
    unittest.main()
