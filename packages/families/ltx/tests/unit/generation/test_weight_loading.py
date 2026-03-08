from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlxr.families.ltx._generation_backend.model_config import (
    LTXModelConfig,
    LTXModelType,
)
from mlxr.families.ltx._generation_backend.transformer_model import LTXModel
from mlxr.families.ltx._generation_backend.weight_loading import (
    align_module_dtype_to_weights,
)


def test_align_module_dtype_to_weights_uses_dominant_dtype() -> None:
    module = nn.Linear(4, 3)
    weights = {
        "weight": mx.ones((3, 4), dtype=mx.bfloat16),
        "bias": mx.ones((3,), dtype=mx.bfloat16),
        "tiny": mx.ones((1,), dtype=mx.float32),
    }

    target_dtype = align_module_dtype_to_weights(
        module,
        weights,
        context="Test weights",
    )

    assert target_dtype == mx.bfloat16
    assert module.weight.dtype == mx.bfloat16


def test_transformer_loader_uses_checkpoint_weight_dtype() -> None:
    config = LTXModelConfig.from_dict(
        {
            "model_type": LTXModelType.AudioVideo,
            "num_attention_heads": 1,
            "attention_head_dim": 8,
            "in_channels": 4,
            "out_channels": 4,
            "num_layers": 1,
            "cross_attention_dim": 8,
            "caption_channels": 8,
            "audio_num_attention_heads": 1,
            "audio_attention_head_dim": 4,
            "audio_in_channels": 2,
            "audio_out_channels": 2,
            "audio_cross_attention_dim": 4,
            "audio_caption_channels": 4,
            "positional_embedding_theta": 10000.0,
            "positional_embedding_max_pos": [2, 8, 8],
            "audio_positional_embedding_max_pos": [2],
            "use_middle_indices_grid": True,
            "rope_type": "interleaved",
            "double_precision_rope": False,
            "timestep_scale_multiplier": 1000,
            "av_ca_timestep_scale_multiplier": 1000,
            "norm_eps": 1e-6,
            "apply_gated_attention": True,
            "cross_attention_adaln": True,
            "caption_proj_before_connector": False,
        }
    )
    seed_model = LTXModel(config)
    parameter_tree = tree_flatten(seed_model.parameters(), destination={})
    assert isinstance(parameter_tree, dict)

    weights_override = {
        f"model.diffusion_model.{key}": value.astype(mx.bfloat16)
        for key, value in parameter_tree.items()
    }

    loaded_model = LTXModel.from_pretrained(
        Path("/tmp/unused.safetensors"),
        config=config,
        strict=True,
        weights_override=weights_override,
    )
    loaded_parameters = tree_flatten(loaded_model.parameters(), destination={})
    assert isinstance(loaded_parameters, dict)

    dtypes = {value.dtype for value in loaded_parameters.values()}
    assert dtypes == {mx.bfloat16}
