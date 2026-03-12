from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import numpy as np

from ._nn_compat import ModuleBase, build_linear, gelu_approx


@dataclass
class Qwen2VLVisionConfig:
    depth: int = 32
    hidden_size: int = 1280
    hidden_act: str = "silu"
    intermediate_size: int = 3420
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    window_size: int = 112
    out_hidden_size: int = 3584
    fullatt_block_indexes: tuple[int, ...] = (7, 15, 23, 31)

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "Qwen2VLVisionConfig":
        fullatt_block_indexes = _require_int_sequence(
            raw.get("fullatt_block_indexes"), key="fullatt_block_indexes"
        )
        return cls(
            depth=_require_int(raw.get("depth"), key="depth"),
            hidden_size=_require_int(raw.get("hidden_size"), key="hidden_size"),
            hidden_act=str(raw.get("hidden_act", "silu")),
            intermediate_size=_require_int(
                raw.get("intermediate_size"), key="intermediate_size"
            ),
            num_heads=_require_int(raw.get("num_heads"), key="num_heads"),
            in_channels=_require_int(raw.get("in_channels"), key="in_channels"),
            patch_size=_require_int(raw.get("patch_size"), key="patch_size"),
            spatial_merge_size=_require_int(
                raw.get("spatial_merge_size"), key="spatial_merge_size"
            ),
            temporal_patch_size=_require_int(
                raw.get("temporal_patch_size"), key="temporal_patch_size"
            ),
            window_size=_require_int(raw.get("window_size"), key="window_size"),
            out_hidden_size=_require_int(
                raw.get("out_hidden_size"), key="out_hidden_size"
            ),
            fullatt_block_indexes=fullatt_block_indexes,
        )


class VisionRMSNorm(ModuleBase):
    def __init__(self, hidden_size: int, *, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = mx.ones((hidden_size,), dtype=mx.float32)
        self.eps = eps

    def __call__(self, hidden_states: mx.array) -> mx.array:
        normalized = mx.fast.rms_norm(
            hidden_states.astype(mx.float32),
            self.weight.astype(mx.float32),
            self.eps,
        )
        return normalized.astype(hidden_states.dtype)


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _rotate_half(x: mx.array) -> mx.array:
    half = int(x.shape[-1]) // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _apply_rotary_pos_emb_vision(
    queries: mx.array,
    keys: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> tuple[mx.array, mx.array]:
    cos = mx.expand_dims(cos.astype(mx.float32), axis=1)
    sin = mx.expand_dims(sin.astype(mx.float32), axis=1)
    queries_float = queries.astype(mx.float32)
    keys_float = keys.astype(mx.float32)
    return (
        (queries_float * cos + _rotate_half(queries_float) * sin).astype(queries.dtype),
        (keys_float * cos + _rotate_half(keys_float) * sin).astype(keys.dtype),
    )


def _split_by_cu_seqlens(
    hidden: mx.array,
    cu_seqlens: mx.array,
) -> list[mx.array]:
    lengths = np.diff(np.asarray(cu_seqlens, dtype=np.int32))
    starts = np.asarray(cu_seqlens[:-1], dtype=np.int32)
    return [
        hidden[int(start) : int(start + length)]
        for start, length in zip(starts, lengths, strict=True)
        if int(length) > 0
    ]


class VisionPatchEmbed(ModuleBase):
    def __init__(self, config: Qwen2VLVisionConfig) -> None:
        super().__init__()
        input_dims = (
            config.in_channels
            * config.temporal_patch_size
            * config.patch_size
            * config.patch_size
        )
        self.proj = build_linear(input_dims, config.hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.proj(hidden_states)


class VisionRotaryEmbedding(ModuleBase):
    def __init__(self, dim: int, *, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / float(dim))
        )
        self._inv_freq = tuple(float(value.item()) for value in inv_freq)

    def __call__(self, sequence_length: int) -> mx.array:
        positions = mx.arange(sequence_length, dtype=mx.float32)
        inv_freq = mx.array(self._inv_freq, dtype=mx.float32)
        return positions[:, None] * inv_freq[None, :]


class VisionMLP(ModuleBase):
    def __init__(self, config: Qwen2VLVisionConfig) -> None:
        super().__init__()
        self.gate_proj = build_linear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
        )
        self.up_proj = build_linear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
        )
        self.down_proj = build_linear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.down_proj(
            _silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class VisionAttention(ModuleBase):
    def __init__(self, config: Qwen2VLVisionConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = build_linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = build_linear(config.hidden_size, config.hidden_size, bias=True)
        self.scale = self.head_dim**-0.5

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        cu_seqlens: mx.array,
        cos: mx.array,
        sin: mx.array,
    ) -> mx.array:
        seq_length = int(hidden_states.shape[0])
        qkv = self.qkv(hidden_states).reshape(
            seq_length, 3, self.num_heads, self.head_dim
        )
        queries, keys, values = mx.split(qkv, 3, axis=1)
        queries = mx.squeeze(queries, axis=1)
        keys = mx.squeeze(keys, axis=1)
        values = mx.squeeze(values, axis=1)
        queries, keys = _apply_rotary_pos_emb_vision(queries, keys, cos, sin)
        query_chunks = _split_by_cu_seqlens(queries, cu_seqlens)
        key_chunks = _split_by_cu_seqlens(keys, cu_seqlens)
        value_chunks = _split_by_cu_seqlens(values, cu_seqlens)
        outputs: list[mx.array] = []
        for query_chunk, key_chunk, value_chunk in zip(
            query_chunks, key_chunks, value_chunks, strict=True
        ):
            attended = mx.fast.scaled_dot_product_attention(
                mx.expand_dims(query_chunk.transpose(1, 0, 2), axis=0),
                mx.expand_dims(key_chunk.transpose(1, 0, 2), axis=0),
                mx.expand_dims(value_chunk.transpose(1, 0, 2), axis=0),
                scale=self.scale,
            )
            outputs.append(
                mx.squeeze(attended, axis=0)
                .transpose(1, 0, 2)
                .reshape(
                    int(query_chunk.shape[0]),
                    self.num_heads * self.head_dim,
                )
            )
        return self.proj(mx.concatenate(outputs, axis=0))


class VisionBlock(ModuleBase):
    def __init__(self, config: Qwen2VLVisionConfig) -> None:
        super().__init__()
        self.norm1 = VisionRMSNorm(config.hidden_size)
        self.norm2 = VisionRMSNorm(config.hidden_size)
        self.attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        cu_seqlens: mx.array,
        cos: mx.array,
        sin: mx.array,
    ) -> mx.array:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            cos=cos,
            sin=sin,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class VisionPatchMerger(ModuleBase):
    def __init__(self, config: Qwen2VLVisionConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.ln_q = VisionRMSNorm(config.hidden_size)
        self.mlp0 = build_linear(self.hidden_size, self.hidden_size, bias=True)
        self.mlp2 = build_linear(self.hidden_size, config.out_hidden_size, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        normalized = self.ln_q(hidden_states).reshape(-1, self.hidden_size)
        return self.mlp2(gelu_approx(self.mlp0(normalized)))


class Qwen2VLVisionModel(ModuleBase):
    def __init__(self, config: Qwen2VLVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = set(config.fullatt_block_indexes)
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.patch_embed = VisionPatchEmbed(config)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.blocks = [VisionBlock(config) for _ in range(config.depth)]
        self.merger = VisionPatchMerger(config)

    def rot_pos_emb(self, grid_thw: mx.array) -> mx.array:
        grid = np.asarray(grid_thw, dtype=np.int32)
        pos_ids: list[np.ndarray] = []
        for temporal, height, width in grid.tolist():
            hpos = np.arange(height, dtype=np.int32)[:, None].repeat(width, axis=1)
            hpos = hpos.reshape(
                height // self.spatial_merge_size,
                self.spatial_merge_size,
                width // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos = np.transpose(hpos, (0, 2, 1, 3)).reshape(-1)
            wpos = np.arange(width, dtype=np.int32)[None, :].repeat(height, axis=0)
            wpos = wpos.reshape(
                height // self.spatial_merge_size,
                self.spatial_merge_size,
                width // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos = np.transpose(wpos, (0, 2, 1, 3)).reshape(-1)
            pos_ids.append(
                np.repeat(
                    np.stack([hpos, wpos], axis=-1)[None, :, :],
                    repeats=temporal,
                    axis=0,
                ).reshape(-1, 2)
            )
        pos = np.concatenate(pos_ids, axis=0)
        max_grid_size = int(grid[:, 1:].max())
        rotary = np.asarray(self.rotary_pos_emb(max_grid_size), dtype=np.float32)
        gathered = rotary[pos]
        return mx.array(gathered.reshape(gathered.shape[0], -1), dtype=mx.float32)

    def get_window_index(self, grid_thw: mx.array) -> tuple[np.ndarray, np.ndarray]:
        window_index: list[np.ndarray] = []
        cu_window_seqlens: list[int] = [0]
        window_index_id = 0
        vit_merger_window_size = (
            self.window_size // self.spatial_merge_size // self.patch_size
        )
        for grid_t, grid_h, grid_w in np.asarray(grid_thw, dtype=np.int32).tolist():
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size
            index = np.arange(grid_t * llm_grid_h * llm_grid_w, dtype=np.int32).reshape(
                grid_t,
                llm_grid_h,
                llm_grid_w,
            )
            pad_h = (
                vit_merger_window_size - llm_grid_h % vit_merger_window_size
            ) % vit_merger_window_size
            pad_w = (
                vit_merger_window_size - llm_grid_w % vit_merger_window_size
            ) % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = np.pad(
                index,
                ((0, 0), (0, pad_h), (0, pad_w)),
                constant_values=-100,
            )
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = np.transpose(index_padded, (0, 1, 3, 2, 4)).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = np.sum(index_padded != -100, axis=(2, 3)).reshape(-1)
            index_new = index_padded.reshape(-1)
            index_new = index_new[index_new != -100]
            window_index.append(index_new + window_index_id)
            cu_window_tmp = (
                np.cumsum(seqlens) * self.spatial_merge_unit + cu_window_seqlens[-1]
            )
            cu_window_seqlens.extend(cu_window_tmp.tolist())
            window_index_id += grid_t * llm_grid_h * llm_grid_w
        return np.concatenate(window_index, axis=0), np.unique(
            np.asarray(cu_window_seqlens, dtype=np.int32)
        )

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        grid_thw: mx.array,
    ) -> mx.array:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        seq_len = int(hidden_states.shape[0])
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit,
            self.spatial_merge_unit,
            int(hidden_states.shape[-1]),
        )
        hidden_states = hidden_states[mx.array(window_index, dtype=mx.int32)]
        hidden_states = hidden_states.reshape(seq_len, int(hidden_states.shape[-1]))
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit,
            self.spatial_merge_unit,
            int(rotary_pos_emb.shape[-1]),
        )
        rotary_pos_emb = rotary_pos_emb[mx.array(window_index, dtype=mx.int32)]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, int(rotary_pos_emb.shape[-1]))
        emb = mx.concatenate([rotary_pos_emb, rotary_pos_emb], axis=-1)
        cos = mx.cos(emb)
        sin = mx.sin(emb)
        cu_seqlens = np.cumsum(
            np.repeat(
                np.asarray(grid_thw[:, 1], dtype=np.int32)
                * np.asarray(grid_thw[:, 2], dtype=np.int32),
                np.asarray(grid_thw[:, 0], dtype=np.int32),
            ),
            dtype=np.int32,
        )
        cu_seqlens = np.pad(cu_seqlens, (1, 0), constant_values=0)
        cu_seqlens_mx = mx.array(cu_seqlens, dtype=mx.int32)
        cu_window_seqlens_mx = mx.array(cu_window_seqlens, dtype=mx.int32)
        for layer_index, block in enumerate(self.blocks):
            current = (
                cu_seqlens_mx
                if layer_index in self.fullatt_block_indexes
                else cu_window_seqlens_mx
            )
            hidden_states = block(
                hidden_states,
                cu_seqlens=current,
                cos=cos,
                sin=sin,
            )
        merged = self.merger(hidden_states)
        reverse_indices = mx.array(np.argsort(window_index), dtype=mx.int32)
        return merged[reverse_indices, :]


def _require_int(raw: object, *, key: str) -> int:
    if not isinstance(raw, int):
        raise ValueError(f"Qwen2.5-VL vision_config.{key} must be an integer")
    return raw


def _require_int_sequence(raw: object, *, key: str) -> tuple[int, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(
            f"Qwen2.5-VL vision_config.{key} must be a sequence of integers"
        )
    values: list[int] = []
    for item in raw:
        if not isinstance(item, int):
            raise ValueError(
                f"Qwen2.5-VL vision_config.{key} must contain only integers"
            )
        values.append(item)
    return tuple(values)


def load_local_qwen2_vl_vision_model(model_path: Path) -> Qwen2VLVisionModel:
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise ValueError(f"Qwen2.5-VL config.json not found under '{model_path}'")
    raw = json.loads(config_path.read_text("utf-8"))
    vision_raw = raw.get("vision_config")
    if not isinstance(vision_raw, dict):
        raise ValueError("Qwen2.5-VL config is missing vision_config")
    config = Qwen2VLVisionConfig.from_dict(vision_raw)
    model = Qwen2VLVisionModel(config)
    loaded_keys: set[str] = set()
    for weight_file in _weight_files(model_path):
        shard = mx.load(str(weight_file))
        if not isinstance(shard, dict):
            raise RuntimeError(
                f"Qwen2.5-VL weight shard '{weight_file}' did not load into a weight mapping"
            )
        items: list[tuple[str, mx.array]] = []
        for key, value in shard.items():
            if not key.startswith("visual."):
                continue
            stripped = key[len("visual.") :]
            stripped = stripped.replace("merger.mlp.0.", "merger.mlp0.")
            stripped = stripped.replace("merger.mlp.2.", "merger.mlp2.")
            if stripped == "patch_embed.proj.weight":
                value = value.reshape(value.shape[0], -1)
            items.append((stripped, value))
            loaded_keys.add(stripped)
        if items:
            model.load_weights(items, strict=False)
    _assert_loaded_parameter_coverage(model, loaded_keys=loaded_keys)
    return model


def _weight_files(model_path: Path) -> tuple[Path, ...]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text("utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise RuntimeError(
                f"Qwen2.5-VL weight index '{index_path}' is missing a valid weight_map"
            )
        ordered_names = list(dict.fromkeys(str(value) for value in weight_map.values()))
        return tuple(model_path / name for name in ordered_names)
    single_file = model_path / "model.safetensors"
    if single_file.exists():
        return (single_file,)
    files = tuple(sorted(model_path.glob("*.safetensors")))
    if files:
        return files
    raise ValueError(f"No Qwen2.5-VL safetensors weights found under '{model_path}'")


def _assert_loaded_parameter_coverage(
    model: Qwen2VLVisionModel, *, loaded_keys: set[str]
) -> None:
    from mlx.utils import tree_flatten

    expected = tree_flatten(model.parameters(), destination={})
    expected_keys = {str(key) for key in expected}
    missing = expected_keys - loaded_keys
    unexpected = loaded_keys - expected_keys
    if not missing and not unexpected:
        return
    details: list[str] = []
    if missing:
        details.append(f"missing {len(missing)} keys")
    if unexpected:
        details.append(f"unexpected {len(unexpected)} keys")
    raise RuntimeError(
        f"Qwen2.5-VL vision weight coverage mismatch: {'; '.join(details)}"
    )


__all__ = [
    "Qwen2VLVisionConfig",
    "Qwen2VLVisionModel",
    "load_local_qwen2_vl_vision_model",
]
