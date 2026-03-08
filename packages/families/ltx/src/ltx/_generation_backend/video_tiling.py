from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mlx.core as mx

from .types import MLXArray, _TilingConfigInstance


def _trapezoidal_mask_1d(
    length: int,
    *,
    left_ramp: int,
    right_ramp: int,
    left_starts_from_zero: bool,
) -> MLXArray:
    if length <= 0:
        raise ValueError("Tile mask length must be positive")

    left_ramp = max(0, min(left_ramp, length))
    right_ramp = max(0, min(right_ramp, length))

    mask = mx.ones((length,), dtype=mx.float32)
    if left_ramp > 0:
        interval_length = left_ramp + 1 if left_starts_from_zero else left_ramp + 2
        ramp = mx.linspace(0.0, 1.0, interval_length, dtype=mx.float32)[:-1]
        if not left_starts_from_zero:
            ramp = ramp[1:]
        mask = mx.concatenate([ramp[:left_ramp], mask[left_ramp:]], axis=0)
    if right_ramp > 0:
        ramp = mx.linspace(1.0, 0.0, right_ramp + 2, dtype=mx.float32)[1:-1]
        mask = mx.concatenate([mask[:-right_ramp], ramp], axis=0)
    return mx.clip(mask, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class SpatialTilingConfig:
    tile_size_in_pixels: int
    tile_overlap_in_pixels: int = 0

    def __post_init__(self) -> None:
        if self.tile_size_in_pixels < 64:
            raise ValueError("tile_size_in_pixels must be at least 64")
        if self.tile_size_in_pixels % 32 != 0:
            raise ValueError("tile_size_in_pixels must be divisible by 32")
        if self.tile_overlap_in_pixels % 32 != 0:
            raise ValueError("tile_overlap_in_pixels must be divisible by 32")
        if self.tile_overlap_in_pixels >= self.tile_size_in_pixels:
            raise ValueError("Spatial overlap must be smaller than tile size")


@dataclass(frozen=True, slots=True)
class TemporalTilingConfig:
    tile_size_in_frames: int
    tile_overlap_in_frames: int = 0

    def __post_init__(self) -> None:
        if self.tile_size_in_frames < 16:
            raise ValueError("tile_size_in_frames must be at least 16")
        if self.tile_size_in_frames % 8 != 0:
            raise ValueError("tile_size_in_frames must be divisible by 8")
        if self.tile_overlap_in_frames % 8 != 0:
            raise ValueError("tile_overlap_in_frames must be divisible by 8")
        if self.tile_overlap_in_frames >= self.tile_size_in_frames:
            raise ValueError("Temporal overlap must be smaller than tile size")


@dataclass(frozen=True, slots=True)
class TilingConfig:
    spatial_config: SpatialTilingConfig | None = None
    temporal_config: TemporalTilingConfig | None = None

    @classmethod
    def default(cls) -> TilingConfig:
        return cls(
            spatial_config=SpatialTilingConfig(
                tile_size_in_pixels=512,
                tile_overlap_in_pixels=64,
            ),
            temporal_config=TemporalTilingConfig(
                tile_size_in_frames=64,
                tile_overlap_in_frames=24,
            ),
        )

    @classmethod
    def auto(
        cls,
        height: int,
        width: int,
        num_frames: int,
        *,
        spatial_threshold: int = 512,
        temporal_threshold: int = 65,
    ) -> TilingConfig | None:
        needs_spatial = height > spatial_threshold or width > spatial_threshold
        needs_temporal = num_frames > temporal_threshold
        if not needs_spatial and not needs_temporal:
            return None

        spatial_config = None
        temporal_config = None
        if needs_spatial:
            max_dim = max(height, width)
            tile_size = 384 if max_dim <= 768 else 512 if max_dim <= 1024 else 384
            spatial_config = SpatialTilingConfig(
                tile_size_in_pixels=tile_size,
                tile_overlap_in_pixels=64,
            )
        if needs_temporal:
            if num_frames > 200:
                temporal_config = TemporalTilingConfig(32, 8)
            elif num_frames > 100:
                temporal_config = TemporalTilingConfig(48, 16)
            else:
                temporal_config = TemporalTilingConfig(64, 24)
        return cls(spatial_config=spatial_config, temporal_config=temporal_config)


@dataclass(frozen=True, slots=True)
class _DimensionIntervals:
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    left_ramps: tuple[int, ...]
    right_ramps: tuple[int, ...]


def _split_spatial(
    *,
    tile_size: int,
    overlap: int,
    dimension_size: int,
) -> _DimensionIntervals:
    if dimension_size <= tile_size:
        return _DimensionIntervals((0,), (dimension_size,), (0,), (0,))

    amount = (dimension_size + tile_size - 2 * overlap - 1) // (tile_size - overlap)
    starts = tuple(index * (tile_size - overlap) for index in range(amount))
    ends = list(start + tile_size for start in starts)
    ends[-1] = dimension_size
    left_ramps = (0, *([overlap] * (amount - 1)))
    right_ramps = (*([overlap] * (amount - 1)), 0)
    return _DimensionIntervals(
        starts, tuple(ends), tuple(left_ramps), tuple(right_ramps)
    )


def _split_temporal(
    *,
    tile_size: int,
    overlap: int,
    dimension_size: int,
) -> _DimensionIntervals:
    base = _split_spatial(
        tile_size=tile_size,
        overlap=overlap,
        dimension_size=dimension_size,
    )
    if len(base.starts) == 1:
        return base
    starts = list(base.starts)
    left_ramps = list(base.left_ramps)
    for index in range(1, len(starts)):
        starts[index] -= 1
        left_ramps[index] += 1
    return _DimensionIntervals(
        tuple(starts),
        base.ends,
        tuple(left_ramps),
        base.right_ramps,
    )


def _map_temporal_slice(
    *,
    begin: int,
    end: int,
    left_ramp: int,
    right_ramp: int,
    scale: int,
) -> tuple[slice, MLXArray]:
    start = begin * scale
    stop = 1 + (end - 1) * scale
    left_scaled = 1 + (left_ramp - 1) * scale if left_ramp > 0 else 0
    right_scaled = right_ramp * scale
    mask = _trapezoidal_mask_1d(
        stop - start,
        left_ramp=left_scaled,
        right_ramp=right_scaled,
        left_starts_from_zero=True,
    )
    return slice(start, stop), mask


def _map_spatial_slice(
    *,
    begin: int,
    end: int,
    left_ramp: int,
    right_ramp: int,
    scale: int,
) -> tuple[slice, MLXArray]:
    start = begin * scale
    stop = end * scale
    mask = _trapezoidal_mask_1d(
        stop - start,
        left_ramp=left_ramp * scale,
        right_ramp=right_ramp * scale,
        left_starts_from_zero=False,
    )
    return slice(start, stop), mask


def decode_with_tiling(
    *,
    decoder_fn: Callable[..., MLXArray],
    latents: MLXArray,
    tiling_config: _TilingConfigInstance,
    spatial_scale: int = 32,
    temporal_scale: int = 8,
    causal: bool = False,
    timestep: MLXArray | None = None,
    chunked_conv: bool = False,
    on_frames_ready: Callable[[MLXArray, int], None] | None = None,
) -> MLXArray:
    batch, _, latent_frames, latent_height, latent_width = latents.shape
    output_frames = 1 + (latent_frames - 1) * temporal_scale
    output_height = latent_height * spatial_scale
    output_width = latent_width * spatial_scale

    if tiling_config.spatial_config is None:
        spatial_tile_size = max(latent_height, latent_width)
        spatial_overlap = 0
    else:
        spatial_tile_size = (
            tiling_config.spatial_config.tile_size_in_pixels // spatial_scale
        )
        spatial_overlap = (
            tiling_config.spatial_config.tile_overlap_in_pixels // spatial_scale
        )

    if tiling_config.temporal_config is None:
        temporal_tile_size = latent_frames
        temporal_overlap = 0
    else:
        temporal_tile_size = (
            tiling_config.temporal_config.tile_size_in_frames // temporal_scale
        )
        temporal_overlap = (
            tiling_config.temporal_config.tile_overlap_in_frames // temporal_scale
        )

    temporal_intervals = _split_temporal(
        tile_size=temporal_tile_size,
        overlap=temporal_overlap,
        dimension_size=latent_frames,
    )
    height_intervals = _split_spatial(
        tile_size=spatial_tile_size,
        overlap=spatial_overlap,
        dimension_size=latent_height,
    )
    width_intervals = _split_spatial(
        tile_size=spatial_tile_size,
        overlap=spatial_overlap,
        dimension_size=latent_width,
    )

    output = mx.zeros(
        (batch, 3, output_frames, output_height, output_width), dtype=mx.float32
    )
    weights = mx.zeros(
        (batch, 1, output_frames, output_height, output_width), dtype=mx.float32
    )
    emitted_frames = 0

    for t_index, t_start in enumerate(temporal_intervals.starts):
        t_end = temporal_intervals.ends[t_index]
        out_t_slice, t_mask = _map_temporal_slice(
            begin=t_start,
            end=t_end,
            left_ramp=temporal_intervals.left_ramps[t_index],
            right_ramp=temporal_intervals.right_ramps[t_index],
            scale=temporal_scale,
        )
        for h_index, h_start in enumerate(height_intervals.starts):
            h_end = height_intervals.ends[h_index]
            out_h_slice, h_mask = _map_spatial_slice(
                begin=h_start,
                end=h_end,
                left_ramp=height_intervals.left_ramps[h_index],
                right_ramp=height_intervals.right_ramps[h_index],
                scale=spatial_scale,
            )
            for w_index, w_start in enumerate(width_intervals.starts):
                w_end = width_intervals.ends[w_index]
                out_w_slice, w_mask = _map_spatial_slice(
                    begin=w_start,
                    end=w_end,
                    left_ramp=width_intervals.left_ramps[w_index],
                    right_ramp=width_intervals.right_ramps[w_index],
                    scale=spatial_scale,
                )

                tile_latents = latents[
                    :, :, t_start:t_end, h_start:h_end, w_start:w_end
                ]
                tile_output = decoder_fn(
                    tile_latents,
                    causal=causal,
                    timestep=timestep,
                    debug=False,
                    chunked_conv=chunked_conv,
                )
                actual_t = min(
                    int(tile_output.shape[2]), out_t_slice.stop - out_t_slice.start
                )
                actual_h = min(
                    int(tile_output.shape[3]), out_h_slice.stop - out_h_slice.start
                )
                actual_w = min(
                    int(tile_output.shape[4]), out_w_slice.stop - out_w_slice.start
                )

                blend_mask = (
                    t_mask[:actual_t].reshape(1, 1, -1, 1, 1)
                    * h_mask[:actual_h].reshape(1, 1, 1, -1, 1)
                    * w_mask[:actual_w].reshape(1, 1, 1, 1, -1)
                )
                tile_slice = tile_output[:, :, :actual_t, :actual_h, :actual_w].astype(
                    mx.float32
                )

                t_out_start = out_t_slice.start
                h_out_start = out_h_slice.start
                w_out_start = out_w_slice.start
                t_out_end = t_out_start + actual_t
                h_out_end = h_out_start + actual_h
                w_out_end = w_out_start + actual_w

                output[
                    :,
                    :,
                    t_out_start:t_out_end,
                    h_out_start:h_out_end,
                    w_out_start:w_out_end,
                ] = (
                    output[
                        :,
                        :,
                        t_out_start:t_out_end,
                        h_out_start:h_out_end,
                        w_out_start:w_out_end,
                    ]
                    + tile_slice * blend_mask
                )
                weights[
                    :,
                    :,
                    t_out_start:t_out_end,
                    h_out_start:h_out_end,
                    w_out_start:w_out_end,
                ] = (
                    weights[
                        :,
                        :,
                        t_out_start:t_out_end,
                        h_out_start:h_out_end,
                        w_out_start:w_out_end,
                    ]
                    + blend_mask
                )
                mx.eval(output, weights)

        if on_frames_ready is not None and t_index < len(temporal_intervals.starts) - 1:
            next_tile_start_latent = temporal_intervals.starts[t_index + 1]
            next_output_start = (
                0
                if next_tile_start_latent == 0
                else 1 + (next_tile_start_latent - 1) * temporal_scale
            )
            if next_output_start > emitted_frames:
                finalized_weights = mx.maximum(
                    weights[:, :, emitted_frames:next_output_start, :, :],
                    1e-8,
                )
                finalized_output = (
                    output[:, :, emitted_frames:next_output_start, :, :]
                    / finalized_weights
                ).astype(latents.dtype)
                mx.eval(finalized_output)
                on_frames_ready(finalized_output, emitted_frames)
                emitted_frames = next_output_start

    normalized = output / mx.maximum(weights, 1e-8)
    mx.eval(normalized)
    if on_frames_ready is not None and emitted_frames < output_frames:
        finalized_output = normalized[:, :, emitted_frames:, :, :].astype(latents.dtype)
        mx.eval(finalized_output)
        on_frames_ready(finalized_output, emitted_frames)
    return normalized.astype(latents.dtype)
