from __future__ import annotations

import mlx.core as mx

from ... import _nn_compat as nn
from .blocks import (
    build_downsampling_stages,
    build_mid_block,
    build_upsampling_stages,
    run_mid_block,
)
from .contracts import (
    AudioCausalityAxis,
    AudioLatentShape,
    AudioNormKind,
    AudioPatchifier,
    PerChannelStatistics,
)
from .layers import AudioFeatureLayer, build_audio_normalization, make_audio_conv2d

LATENT_DOWNSAMPLE_FACTOR = 4


def _resolve_latent_mel_bins(
    *,
    input_mel_bins: int,
    channel_multipliers: tuple[int, ...],
) -> int:
    stage_count = len(channel_multipliers)
    architecture_downsample_factor = int(2 ** max(stage_count - 1, 0))
    if architecture_downsample_factor != LATENT_DOWNSAMPLE_FACTOR:
        raise ValueError(
            "Audio autoencoder architecture no longer matches the expected latent downsample factor"
        )
    return int(max(input_mel_bins // architecture_downsample_factor, 1))


def _silu(x: mx.array) -> mx.array:
    return nn.SiLU()(x)


class AudioEncoderModel(nn.Module):
    sample_rate: int
    mel_hop_length: int
    n_fft: int
    mel_bins: int
    is_causal: bool
    in_channels: int
    z_channels: int
    double_z: bool
    patchifier: AudioPatchifier
    norm_out: AudioFeatureLayer

    def __init__(
        self,
        *,
        base_channels: int,
        channel_multipliers: tuple[int, ...],
        num_res_blocks: int,
        attn_resolutions: set[int],
        dropout: float,
        in_channels: int,
        resolution: int,
        latent_channels: int,
        double_z: bool,
        norm_kind: AudioNormKind,
        causality_axis: AudioCausalityAxis,
        mid_block_add_attention: bool,
        sample_rate: int,
        mel_hop_length: int,
        n_fft: int,
        mel_bins: int,
        is_causal: bool,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.mel_hop_length = mel_hop_length
        self.n_fft = n_fft
        self.mel_bins = mel_bins
        self.is_causal = is_causal
        self.in_channels = in_channels
        self.z_channels = latent_channels
        self.double_z = double_z
        self.patchifier = AudioPatchifier(
            patch_size=1,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            sample_rate=sample_rate,
            hop_length=mel_hop_length,
            is_causal=is_causal,
        )
        latent_mel_bins = _resolve_latent_mel_bins(
            input_mel_bins=mel_bins,
            channel_multipliers=channel_multipliers,
        )
        flattened_latent_width = latent_channels * latent_mel_bins
        self._per_channel_statistics = PerChannelStatistics(flattened_latent_width)
        self.conv_in = make_audio_conv2d(
            in_channels,
            base_channels,
            kernel_size=3,
            stride=1,
            causality_axis=causality_axis,
        )
        self.down, block_in = build_downsampling_stages(
            ch=base_channels,
            ch_mult=channel_multipliers,
            num_res_blocks=num_res_blocks,
            resolution=resolution,
            temb_channels=0,
            dropout=dropout,
            norm_kind=norm_kind,
            causality_axis=causality_axis,
            attn_resolutions=attn_resolutions,
            resample_with_conv=True,
        )
        self.mid = build_mid_block(
            channels=block_in,
            temb_channels=0,
            dropout=dropout,
            norm_kind=norm_kind,
            causality_axis=causality_axis,
            add_attention=mid_block_add_attention,
        )
        self.norm_out = build_audio_normalization(block_in, norm_kind=norm_kind)
        self.conv_out = make_audio_conv2d(
            block_in,
            2 * latent_channels if double_z else latent_channels,
            kernel_size=3,
            stride=1,
            causality_axis=causality_axis,
        )

    def __call__(self, spectrogram: mx.array) -> mx.array:
        if spectrogram.ndim != 4:
            raise ValueError(
                f"AudioEncoderModel expected 4D input, got {tuple(int(s) for s in spectrogram.shape)!r}"
            )
        if int(spectrogram.shape[1]) == self.in_channels:
            spectrogram = mx.transpose(spectrogram, (0, 2, 3, 1))
        hidden = self.conv_in(spectrogram)
        for level in range(len(self.down)):
            stage = self.down[level]
            for block_index in range(len(stage.block)):
                hidden = stage.block[block_index](hidden, temb=None)
                if block_index in stage.attn:
                    hidden = stage.attn[block_index](hidden)
            if stage.downsample is not None:
                hidden = stage.downsample(hidden)
        hidden = run_mid_block(self.mid, hidden)
        hidden = _silu(self.norm_out(hidden))
        latent_output = self.conv_out(hidden)
        means = (
            latent_output[..., : self.z_channels] if self.double_z else latent_output
        )
        channels_last = mx.transpose(means, (0, 3, 1, 2))
        latent_shape = AudioLatentShape(
            batch=int(channels_last.shape[0]),
            channels=int(channels_last.shape[1]),
            frames=int(channels_last.shape[2]),
            mel_bins=int(channels_last.shape[3]),
        )
        patched = self.patchifier.patchify(channels_last)
        normalized = self.per_channel_statistics.normalize(patched)
        return self.patchifier.unpatchify(normalized, latent_shape)

    @property
    def per_channel_statistics(self) -> PerChannelStatistics:
        return self._per_channel_statistics


class AudioDecoderModel(nn.Module):
    out_channels: int
    causality_axis: AudioCausalityAxis
    mel_bins: int | None
    sample_rate: int
    mel_hop_length: int
    is_causal: bool
    z_channels: int
    patchifier: AudioPatchifier
    norm_out: AudioFeatureLayer

    def __init__(
        self,
        *,
        base_channels: int,
        out_channels: int,
        channel_multipliers: tuple[int, ...],
        num_res_blocks: int,
        attn_resolutions: set[int],
        resolution: int,
        latent_channels: int,
        norm_kind: AudioNormKind,
        causality_axis: AudioCausalityAxis,
        dropout: float,
        mid_block_add_attention: bool,
        sample_rate: int,
        mel_hop_length: int,
        is_causal: bool,
        mel_bins: int | None,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.causality_axis = causality_axis
        self.mel_bins = mel_bins
        self.sample_rate = sample_rate
        self.mel_hop_length = mel_hop_length
        self.is_causal = is_causal
        self.z_channels = latent_channels
        self.patchifier = AudioPatchifier(
            patch_size=1,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            sample_rate=sample_rate,
            hop_length=mel_hop_length,
            is_causal=is_causal,
        )
        if mel_bins is None:
            raise ValueError(
                "AudioDecoderModel requires mel_bins so per-channel statistics can match the latent contract"
            )
        latent_mel_bins = _resolve_latent_mel_bins(
            input_mel_bins=mel_bins,
            channel_multipliers=channel_multipliers,
        )
        flattened_latent_width = latent_channels * latent_mel_bins
        self._per_channel_statistics = PerChannelStatistics(flattened_latent_width)
        initial_block_channels = base_channels * channel_multipliers[-1]
        self.conv_in = make_audio_conv2d(
            latent_channels,
            initial_block_channels,
            kernel_size=3,
            stride=1,
            causality_axis=causality_axis,
        )
        self.mid = build_mid_block(
            channels=initial_block_channels,
            temb_channels=0,
            dropout=dropout,
            norm_kind=norm_kind,
            causality_axis=causality_axis,
            add_attention=mid_block_add_attention,
        )
        self.up, final_block_channels = build_upsampling_stages(
            ch=base_channels,
            ch_mult=channel_multipliers,
            num_res_blocks=num_res_blocks,
            resolution=resolution,
            temb_channels=0,
            dropout=dropout,
            norm_kind=norm_kind,
            causality_axis=causality_axis,
            attn_resolutions=attn_resolutions,
            resample_with_conv=True,
            initial_block_channels=initial_block_channels,
        )
        self.norm_out = build_audio_normalization(
            final_block_channels,
            norm_kind=norm_kind,
        )
        self.conv_out = make_audio_conv2d(
            final_block_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            causality_axis=causality_axis,
        )

    def __call__(self, sample: mx.array) -> mx.array:
        if sample.ndim != 4:
            raise ValueError(
                f"AudioDecoderModel expected 4D input, got {tuple(int(s) for s in sample.shape)!r}"
            )
        if int(sample.shape[1]) == self.z_channels:
            latent = sample
        elif int(sample.shape[-1]) == self.z_channels:
            latent = mx.transpose(sample, (0, 3, 1, 2))
        else:
            raise ValueError(
                "AudioDecoderModel input does not match latent channel contract"
            )
        latent, target_shape = self._denormalize_latents(latent)
        hidden = self.conv_in(mx.transpose(latent, (0, 2, 3, 1)))
        hidden = run_mid_block(self.mid, hidden)
        for level in reversed(range(len(self.up))):
            stage = self.up[level]
            for block_index in range(len(stage.block)):
                hidden = stage.block[block_index](hidden, temb=None)
                if block_index in stage.attn:
                    hidden = stage.attn[block_index](hidden)
            if stage.upsample is not None:
                hidden = stage.upsample(hidden)
        hidden = _silu(self.norm_out(hidden))
        decoded = self.conv_out(hidden)
        return self._adjust_output_shape(decoded, target_shape)

    def _denormalize_latents(
        self,
        sample: mx.array,
    ) -> tuple[mx.array, AudioLatentShape]:
        latent_shape = AudioLatentShape(
            batch=int(sample.shape[0]),
            channels=int(sample.shape[1]),
            frames=int(sample.shape[2]),
            mel_bins=int(sample.shape[3]),
        )
        patched = self.patchifier.patchify(sample)
        denormalized = self.per_channel_statistics.un_normalize(patched)
        unpatchified = self.patchifier.unpatchify(denormalized, latent_shape)
        target_frames = latent_shape.frames * LATENT_DOWNSAMPLE_FACTOR
        if self.causality_axis is not AudioCausalityAxis.NONE:
            target_frames = max(target_frames - (LATENT_DOWNSAMPLE_FACTOR - 1), 1)
        target_shape = AudioLatentShape(
            batch=latent_shape.batch,
            channels=self.out_channels,
            frames=target_frames,
            mel_bins=self.mel_bins
            if self.mel_bins is not None
            else latent_shape.mel_bins,
        )
        return unpatchified, target_shape

    def _adjust_output_shape(
        self,
        decoded_output: mx.array,
        target_shape: AudioLatentShape,
    ) -> mx.array:
        target_time = target_shape.frames
        target_freq = target_shape.mel_bins
        target_channels = target_shape.channels
        cropped = decoded_output[
            :,
            : min(int(decoded_output.shape[1]), target_time),
            : min(int(decoded_output.shape[2]), target_freq),
            :target_channels,
        ]
        time_padding = target_time - int(cropped.shape[1])
        freq_padding = target_freq - int(cropped.shape[2])
        if time_padding > 0 or freq_padding > 0:
            cropped = mx.pad(
                cropped,
                [
                    (0, 0),
                    (0, max(time_padding, 0)),
                    (0, max(freq_padding, 0)),
                    (0, 0),
                ],
            )
        exact = cropped[:, :target_time, :target_freq, :target_channels]
        return mx.transpose(exact, (0, 3, 1, 2))

    @property
    def per_channel_statistics(self) -> PerChannelStatistics:
        return self._per_channel_statistics
