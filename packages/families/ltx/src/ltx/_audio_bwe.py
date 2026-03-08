from __future__ import annotations

import math
from collections.abc import Callable
from typing import Protocol

import mlx.core as mx
import numpy as np

from ._nn_compat import Module


class _MelSpectrogramComputer(Protocol):
    def mel_spectrogram(
        self, y: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]: ...


def _hann_sinc_filter1d(
    *,
    ratio: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
) -> mx.array:
    width = math.ceil(lowpass_filter_width / rolloff)
    kernel_size = 2 * width * ratio + 1
    time_axis = (
        np.arange(kernel_size, dtype=np.float32) / float(ratio) - width
    ) * float(rolloff)
    time_clamped = np.clip(time_axis, -lowpass_filter_width, lowpass_filter_width)
    window = np.cos(time_clamped * math.pi / lowpass_filter_width / 2.0) ** 2
    sinc_filter = np.sinc(time_axis) * window * rolloff / float(ratio)
    return mx.array(sinc_filter.reshape(1, kernel_size, 1), dtype=mx.float32)


class _WaveformResampler(Module):
    def __init__(self, *, input_sample_rate: int, output_sample_rate: int) -> None:
        super().__init__()
        if output_sample_rate % input_sample_rate != 0:
            raise ValueError(
                "LTX BWE resampler requires an integer output/input sample-rate ratio"
            )
        self.ratio = output_sample_rate // input_sample_rate
        self.stride = self.ratio
        self.filter = _hann_sinc_filter1d(ratio=self.ratio)
        width = math.ceil(6 / 0.99)
        kernel_size = 2 * width * self.ratio + 1
        self.pad = width
        self.pad_left = 2 * width * self.ratio
        self.pad_right = kernel_size - self.ratio

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim != 3:
            raise ValueError(
                f"LTX BWE resampler expects [batch, channels, time], got shape {x.shape}"
            )
        waveform = mx.transpose(x, (0, 2, 1))
        _, _, channels = waveform.shape
        padded = mx.pad(
            waveform,
            [(0, 0), (self.pad, self.pad), (0, 0)],
            mode="edge",
        )
        expanded_filter = mx.broadcast_to(
            self.filter.astype(waveform.dtype),
            (channels, self.filter.shape[1], self.filter.shape[2]),
        )
        upsampled = self.ratio * mx.conv_transpose1d(
            padded,
            expanded_filter,
            stride=self.stride,
            groups=channels,
        )
        upsampled = upsampled[:, self.pad_left : -self.pad_right, :]
        return mx.transpose(upsampled, (0, 2, 1))


class _STFTFn(Module):
    def __init__(self, *, filter_length: int, hop_length: int, win_length: int) -> None:
        super().__init__()
        self.hop_length = hop_length
        self.win_length = win_length
        n_freqs = filter_length // 2 + 1
        self.forward_basis = mx.zeros((n_freqs * 2, filter_length, 1), dtype=mx.float32)
        self.inverse_basis = mx.zeros((n_freqs * 2, filter_length, 1), dtype=mx.float32)

    def __call__(self, y: mx.array) -> tuple[mx.array, mx.array]:
        if y.ndim != 2:
            raise ValueError(f"LTX BWE STFT expects [batch, time], got shape {y.shape}")
        left_pad = max(0, self.win_length - self.hop_length)
        waveform = mx.pad(y[:, :, None], [(0, 0), (left_pad, 0), (0, 0)])
        spec = mx.conv1d(
            waveform, self.forward_basis.astype(y.dtype), stride=self.hop_length
        )
        n_freqs = self.forward_basis.shape[0] // 2
        real = spec[:, :, :n_freqs]
        imag = spec[:, :, n_freqs:]
        magnitude = mx.sqrt(real**2 + imag**2)
        phase = mx.arctan2(imag.astype(mx.float32), real.astype(mx.float32)).astype(
            real.dtype
        )
        return magnitude, phase


class AudioMelSTFT(Module):
    def __init__(
        self,
        *,
        filter_length: int,
        hop_length: int,
        win_length: int,
        n_mel_channels: int,
    ) -> None:
        super().__init__()
        self.stft_fn = _STFTFn(
            filter_length=filter_length,
            hop_length=hop_length,
            win_length=win_length,
        )
        n_freqs = filter_length // 2 + 1
        self.mel_basis = mx.zeros((n_mel_channels, n_freqs), dtype=mx.float32)

    def mel_spectrogram(
        self, y: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        magnitude, phase = self.stft_fn(y)
        energy = mx.linalg.norm(magnitude, axis=-1)
        mel = mx.matmul(
            magnitude.astype(mx.float32), self.mel_basis.astype(mx.float32).T
        )
        log_mel = mx.log(mx.maximum(mel, mx.array(1e-5, dtype=mel.dtype)))
        log_mel = mx.transpose(log_mel, (0, 2, 1))
        magnitude = mx.transpose(magnitude, (0, 2, 1))
        phase = mx.transpose(phase, (0, 2, 1))
        return log_mel, magnitude, phase, energy


class AudioVocoderWithBWE(Module):
    def __init__(
        self,
        *,
        vocoder: Callable[[mx.array], mx.array],
        bwe_generator: Callable[[mx.array], mx.array],
        mel_stft: _MelSpectrogramComputer,
        input_sample_rate: int,
        output_sample_rate: int,
        hop_length: int,
    ) -> None:
        super().__init__()
        self.vocoder = vocoder
        self.bwe_generator = bwe_generator
        self.mel_stft = mel_stft
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.hop_length = hop_length
        self.resampler: Callable[[mx.array], mx.array] = _WaveformResampler(
            input_sample_rate=input_sample_rate,
            output_sample_rate=output_sample_rate,
        )

    def _compute_mel(self, audio: mx.array) -> mx.array:
        if audio.ndim != 3:
            raise ValueError(
                f"LTX BWE vocoder expects [batch, channels, time], got shape {audio.shape}"
            )
        batch, channels, time = audio.shape
        flat = mx.reshape(audio, (batch * channels, time))
        mel, _, _, _ = self.mel_stft.mel_spectrogram(flat)
        mel_frames = mel.shape[-1]
        mel_channels = mel.shape[1]
        mel = mx.reshape(mel, (batch, channels, mel_channels, mel_frames))
        return mel

    def __call__(self, mel_spec: mx.array) -> mx.array:
        waveform = self.vocoder(mel_spec)
        _, _, low_rate_length = waveform.shape
        output_length = (
            low_rate_length * self.output_sample_rate // self.input_sample_rate
        )
        remainder = low_rate_length % self.hop_length
        if remainder != 0:
            waveform = mx.pad(
                waveform,
                [(0, 0), (0, 0), (0, self.hop_length - remainder)],
            )
        mel = self._compute_mel(waveform)
        mel_for_bwe = mx.transpose(mel, (0, 1, 3, 2))
        residual = self.bwe_generator(mel_for_bwe)
        skip = self.resampler(waveform)
        if residual.shape != skip.shape:
            raise RuntimeError(
                "LTX BWE residual/skip shape mismatch: "
                f"{residual.shape} != {skip.shape}"
            )
        combined = mx.clip(residual + skip, -1.0, 1.0)
        return combined[:, :, :output_length]
