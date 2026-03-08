from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


def _hz_to_mel_slany(frequencies: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0

    mel = frequencies / f_sp
    log_mask = frequencies >= min_log_hz
    mel[log_mask] = min_log_mel + np.log(frequencies[log_mask] / min_log_hz) / logstep
    return mel.astype(np.float32, copy=False)


def _mel_to_hz_slany(mels: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0

    hz = mels * f_sp
    log_mask = mels >= min_log_mel
    hz[log_mask] = min_log_hz * np.exp((mels[log_mask] - min_log_mel) * logstep)
    return hz.astype(np.float32, copy=False)


def _slaney_mel_filter_bank(
    *,
    sample_rate: int,
    n_fft: int,
    mel_bins: int,
) -> npt.NDArray[np.float32]:
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs, dtype=np.float32)
    mel_min = _hz_to_mel_slany(np.array([0.0], dtype=np.float32))[0]
    mel_max = _hz_to_mel_slany(np.array([sample_rate / 2.0], dtype=np.float32))[0]
    mel_points = np.linspace(mel_min, mel_max, mel_bins + 2, dtype=np.float32)
    hz_points = _mel_to_hz_slany(mel_points)

    filters = np.zeros((mel_bins, n_freqs), dtype=np.float32)
    for index in range(mel_bins):
        left = hz_points[index]
        center = hz_points[index + 1]
        right = hz_points[index + 2]
        if not (left < center < right):
            continue
        up = (fft_freqs - left) / (center - left)
        down = (right - fft_freqs) / (right - center)
        filters[index] = np.maximum(0.0, np.minimum(up, down))

    energy_norm = 2.0 / np.maximum(
        hz_points[2 : mel_bins + 2] - hz_points[:mel_bins],
        np.finfo(np.float32).eps,
    )
    filters *= energy_norm[:, None]
    return filters.astype(np.float32, copy=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class AudioProcessor:
    sample_rate: int
    mel_bins: int
    mel_hop_length: int
    n_fft: int

    def __post_init__(self) -> None:
        if self.sample_rate < 1:
            raise ValueError("AudioProcessor sample_rate must be >= 1")
        if self.mel_bins < 1:
            raise ValueError("AudioProcessor mel_bins must be >= 1")
        if self.mel_hop_length < 1:
            raise ValueError("AudioProcessor mel_hop_length must be >= 1")
        if self.n_fft < 2:
            raise ValueError("AudioProcessor n_fft must be >= 2")
        if self.n_fft % 2 != 0:
            raise ValueError("AudioProcessor n_fft must be even")

    def waveform_to_mel(
        self,
        waveform: npt.NDArray[np.float32],
        sample_rate: int,
    ) -> npt.NDArray[np.float32]:
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"AudioProcessor expected sample_rate={self.sample_rate}, got {sample_rate}"
            )

        audio = np.asarray(waveform, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[None, :]
        if audio.ndim != 2:
            raise ValueError(
                "AudioProcessor expects waveform shape [channels, samples] or [samples]"
            )

        pad = self.n_fft // 2
        padded = np.pad(audio, ((0, 0), (pad, pad)), mode="reflect")
        frames = np.lib.stride_tricks.sliding_window_view(
            padded, window_shape=self.n_fft, axis=-1
        )[:, :: self.mel_hop_length, :]
        window = np.hanning(self.n_fft).astype(np.float32, copy=False)
        windowed = frames * window[None, None, :]
        spectrum = np.fft.rfft(windowed, axis=-1)
        magnitude = np.abs(spectrum).astype(np.float32, copy=False)

        mel_filter = _slaney_mel_filter_bank(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            mel_bins=self.mel_bins,
        )
        mel = magnitude.astype(np.float32, copy=False) @ mel_filter.T.astype(
            np.float32, copy=False
        )
        log_mel = np.log(np.maximum(mel, np.float32(1e-5)))
        batched = np.asarray(log_mel[None, :, :, :], dtype=np.float32)
        return batched
