from __future__ import annotations

from .types import MLXArray, _AudioDecoderLike, _VocoderLike


def decode_audio(
    audio_latents: MLXArray,
    audio_decoder: _AudioDecoderLike,
    vocoder: _VocoderLike,
) -> MLXArray:
    decoded_audio = audio_decoder(audio_latents)
    waveform = vocoder(decoded_audio)
    if int(waveform.shape[0]) == 1:
        return waveform[0]
    return waveform
