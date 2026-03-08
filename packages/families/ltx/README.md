# model-family-ltx

This package is the first family-adapter target for MLXR.

First implementation slice:

- local `LTX-2.3 Fast` text-to-video
- local `LTX-2.3 Fast` image-to-video
- local `LTX-2.3 Fast` audio-conditioned video with resolved runtime audio handles
- strict-local MLX-native prompt encoding
- artifactized fast-path assets with validated component payloads
- resolved image-conditioning handles in the runtime worker
- resolved audio-conditioning handles in the runtime worker
- runtime-managed `mp4` and `wav` outputs with per-stage memory and timing telemetry

Current measured caveat:

- the repo-owned strict-local MLX prompt encoder and distilled two-stage generation backend now complete direct artifact-backed `LTX-2.3` T2V and I2V smoke tests against the real upstream weights
- the bridge now fails closed on prompt/generation contract drift, 22B transformer feature mismatches, missing VAE latent statistics, and unsupported x2 upsampler layouts instead of silently producing low-fidelity output
- the current fixed-seed dog ladder now passes a clear-dog `384x224 / 17f / 24fps` rung and a coherent `768x512 / 33f / 24fps` rung against the real Hugging Face-converted artifact set
- the AV bridge now exports checkpoint-backed audio as muxed `mp4` or standalone `wav`, and the current repo-owned MLX audio path now includes the checkpoint’s full BWE wrapper instead of stopping at the `AMP1` base vocoder
- the current `video.condition.audio` slice is real on the artifact-backed bridge and preserves the resolved reference audio into the output path; that is a truthful preserved-reference capability, but it is still separate from broader scene-semantic quality on the conditioned path, and reference-video input, retake, interpolation, one-stage, HQ, and LoRA-driven controls are still tracked separately in the capability matrix
- the current text-first prompt stack now uses a shared repo-owned shaper for explicit audio details plus `natural_audio` / `no_music` / duration / orientation cues, but those remain advisory text controls until semantic validation says otherwise; the first 10-second dog showcase run now passes visually and still fails the no-music audio bar
- the current first promoted 10-second text-first showcase pack is now the score-friendly set under `tmp/showcase-runs/showcase-pack-20260308.json`: anime neon chase, stop-motion workshop, vintage nostalgic street, cinematic spacewalk, and noir rainy city all clear the current visual/audio review bar at `384x224 / 241f / 24fps`
- the natural-audio showcase story is still separate: the stronger Gemini review path that inspects both the MP4 and an extracted review WAV still classifies the current dog, nature, and natural-vintage text-first clips as music-like, so natural-scene realism remains a truthful open gap
- the remaining truth gap is broader capability completion plus recommended-resolution and HQ-profile validation once that capability matrix is green

Out of first slice:

- recommended-resolution and HQ profile validation
- retake
- reference-video conditioning
- keyframe interpolation
- IC-LoRA parity
- advanced Comfy node parity
