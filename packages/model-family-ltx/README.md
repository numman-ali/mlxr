# model-family-ltx

This package is the first family-adapter target for MLXR.

First implementation slice:

- local `LTX-2.3 Fast` text-to-video
- local `LTX-2.3 Fast` image-to-video
- strict-local MLX-native prompt encoding
- artifactized fast-path assets with validated component payloads
- resolved image-conditioning handles in the runtime worker
- runtime-managed `mp4` outputs with per-stage memory and timing telemetry

Out of first slice:

- checkpoint-faithful denoise/video generation
- retake
- audio-to-video
- IC-LoRA parity
- advanced Comfy node parity
