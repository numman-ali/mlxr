# Z-Image Family Adapter

This package carries the `MLXR` family contract for `Z-Image`.

The current truthful slice includes:

- source inspection for diffusers-style `Z-Image` bundles
- componentized portable-artifact conversion for `transformer`, `vae`, `text_encoder`, `tokenizer`, and `scheduler`
- workflow planning for prompt-only `image.generate`
- native Apple Silicon / MLX execution for prompt-only still-image generation
- runtime-managed `png` and `jpg` outputs
- canonical `mlxr generate` support for `width`, `height`, `seed`, `num_inference_steps`, `guidance_scale`, and optional `negative_prompt`
- shared runtime job telemetry for stage timings, progress, and MLX memory snapshots
- env-gated family trace metadata for deeper denoise and decode benchmarking
- family-local `extensions.z_image` support for base-model `cfg_normalization` and `cfg_truncation`

What is still intentionally missing:

- prompt-list or multi-image batch generation in one job
- `num_images_per_prompt`
- latent output mode
- product-surface performance knobs such as attention backend selection or compile/offload toggles
- editing or omni-base task rows
- benchmark-backed claims for bilingual text rendering or broader quality beyond the current smoke and showcase rungs

Current status for the released rows:

- `Z-Image-Turbo`
  - works end to end through the canonical runtime and CLI path
  - current promoted slice is prompt-only `image.generate`
- `Z-Image`
  - canonical runtime/CLI smoke now completes with negative prompt plus nonzero guidance
  - family-local `cfg_normalization` and `cfg_truncation` are now wired through workflow/job extensions without widening the shared public surface
  - the first balanced quality rung now completes with a coherent `768x768 / 28-step / cfg=4.0` runtime receipt and trace-bearing metrics
  - broader quality validation is still in progress, and larger `1024x1024 / 36-step` base runs are still outside a quick dev-loop budget on this machine
- `Z-Image-Omni-Base`
  - unreleased upstream, blocked
- `Z-Image-Edit`
  - unreleased upstream, blocked
