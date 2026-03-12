# FLUX.2 Family Adapter

This package carries the first-class `MLXR` family contract for `FLUX.2`.

Current truthful scope:

- source inspection for official diffusers-style `FLUX.2` bundles
- componentized portable-artifact conversion for the main released rows
- workflow planning for:
  - `image.generate`
  - `image.edit`
- owned MLX execution for:
  - `flux.2-klein-4b` `image.generate`
  - `flux.2-klein-9b` `image.generate`
  - `flux.2-klein-4b` single-reference `image.edit`
  - `flux.2-klein-9b` single-reference `image.edit`
  - `flux.2-klein-base-4b` `image.generate`
  - `flux.2-klein-base-9b` `image.generate`
  - `flux.2-klein-base-*` `image.edit`
- default runtime-registry enablement
- runtime-managed `png` and `jpg` output artifacts
- trusted-local image and LoRA inputs through the canonical job/CLI surface
- family-local extension parsing for prompt-upsampling intent

Current recommended rows:

- `flux.2-klein-9b`
- `flux.2-klein-4b`
- `flux.2-klein-base-4b` when the undistilled CFG path matters more than the
  faster distilled defaults
- `flux.2-klein-base-9b` when you want the strongest currently validated base
  row and can afford the extra runtime and memory cost

What is intentionally not claimed yet:

- quality-promoted multi-reference `image.edit`
- promoted real-run validation receipts for `flux.2-klein-base-9b`
- `flux.2-dev` runtime execution
- LoRA loading in the owned MLX backend
- quantized owned MLX execution
- first-party prompt upsampling as a required runtime dependency
- promoted real-run validation receipts for every supported row

Current runtime notes:

- distilled `klein` rows are fail-closed to the upstream 4-step recipe with
  `guidance_scale=1.0`
- base `klein` rows now use the official unconditional-plus-prompt CFG path with
  default `50` steps and `guidance_scale=4.0`
- `9b` is the stronger current editing default
- `4b` is materially lighter and works well for prompt-only generate plus
  simpler single-reference edits
- `flux.2-klein-base-4b` now has a real canonical `mlxr` generate receipt and a
  strict Gemini match review at `512x512 / 50 steps / guidance=4.0`
- `flux.2-klein-base-9b` now also has a real canonical `mlxr` generate receipt
  and a strict Gemini match review at `512x512 / 50 steps / guidance=4.0`
- `flux.2-klein-base-4b` single-reference editing is real but still
  prompt-sensitive; the first preservation-heavy edit drifted the source object
  details, so it remains under validation instead of fully promoted
- `flux.2-klein-base-9b` is materially heavier than `base-4b`; the first real
  generate receipt took about `217s` and peaked around `38 GB` active memory on
  this machine
- multi-reference `image.edit` is implemented for the current `klein` rows but
  not quality-promoted yet; current real receipts collapse too strongly toward
  the later reference image instead of blending the sources

See [23-flux2-capability-matrix.md](/Users/numman/Repos/mlxr/docs/research/23-flux2-capability-matrix.md)
for the family truth table and current validation receipts.
