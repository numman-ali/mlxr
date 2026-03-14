# FLUX.2 Family Notes

Status: companion family note for rationale, pressure, and bring-up context

For the current repo truth by row, read
[flux2-capability-matrix.md](./flux2-capability-matrix.md) together with this
note.

## Recommendation

Current recommendation:

- treat `flux.2-klein-9b` as the primary general-purpose klein row
- treat `flux.2-klein-9b-kv` as the edit-optimized klein row once it has real
  speed and quality receipts
- keep `flux.2-klein-4b` as the lighter sibling for faster or lower-memory
  bring-up
- keep `flux.2-dev` as the higher-end quality or editing row
- treat the base klein rows as important later parity rows, not the first bring-up

## Why This Family Matters

`FLUX.2` pressures different seams than `Qwen-Image` and `Z-Image`:

- one family spans generation and multi-reference editing across all major rows
- `klein` and `dev` do not share one prompt-encoder contract
- prompt upsampling is a first-party sidecar feature for harder prompts
- the family mixes Apache and non-commercial licensing across variants

This makes `FLUX.2` a strong cross-check that `MLXR` is not quietly becoming
`Z-Image`-shaped.

## Truthful Upstream Surface

Current official open-weight rows:

- `flux.2-klein-4b`
- `flux.2-klein-9b`
- `flux.2-klein-9b-kv`
- `flux.2-klein-base-4b`
- `flux.2-klein-base-9b`
- `flux.2-dev`

The official repo presents all of them as supporting:

- `image.generate`
- single-reference `image.edit`
- multi-reference `image.edit`

## Current Truthful MLXR Slice

Now real through the canonical `mlxr` runtime and CLI:

- `flux.2-klein-9b`
  - `image.generate`
  - single-reference `image.edit`
- `flux.2-klein-4b`
  - `image.generate`
  - single-reference `image.edit`
- `flux.2-klein-9b-kv`
  - implemented for `image.generate`
  - implemented for reference-conditioned `image.edit`
  - first canonical `mlxr` single-reference edit smoke now exists with KV stage
    metrics
  - not yet promoted and not yet validated against the upstream `-kv` bundle

Still open after that first owned slice:

- upstream-bundle validation and same-machine speed receipts for
  `flux.2-klein-9b-kv`
- quality-promoted multi-reference `image.edit`
- `flux.2-klein-base-4b`
- `flux.2-klein-base-9b`
- `flux.2-dev`

Deferred for the first promoted slice:

- local prompt upsampling
- OpenRouter-backed prompt upsampling
- watermark and moderation sidecars as hard product requirements
- quantized owned execution

## Important Platform Pressure

`FLUX.2` exposes several platform truths we should not hide:

- `klein` rows use Qwen3 text encoders
- `dev` uses Mistral
- guidance-distilled and CFG paths are different sampler behaviors, not just
  parameter changes
- `flux.2-klein-9b-kv` is not a new family contract; it is the same klein `9b`
  row plus a dedicated reference-token KV-cache execution path
- multi-reference editing is one engine path with AE-encoded reference tokens
  rather than a separate pipeline family
- the current owned multi-reference path is not quality-promoted yet because
  real receipts collapse too strongly toward the later reference image

## Product Role

`FLUX.2` is a good fit for:

- generation and editing in one family
- higher-quality related still generation before LTX
- future local desktop image-generation options that are not tied to one vendor

## Sources

- `references/official/flux2/README.md`
- `references/official/flux2/scripts/cli.py`
- `references/official/flux2/src/flux2/util.py`
- `references/official/flux2/src/flux2/sampling.py`
- `references/official/flux2/docs/flux2_klein_kv_cache.md`
- `references/official/flux2/docs/flux2_with_prompt_upsampling.md`
- `tmp/hf-flux2-klein-4b-live/model_index.json`
- `tmp/hf-flux2-klein-9b-live/model_index.json`
- `tmp/hf-flux2-klein-9b-kv-meta-20260312/model_index.json`
- `tmp/hf-flux2-dev-meta/model_index.json`
