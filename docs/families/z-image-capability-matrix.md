# Z-Image Capability Matrix

Status: canonical truth table for the current `Z-Image` family surface in `MLXR`

## Purpose

This document is the canonical truth map for the current `Z-Image` family
surface in `MLXR`.

Use it to answer five questions clearly:

1. which upstream `Z-Image` rows matter right now
2. how they map into the shared `MLXR` task surface
3. whether the row is implemented
4. whether the row is promoted or still only supported
5. which gaps are still genuinely open

Read this together with:

- [z-image-family-notes.md](./z-image-family-notes.md) for rationale, product
  pressure, and bring-up context

## Status labels

- `implemented`: code path exists in the owned runtime today
- `receipt-backed`: at least one canonical `mlxr` runtime receipt exists
- `promoted`: safe to describe as current repo truth
- `supported but unpromoted`: real and usable, but not yet defended as the
  default public claim
- `blocked`: intentionally out of the current runtime claim

## Current matrix

| Upstream row or lane | MLXR task shape | Current runtime state | Validation state | Current repo truth |
| --- | --- | --- | --- | --- |
| `Z-Image-Turbo` | `image.generate` | implemented | receipt-backed fast prompt-only generate | promoted supported row |
| `Z-Image` base row | `image.generate` | implemented | receipt-backed at the balanced `768x768 / 28-step / cfg=4.0` rung | supported but unpromoted quality row |
| `Z-Image-Omni-Base` | broader generation and editing base | blocked | not started | not yet supported |
| `Z-Image-Edit` | image editing | blocked | not started | not yet supported |

## What is truly real today

The current owned runtime genuinely supports:

- prompt-only still-image generation through `Z-Image-Turbo`
- the same prompt-only generation surface through the released base `Z-Image`
  row
- runtime-managed `png` and `jpg` outputs
- source inspection and portable-artifact conversion for official
  diffusers-style bundles
- family-local `cfg_normalization` and `cfg_truncation` options on the base
  row through `extensions.z_image`

## Current promoted truth versus support truth

Promoted truth:

- `Z-Image-Turbo` is the current promoted prompt-only still-image row
- the family is a real local image-generation slice in `MLXR`

Supported but not yet fully promoted:

- the released base `Z-Image` generation row
- higher-rung quality claims on the base row
- any benchmark-backed larger-profile quality story on this machine

## Known remaining gaps

Still genuinely missing in `MLXR` today:

- `Z-Image-Edit`
- `Z-Image-Omni-Base`
- any ControlNet-style claims
- any prompt-enhancement claim as a separate runtime feature
- batch prompt generation in one runtime job
- product-surface exposure of family-local base-model tuning knobs

## Important nuance on family-local controls

The base row currently exposes `cfg_normalization` and `cfg_truncation`
through `extensions.z_image`.

That is truthful family-local runtime support. It is not yet a shared
cross-family product-surface concept, and it should stay documented that way
until a broader abstraction exists.
