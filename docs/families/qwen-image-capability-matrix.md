# Qwen-Image Capability Matrix

## Purpose

This document is the canonical truth map for the current `Qwen-Image` family
surface in `MLXR`.

Use it to answer six questions clearly:

1. which upstream `Qwen-Image` rows matter right now
2. how they map into the shared `MLXR` task surface
3. whether the row is implemented
4. whether the row has real receipts
5. whether the row is promoted or only supported
6. which gaps are still genuinely open

Read this together with:

- [qwen-image-family-notes.md](./qwen-image-family-notes.md) for
  the broader family rationale and source notes
- [qwen-bring-up-todo.md](../working/qwen-bring-up-todo.md) for
  the current evaluation and polish backlog

## Status labels

- `implemented`: code path exists in the owned runtime today
- `receipt-backed`: at least one real canonical `mlxr` runtime receipt exists
- `promoted`: safe to describe as current repo truth
- `supported but unpromoted`: real and usable, but not yet fully review-backed
- `blocked`: intentionally fail-closed in the current runtime

## Current matrix

| Upstream row or lane | MLXR task shape | Current runtime state | Validation state | Current repo truth |
| --- | --- | --- | --- | --- |
| `Qwen-Image-2512` base generate | `image.generate` | implemented | receipt-backed at official `1664x928 / 50-step` rung | promoted supported row |
| `Qwen-Image-2512` + LightX Lightning | `image.generate` plus `--lora` and `qwen_image.scheduler_preset=lightning` | implemented | receipt-backed at heavy `1664x928 / 8-step` rung | promoted fast lane |
| `Qwen-Image-2512` + Wuli turbo LoRA | `image.generate` plus `--lora` and `qwen_image.scheduler_preset=turbo_wuli` | implemented | receipt-backed at heavy `1664x928 / 4-step` rung | promoted fast lane |
| `Qwen-Image-Edit-2511` base edit | `image.edit` | implemented | receipt-backed on the official `1344x768 / 40-step` edit recipe | supported but unpromoted for quality |
| `Qwen-Image-Edit-2511` + Lightning | `image.edit` plus `--lora` and `qwen_image.scheduler_preset=lightning` | implemented | receipt-backed for practical single-image and multi-image edit | supported practical fast lane |
| `Qwen-Image-Edit-2511` + Multiple Angles LoRA | `image.edit` plus `--lora` | planned additive edit lane | not started | not yet supported |
| `Qwen-Image` older generate row | `image.generate` | code-supported | no fresh separate promoted receipt | supported in contract, not promoted |
| `Qwen-Image-Edit` older edit row | `image.edit` | code-supported | no fresh separate promoted receipt | supported in contract, not promoted |
| `Qwen-Image-Edit-2509` | `image.edit` | code-supported | no fresh separate promoted receipt | supported in contract, not promoted |
| `Qwen-Image-Layered` | layered image decomposition | blocked | not started | not yet supported |
| `Qwen-Image-2.0` | later family row | blocked | not started | not yet supported |

## What is truly real today

The current owned runtime genuinely supports:

- base `Qwen-Image-2512` `image.generate`
- base `Qwen-Image-Edit-2511` `image.edit`
- LightX Lightning fast generate and fast edit through the generic LoRA surface
- Wuli fast generate through the same generic LoRA surface
- runtime-managed `png` and `jpg` outputs
- trusted-local image and LoRA inputs through the canonical job and CLI path

The important practical detail is that Qwen is no longer “family contract
only.” It now has real heavy-rung receipts behind both base generation and base
editing, and the fast lanes are layered on top of that base surface rather than
replacing it.

## Current promoted truth versus support truth

Promoted truth:

- heavy-rung base `Qwen-Image-2512` generation is real and healthy
- heavy-rung LightX and Wuli generation lanes are real and healthy
- base `Qwen-Image-Edit-2511` is real on the official recipe
- Lightning edit is the first practical fast lane on this machine

Supported but not yet fully promoted:

- older `Qwen-Image` generate and edit rows
- base multi-image edit as a separately defended promoted claim
- full Gemini-review coverage for every supported generate and edit row

## Known remaining gaps

Still genuinely missing in `MLXR` today:

- `Qwen-Image-Layered`
- `Qwen-Image-2.0`
- `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA` as a validated additive edit
  lane on top of the current base `image.edit` surface
- prompt enhancement as a runnable local runtime feature
- the broader diffusers-only Qwen rows beyond plain generate and edit, such as:
  - `img2img`
  - inpaint
  - ControlNet variants

These are real upstream pressures, but they are not currently part of the
owned `MLXR` Qwen runtime claim.

## Important nuance on edit

The base edit row is real.

The earlier black-frame probe at `256x256 / 1 step` was only a backend smoke
and is not the correct quality verdict for the family. The fair verdict comes
from the official recipe, and that base receipt now exists.

So the current truthful framing is:

- base edit works
- Lightning edit is faster and currently more practical
- neither fact cancels the other

## Canonical evidence anchors

Generate:

- `tmp/showcase-runs/qwen-official-eval-20260311T191251Z/floating_city_base50_1664x928.png`
- `tmp/showcase-runs/qwen-official-eval-20260311T191251Z/floating_city_lightx8_1664x928.png`
- `tmp/showcase-runs/qwen-official-eval-20260311T190800Z/floating_city_wuli4_1664x928.png`

Edit:

- `tmp/showcase-runs/qwen-official-eval-20260311T221605Z/kenji_edit_base40_1344x768.png`
- `tmp/showcase-runs/qwen-official-eval-20260311T221605Z/kenji_edit_lightning4_1344x768.png`
- `tmp/manual-runs/qwen-edit-cli-lightning-4step-multiref.png`

Debug oracle:

- `tmp/manual-runs/qwen-decode-oracle-20260311T190307Z/`

## Recommended next Qwen work

The next best Qwen tasks are not more base bring-up. They are:

1. review and promotion coverage for the already-real rows
2. a true base multi-image edit receipt
3. only then `Layered` or broader Qwen pipeline expansion
