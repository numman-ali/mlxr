# FLUX.2 Capability Matrix

Status: current owned-runtime truth table for the `FLUX.2` family in `MLXR`.

Use this file when deciding whether a `FLUX.2` row is truly green, merely
implemented, or still fail-closed.

## Upstream Rows

| Upstream row | Step-distilled | Guidance-distilled | Official tasks | License |
| --- | --- | --- | --- | --- |
| `flux.2-klein-4b` | yes | yes | `image.generate`, single-ref `image.edit`, multi-ref `image.edit` | Apache-2.0 |
| `flux.2-klein-9b` | yes | yes | `image.generate`, single-ref `image.edit`, multi-ref `image.edit` | non-commercial |
| `flux.2-klein-base-4b` | no | no | `image.generate`, single-ref `image.edit`, multi-ref `image.edit` | Apache-2.0 |
| `flux.2-klein-base-9b` | no | no | `image.generate`, single-ref `image.edit`, multi-ref `image.edit` | non-commercial |
| `flux.2-dev` | no | yes | `image.generate`, single-ref `image.edit`, multi-ref `image.edit`, prompt upsampling | non-commercial |

## Owned MLXR Matrix

| Capability | Rows | Runtime status | Quality status | Validation status | Notes |
| --- | --- | --- | --- | --- | --- |
| `image.generate` | `flux.2-klein-4b` | implemented | promoted | real CLI receipt plus `mflux` comparison pack | distilled recipe only: `4` steps, `guidance_scale=1.0` |
| `image.generate` | `flux.2-klein-9b` | implemented | promoted | real CLI receipt plus `mflux` comparison pack | stronger current default for high-quality stills |
| single-ref `image.edit` | `flux.2-klein-4b` | implemented | promoted | real CLI receipts, including simplified edit rung | more prompt-sensitive than `9b` |
| single-ref `image.edit` | `flux.2-klein-9b` | implemented | promoted | real CLI receipt | current strongest editing row |
| multi-ref `image.edit` | `flux.2-klein-4b` | implemented | not promoted | real CLI receipt plus Gemini mismatch review | current run collapses too strongly toward the later reference image |
| multi-ref `image.edit` | `flux.2-klein-9b` | implemented | not promoted | real CLI receipt plus Gemini mismatch review | same collapse behavior as `4b` in the current recipe |
| `image.generate` | `flux.2-klein-base-4b` | implemented | promoted | real CLI receipt plus strict Gemini match review | owned runtime uses the official CFG-style unconditional+prompt branch; slower than distilled `klein` but now real |
| `image.generate` | `flux.2-klein-base-9b` | implemented | promoted | real CLI receipt plus strict Gemini match review | strongest current owned base row, but materially slower and heavier than base 4B |
| single-ref `image.edit` | `flux.2-klein-base-4b` | implemented | not promoted | real CLI receipts only | first preservation-heavy edit drifted compass and glove; simpler coat-only edit looks strong by eye but still needs a clean semantic review |
| single-ref `image.edit` | `flux.2-klein-base-9b` | implemented | not promoted | gated-source access blocked | same owned base CFG edit engine, but no real receipt yet because the 9B row is gated |
| multi-ref `image.edit` | `flux.2-klein-base-*` | implemented | not promoted | none | do not promote until single-ref base editing is stable first |
| `image.generate` / `image.edit` | `flux.2-dev` | planned | not started | none | different prompt stack from `klein`; treat as separate engine tranche |
| LoRA loading | all rows | fail-closed | not started | unit validation only | CLI/runtime surface accepts refs but owned backend raises not implemented |
| quantized owned execution | all rows | fail-closed | not started | none | keep out of claimed runtime support until real |
| prompt upsampling | `flux.2-dev` family extension | fail-closed | not started | unit validation only | extension parsing exists, but no local or hosted implementation is promoted |
| negative prompt | all rows | fail-closed | not started | unit validation only | current owned backend rejects non-empty negative prompts |

## Current Validation Receipts

Single-reference promoted receipts:

- `flux.2-klein-9b` generate:
  `tmp/manual-runs/20260310T1255Z-flux2-cli-generate.png`
- `flux.2-klein-9b` single-ref edit:
  `tmp/manual-runs/20260310T1300Z-flux2-cli-edit-single.png`
- `flux.2-klein-4b` generate:
  `tmp/manual-runs/20260310T1322Z-flux2-klein4b-cli-generate.png`
- `flux.2-klein-4b` single-ref edit strong rung:
  `tmp/manual-runs/20260310T1340Z-flux2-klein4b-cli-edit-simple.png`
- `flux.2-klein-base-4b` generate:
  `tmp/manual-runs/20260310T162836Z-flux2-klein-base-4b-cli-generate.png`
  with strict Gemini match review under
  `tmp/manual-runs/20260310T162836Z-flux2-klein-base-4b-cli-generate-review/`
- `flux.2-klein-base-9b` generate:
  `tmp/manual-runs/20260310T1731Z-flux2-klein-base-9b-cli-generate.png`
  with strict Gemini match review under
  `tmp/manual-runs/20260310T1731Z-flux2-klein-base-9b-cli-generate-review/`

Comparison receipts:

- same-prompt `mflux` versus `MLXR` pack:
  `tmp/showcase-runs/mflux-flux2-compare-20260310T135555Z/`

Current multi-reference non-promotion receipts:

- `flux.2-klein-9b` multi-ref edit:
  `tmp/manual-runs/20260310T1428Z-flux2-klein9b-cli-edit-multiref.png`
- `flux.2-klein-4b` multi-ref edit:
  `tmp/manual-runs/20260310T1430Z-flux2-klein4b-cli-edit-multiref.png`

Current base-edit under-validation receipts:

- `flux.2-klein-base-4b` preservation-heavy single-ref edit:
  `tmp/manual-runs/20260310T1646Z-flux2-klein-base-4b-cli-edit-single.png`
  with Gemini partial verdict on source preservation
- `flux.2-klein-base-4b` simpler coat-only single-ref edit:
  `tmp/manual-runs/20260310T1653Z-flux2-klein-base-4b-cli-edit-simple.png`
  manual review looks promising, but the Gemini review session did not return cleanly and is not a promotion receipt yet

## Current Recommendation

Use `flux.2-klein-9b` as the best current FLUX row in `MLXR`.

Use `flux.2-klein-4b` when lighter memory and faster turnaround matter more
than the stronger edit behavior of `9b`.

Treat multi-reference `image.edit` as an implementation seam that still needs
fidelity work, not as a promoted product capability.

Treat `flux.2-klein-base-4b` as the first truthful owned base row when you need
the non-distilled CFG path, but not yet as the default recommendation over the
faster distilled `klein` rows.

Use `flux.2-klein-base-9b` when you want the strongest currently validated
owned base row and can afford the extra runtime and memory cost.
