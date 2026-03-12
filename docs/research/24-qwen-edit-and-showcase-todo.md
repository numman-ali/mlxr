# Qwen Overnight TODO

This note captures the concrete next steps for the current `Qwen-Image`
tranche so long-running generation and backend bring-up can continue across
context switches without re-deriving the same decisions.

## Showcase Matrix

- Keep the current comparison matrix sequential only.
- Use one heavy generation at a time.
- The canonical heavy runner is now `scripts/qwen_official_eval.py`.
- The heavyweight common generation rung is now `1664x928` across base, LightX,
  and Wuli; smaller matrices remain useful for unattended sweeps and quick
  comparisons, not for deciding whether full-resolution Qwen is healthy.
- Preserve the current per-profile recipe truth:
  - base: `scheduler_preset=default`, `guidance_scale=4.0`
  - LightX: `scheduler_preset=lightning`, `guidance_scale=1.0`
  - Wuli: `scheduler_preset=turbo_wuli`, `guidance_scale=1.0`
- Capture both `4`-step and `8`-step runs for:
  - base
  - LightX
  - Wuli
- Keep all outputs in one flat showcase folder with:
  - `manifest.json`
  - `summary.csv`
  - one exported PNG per scenario/profile pair
- Once the matrix is complete, inspect the outputs and identify:
  - the strongest speed/quality tradeoff
  - the best human/portrait result
  - the best dense-environment result
  - any profile that is consistently weaker than the others
- Keep the smaller `704x384` matrix as a quick unattended sweep, not as the
  final promotion bar now that the heavyweight `1664x928` rung is proven.
- Done:
  - first heavyweight base `Qwen-Image-2512` official-recipe receipt at
    `1664x928 / 50 steps / guidance=4.0`
  - the shared full-resolution Qwen generation corruption is fixed
  - clean `1664x928` generation receipts now exist for base `50-step`, LightX
    `8-step`, and Wuli `4-step`
  - `scripts/qwen_decode_oracle.py` now saves the canonical same-latent owned
    versus diffusers decode comparison receipt for future regressions
- Current heavyweight boundary:
  - the next promotion work is no longer “make full-res generate stop
    corrupting”
  - it is now “add consistent review coverage and decide which generate or edit
    profiles deserve promotion versus mere support”

## Runtime And Product Surface

- Keep reusing the already-converted `qwen-image-2512-lightning-test` artifact
  and existing LoRA handles from the runtime home during the current showcase
  tranche; avoid duplicate artifact and adapter imports while disk is tight.
- Treat the generic `--lora` surface as correct, but continue moving toward a
  first-class installed-adapter identity rather than repeated transient uploads.
- Keep the current queueing rule honest:
  many requests may queue, but one heavyweight image/video generation job should
  run at a time by default.
- Improve runtime bootstrap helpers only if they reduce unattended-run risk or
  make restart/resume behavior clearer.

## Cache And Disk Follow-Up

- Stop treating `tmp/hf-*-live` mirrors as the long-term product contract.
- Audit which large local mirrors can be replaced by pinned Hugging Face
  snapshot-backed source references instead of duplicated repo-local copies.
- Keep source cache, portable artifact storage, and machine-local execution
  cache conceptually separate.
- If more disk is needed for the edit tranche, prefer migrating to snapshot-ref
  usage or deleting clearly disposable duplicate mirrors only after checking
  which runtime artifacts still depend on them.

## Qwen Edit Bring-Up

- The truthful edit target is the official edit row, not a text-only shortcut.
- `Qwen-Image-Edit-2511` is the primary row to land first.
- The local bundle already exists under `tmp/hf-qwen-image-edit-2511-live`.
- The bundle contract is explicit:
  - `transformer`
  - `vae`
  - `text_encoder`
  - `tokenizer`
  - `scheduler`
  - `processor`
- The upstream `model_index.json` for `2511` points to
  `QwenImageEditPlusPipeline`, which means the first owned edit slice must
  account for the image-aware `processor` path.
- The core edit substrate gap is no longer multimodal prompt conditioning.
  That path is now owned and real; the remaining work is review coverage,
  promotion decisions, and broader row completion such as base multi-image
  proof or later `Layered`.
- Read the official references before coding:
  - `references/official/diffusers/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py`
  - `references/official/diffusers/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py`
  - `references/official/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`
  - `references/official/Qwen-Image/src/examples/edit_demo.py`
- Decide the smallest truthful owned edit slice:
  single-image semantic edit, one prompt, one output, no batching.
- Add a family-local edit prompt encoder path that can accept:
  - prompt text
  - input image
  - negative prompt
- Keep the workflow and shared task surface as they are unless upstream
  semantics force a better abstraction.
- Reuse as much of the generation backend as is truly common:
  - scheduler
  - latent packing/unpacking
  - output encoding
- Keep family-specific edit behavior local:
  - processor loading
  - vision token preparation
  - image-conditioned prompt scaffold
  - edit-pipeline latent combination semantics
- Validate the first owned edit slice with real receipts, not just unit tests.
- Done:
  - owned multimodal prompt encoding through the official processor path
  - owned `zero_cond_t` edit transformer modulation
  - owned VAE encode plus decode
  - first direct base edit receipt
  - first practical Lightning-backed edit receipts
  - first canonical `mlxr generate` edit receipt with Lightning
  - first canonical multi-image `mlxr generate` edit receipt with Lightning
- Current truthful boundary:
  - base edit now has a real official-recipe receipt
  - Lightning edit is still the first practical fast path on this machine
  - the first canonical CLI receipt is
    `tmp/manual-runs/qwen-edit-cli-lightning-4step-wide.png`
- Next:
  - add Gemini review coverage for the cleaned full-resolution generation rows
  - decide whether `4`-step or `8`-step Lightning is the better promoted fast
    edit preset for `MLXR`
  - add a true multi-image base edit receipt, not just Lightning multi-image
  - decide whether the promoted default edit story should be “base for maximum
    fidelity, Lightning for practical speed” or a narrower fast-first framing
  - keep `Qwen-Image-Layered` out of scope until generate/edit docs and reviews
    are fully squared away

## Validation Ladder

- For generation:
  - complete the full scenario/profile matrix
  - read the outputs
  - optionally run Gemini review on selected winners and suspicious failures
- For edit:
  - start with one simple semantic edit
  - then one appearance-preserving edit
  - then one text-editing example if the first two are real
- Do not promote edit capability until there is at least one clean real receipt
  through the canonical runtime path.

## Docs To Update After Each Breakthrough

- `packages/families/qwen-image/README.md`
- `docs/research/21-qwen-image-family-candidate.md`
- `MEMORY.md`
- capability matrix docs once edit becomes truthful
