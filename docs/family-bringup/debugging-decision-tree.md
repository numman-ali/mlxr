# Family Debugging Decision Tree

Status: reusable debugging and validation ladder for new-family bring-up, family re-bases, and "it runs but the output is wrong" failures.

## Why This Exists

A family bring-up can fail in at least four different ways:

- the source contract is misunderstood
- the portable artifact contract is wrong
- the runtime math is wrong
- the canonical CLI/runtime path is different from the backend-only probe

This note exists so we do not treat all of those as one generic "quality problem."

## Core Ladder

Use this order:

1. source truth
2. artifact truth
3. component parity
4. canonical runtime/CLI smoke
5. meaningful semantic or quality rung

If a lower rung is still uncertain, do not promote the higher rung.

## Quick Map

```text
symptom
  |
  +-- load/shape error?
  |     |
  |     +-- yes -> check config, key coverage, tensor layout, dtype, aliasing
  |     |
  |     +-- no
  |
  +-- output artifact is blank / banded / scrambled?
  |     |
  |     +-- yes -> isolate scheduler, latent scaling, decode path, output encode
  |     |
  |     +-- no
  |
  +-- output is coherent noise or wrong semantics?
  |     |
  |     +-- yes -> isolate conditioning path from denoise/decode path
  |     |
  |     +-- no
  |
  +-- backend probe works but CLI/runtime path fails?
        |
        +-- yes -> inspect workflow planning, job params, output/export path, host mapping
        |
        +-- no -> promote only after a meaningful semantic rung
```

## Step 1: Lock Source Truth

Before writing runtime code, answer these explicitly:

- Which upstream variants are actually released?
- Which files are required to run the released slice?
- Which component roles exist in the checkpoint layout?
- Which parts are runtime-critical and which are optional?
- Which dtype, scheduler, tokenizer, and prompt-template assumptions are real?

Use primary sources first:

- official repo
- released Hugging Face or provider checkpoint
- checkpoint config and weight index

Do not treat README marketing claims as the runnable contract.

## Step 2: Lock Artifact Truth

Before debugging model math, verify:

- the family adapter inspects the source truthfully
- conversion preserves the correct component roles
- load-time fail-closed checks reject missing or mismatched payloads
- component dtype policy is deliberate, not accidental

If the artifact contract is wrong, the runtime can look "mathematically broken" when the bug is really in conversion.

## Step 3: Use Component Parity Before Full Inference

Do not jump straight to full generations if the family has multiple moving parts.

Use the smallest parity checks first:

- tokenizer formatting parity
- token ids and attention-mask parity
- prompt-embedding parity
- tiny-module forward parity against a reference implementation
- VAE decode parity on synthetic latents
- scheduler timestep parity

The goal is to find the first rung where outputs diverge.

## Step 4: Isolate By Swapping One Boundary At A Time

When the output exists but is obviously wrong, do not keep changing everything at once.

Use a swap matrix:

```text
owned prompt  + owned image stack
reference prompt + owned image stack
owned prompt  + reference-adjacent image stack
canonical CLI/runtime path + same model bundle
```

Interpretation:

- if reference prompt fixes the image, the conditioning path is the lead suspect
- if reference decode fixes the image, the output path is the lead suspect
- if backend-only works but CLI/runtime fails, the workflow or job layer is suspect

## Step 5: Promote The Canonical Path, Not The Probe

A direct backend probe is evidence.

The product claim should come from the canonical path:

- source registration
- artifact conversion
- workflow planning
- job submission
- runtime-managed output
- CLI or host adapter behavior

If the family works only in a scratch script, it is not landed yet.

## What To Record While Debugging

Keep the evidence that future sessions need:

- exact checkpoint variant and path
- exact prompt and seed
- exact resolution and inference steps
- exact artifact path
- job id and runtime log path
- what swap or isolation test changed the result
- whether the result is a smoke rung or a meaningful quality rung

## Z-Image Example Pattern

`Z-Image-Turbo` exposed the right pattern for "it runs but the image is wrong."

The successful path was:

1. confirm the official released variants and checkpoint layout
2. prove the MLX transformer and VAE matched tiny PyTorch reference modules
3. fix clear runtime-contract mismatches first
4. isolate prompt conditioning from image generation by swapping in reference embeddings
5. confirm the canonical `mlxr generate` path after the prompt fix

The concrete issues we hit were:

- scheduler endpoint mismatch
- VAE decode precision mismatch
- formatted prompt mutation before tokenization
- padding-side and padded-mask mismatch
- prompt extraction logic that assumed the wrong padding contract

The decisive isolation test was:

```text
reference prompt embeddings + owned MLX image stack -> coherent image
owned prompt embeddings + owned MLX image stack     -> broken image
```

That proved the main blocker was the prompt path, not the image stack.

## Fresh-Eyes Questions

Before calling a family fix "done", ask:

- Did we prove the canonical runtime path or only a probe?
- Did we isolate the first bad boundary or just improve the picture?
- Did we capture the debugging ladder somewhere durable?
- Did we update the skill/docs so the next family starts from a better place?
