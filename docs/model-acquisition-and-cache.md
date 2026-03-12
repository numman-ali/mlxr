# Model Acquisition And Cache

Status: product-direction note for the install surface, cache policy, and safe local execution defaults.

## Purpose

`MLXR` should make model installation feel simple without collapsing three different storage concerns into one blurry "model folder".

Those three layers are:

1. source cache
2. portable artifact store
3. machine-local execution cache

They are related, but they are not the same thing.

## Default Source Behavior

For Hugging Face-backed models, the default user path should reuse the normal Hugging Face cache behavior instead of inventing a second private download cache.

The product default should be:

- resolve the supported model by stable `MLXR` model identity
- fetch source files through the Hugging Face provider
- reuse the user or machine's normal Hugging Face cache when possible
- preserve the resolved revision, license, access state, and remote-code policy in provenance

This keeps `MLXR` compatible with normal open-source expectations:

- `hf auth login` keeps working
- gated model access stays understandable
- users do not need to learn a second download tool before they can generate

## Why PyTorch Names Still Appear

Many upstream Hugging Face bundles use filenames like:

- `diffusion_pytorch_model.safetensors`
- `model.safetensors`
- `pytorch_model.bin`

Those names describe how the source bundle was packaged for the wider ecosystem.

They do not, by themselves, say anything about the runtime `MLXR` uses to execute the model.

In `MLXR`, the important split is:

```text
upstream bundle contract
    !=
runtime execution engine
```

So a file named `diffusion_pytorch_model.safetensors` can still be loaded into a fully owned MLX runtime as long as:

- we know the tensor naming contract
- we know the model config contract
- we map those tensors into our own MLX modules

`safetensors` is just a tensor container format. The filename may be PyTorch-flavored, but the file contents are still just named tensors.

That is why `MLXR` can truthfully stay "owned MLX runtime" while still consuming upstream `diffusers` / `transformers`-shaped source bundles.

## The Three Storage Layers

```text
Hugging Face repo / local bundle
            |
            v
   source cache or provider snapshot
            |
            v
   portable MLXR artifact
            |
            v
   machine-local execution cache
```

### 1. Source Cache

This is the downloaded upstream source material.

Examples:

- Hugging Face cached snapshot
- trusted local bundle
- future registry-backed source snapshot

This layer should preserve upstream identity and provenance.

This is also the right place to preserve upstream bundle shape and filenames exactly as they came from the provider, even when those filenames mention `pytorch`.

### 2. Portable Artifact Store

This is the `MLXR`-normalized family artifact that the runtime actually knows how to load and execute.

It should:

- be componentized and family-aware
- preserve provenance back to the source revision
- be reusable across machines when the family format and license allow it
- be the thing a registered `MLXR` model points at

This is the right place for conversion outputs, not the provider cache.

### 3. Machine-Local Execution Cache

This is any machine-specific optimized state built for speed on a given Mac.

Examples:

- compiled kernels
- quantized execution-side blobs
- future MLX compile outputs
- warm execution metadata

This layer can be invalidated more aggressively than portable artifacts.

## Product Default Versus Bring-Up Tooling

During family bring-up, it can still be useful to materialize a Hugging Face repo into a temporary local mirror such as `tmp/hf-...-live/` for surgical inspection.

That is a debugging and research tool, not the desired product experience.

The normal product path should prefer:

- provider-managed Hugging Face cache for source downloads
- runtime-managed artifact creation under `$MLX_RUNTIME_HOME/artifacts-portable/`
- runtime-managed machine-local cache under `$MLX_RUNTIME_HOME/build-cache-local/`

If a temporary local mirror is used for debugging, the repo should treat it as disposable.

## Install Surface Direction

The intended product behavior should be idempotent.

If a user asks for a supported model and it is already installed, `MLXR` should reuse it.

If the source revision changed or the artifact format changed, `MLXR` should make the necessary refresh explicit instead of silently mutating identity.

The current install surface is growing toward commands like:

```text
mlxr models install <model>
mlxr models list
mlxr doctor
```

Later follow-ons should include:

```text
mlxr models update <model>
mlxr models prune
```

And the generation surface should eventually allow:

```text
mlxr generate --model-id <model> ...
```

to auto-install on first use when policy allows it.

## Auth And Gated Models

For gated Hugging Face sources, the normal path should be:

- user runs `hf auth login`
- `MLXR` reuses that auth state through the Hugging Face provider path
- if auth is missing, the runtime fails clearly and tells the user what to do

Advanced users should still be able to:

- provide a local bundle path
- point at a specific repo revision
- register a local converted artifact explicitly

But those are advanced paths, not the default story.

## Artifact Export And Re-Publishing

Portable artifacts should be stored in a way that supports future export and sharing when license terms allow it.

That means:

- the artifact must carry source provenance
- the artifact identity must stay stable
- license and access state must be recorded
- machine-local cache must stay separate so we do not accidentally publish host-specific build products as if they were portable artifacts

Future publish or export commands should operate on portable artifacts, not provider cache directories.

## LoRA And Adapter Direction

The same three-layer split should apply to reusable LoRAs and other adapters.

Ad hoc CLI use such as:

```text
mlxr generate ... --lora /path/to/adapter.safetensors
```

is still useful and should remain supported.

But the long-term product contract for large reusable adapters should not be
"re-upload the whole safetensors blob on every run."

The better direction is:

- keep Hugging Face snapshots or trusted local files as the source layer
- register adapters as stable runtime-managed identities with provenance
- let jobs attach adapters by stable id, path-backed source ref, or trusted
  local handle when appropriate

In plain English:

```text
small one-off adapter = ad hoc input is fine
large reusable adapter = install/register once, attach many times
```

For big LoRAs, pointing at a pinned Hugging Face snapshot is much closer to
official ecosystem behavior than repeatedly shipping the adapter through a job
request. Libraries such as `diffusers` and LightX2V typically load adapters
from local paths or snapshot-backed references, then merge or apply them in
memory at inference time.

That means the medium-term `MLXR` shape should likely grow toward commands such
as:

```text
mlxr adapters install <adapter>
mlxr adapters list
mlxr adapters attach <adapter-id>
```

before treating giant LoRAs as ordinary transient inputs forever.

## Large Local Input Transport

When trusted local users do pass a large input file directly to the runtime,
the transport should prefer streamed local upload or trusted local reference
passing rather than JSON base64 payloads.

That is especially important for large LoRA safetensors, because the adapter can
be gigabytes in size while still being a perfectly normal local workflow input.

So the safe product rule is:

- small inputs can tolerate simpler transports
- large local files should use streamed or reference-based import paths
- reusable large adapters should usually be installed or registered once

## What Belongs In Core

`MLXR` should not rush to build one giant internal clone of Hugging Face `diffusers`.

The better rule is:

- keep family-specific execution logic inside the family package first
- move code into shared core only after at least two families need the same thing

Good candidates for shared core:

- tokenizer loading and typed tokenization helpers
- text-model substrates reused by multiple families
- generic tensor-loading helpers
- scheduler primitives that truly generalize
- machine-local cache and artifact lifecycle policy

Bad candidates for premature core extraction:

- one family's exact transformer block layout
- one family's reference-conditioning quirks
- one family's prompt-enhancement workflow
- one family's editing semantics

In plain English:

```text
shared substrate in core
family semantics in family packages
```

That lets `MLXR` stay performant and honest without coupling every future family to the first image or video engine we happen to bring up.

## Safe Local Execution Policy

For heavyweight image and video generation on Apple Silicon, the default local policy should be conservative:

- queueing is fine
- concurrent user submissions are fine
- but only one heavyweight generation job should actively execute per worker
- and the default local scheduler policy should avoid co-resident loading of multiple big model stacks when that would risk memory pressure or crashes

In plain English:

```text
many requests may queue
one heavy image/video job should run at a time per worker
do not load multiple giant model weights at once by default
```

This matches the current architecture direction in the runtime docs: one heavy media generation job per worker, with broader concurrency policy treated as a measured optimization problem rather than an optimistic default.

## Current Session Note

The current Qwen bring-up path already showed why naming clarity matters:

- `tmp/Qwen-Image-2512` is only a symlink
- it points to `tmp/hf-qwen-image-2512-live`
- that local mirror exists for inspection and family bring-up

That should not become the long-term product contract. The long-term contract should name the supported `MLXR` model cleanly and let the provider layer manage source download details underneath it.
