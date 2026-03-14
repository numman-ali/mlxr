# Repo-Owned Multimodal Inference Engine

Status: supporting research recommendation for the next shared execution substrate in `MLXR`.

This document supports the owned-substrate migration and shared-core direction.
It is not the canonical family truth map, and it is not the active migration
checklist.
Read it alongside:

- [ltx-owned-substrate-migration-plan.md](../working/ltx-owned-substrate-migration-plan.md)
- [ltx-reference-map.md](../families/ltx-reference-map.md)
- [ltx-compatibility-checklist.md](../families/ltx-compatibility-checklist.md)

Current progress note:

- the first real shared package from this plan now exists at
  `packages/core/runtime-mlx-models/`
- the current LTX Gemma prompt path uses that repo-owned substrate instead of
  `mlx-vlm` / `mlx-lm` at runtime
- the remaining work here is to broaden that substrate carefully where the
  ownership is genuinely shared across families

## Executive Recommendation

`MLXR` should build one repo-owned MLX execution substrate for text, VLM, and broader multimodal understanding workloads instead of depending on `mlx-lm` or `mlx-vlm` at runtime.

That substrate should live in `packages/core/`, not inside one family package.

Recommended direction:

- use `mlx-lm` as an official reference and donor for text-generation mechanics
- use `mlx-vlm` as an ecosystem reference and donor for multimodal tower and projector patterns
- do **not** make either package a required runtime dependency for the `MLXR` inference path
- keep family truth in family adapters, but move reusable LM and multimodal execution logic into repo-owned shared packages

The right mental model is:

```text
control plane + workflows + scheduler
            |
            v
  repo-owned MLX model + engine substrate
  - architecture registry
  - model loading and weight sanitation
  - multimodal input preparation
  - prefill and decode
  - cache and session reuse
  - low-level MLX optimization seam
            |
            +--> text family adapter
            +--> VLM family adapter
            +--> omni understanding family adapter
            +--> selected prompt-encoding paths for media families
```

This is a better fit for `MLXR` than either:

- treating `mlx-vlm` as the architecture
- or copying `mlx-lm` into multiple families as a hidden dependency

## Why This Should Exist

The current repo already has the platform seams needed for a second family:

- `ModelFamilyAdapter` already supports source inspection, conversion, load, capability reporting, stage execution, and unload
- the workflow planner already dispatches by family
- the capability schema already has room for `interactive VLM or text session`

What the repo does **not** have yet is a shared repo-owned inference substrate below those families.

Today the clearest prompt-side leak is the LTX backend importing external LM and VLM code directly:

- [`packages/families/ltx/src/mlxr/families/ltx/_prompt_encoding_backend/runtime.py`](../../packages/families/ltx/src/mlxr/families/ltx/_prompt_encoding_backend/runtime.py)

That import pattern is acceptable as a temporary bridge, but it is the wrong end state if the goal is:

- full control of the inference engine
- lower-level MLX optimization
- one coherent runtime story across text and multimodal families

## What `mlx-lm` Gets Right

`mlx-lm` is an official and useful reference, but it is best understood as a strong text-generation shell rather than a unified multimodal engine.

The parts worth borrowing conceptually are:

- config-driven model loading and weight sanitation
- prompt prefill versus decode-step separation
- explicit KV-cache implementations
- quantized KV-cache and quantization policy hooks
- sampler and logits-processor composition
- batch-oriented cache reuse ideas

The best donor surfaces live in:

- [`references/official/mlx-lm/mlx_lm/utils.py`](../../references/official/mlx-lm/mlx_lm/utils.py)
- [`references/official/mlx-lm/mlx_lm/generate.py`](../../references/official/mlx-lm/mlx_lm/generate.py)
- [`references/official/mlx-lm/mlx_lm/models/cache.py`](../../references/official/mlx-lm/mlx_lm/models/cache.py)

But `mlx-lm` remains text-first in ways that matter:

- the server and API shape are tokenizer and chat oriented
- multimodal support is mostly reduced to feeding input embeddings into a language model
- conversion and serving assumptions are centered on text-family ergonomics

So the right use of `mlx-lm` is:

- study it
- borrow patterns
- reimplement the shared substrate inside `MLXR`

The wrong use is:

- make it the hidden runtime base layer for every future text and VLM family

## Why `mlx-vlm` Is Separate From `mlx-lm`

The code structure strongly suggests that `mlx-vlm` exists because the ecosystem extended a text-generation center of gravity instead of designing a modality-agnostic engine from the start.

In practice, `mlx-vlm` is mostly:

- multimodal model loading
- Hugging Face processor integration
- tower and projector glue
- prompt and message formatting
- a shared pattern of "prepare multimodal embeddings, then run LM decode"

Useful reference points:

- [`references/ecosystem/mlx-vlm/mlx_vlm/utils.py`](../../references/ecosystem/mlx-vlm/mlx_vlm/utils.py)
- [`references/ecosystem/mlx-vlm/mlx_vlm/generate.py`](../../references/ecosystem/mlx-vlm/mlx_vlm/generate.py)
- [`references/ecosystem/mlx-vlm/mlx_vlm/models/base.py`](../../references/ecosystem/mlx-vlm/mlx_vlm/models/base.py)

That makes `mlx-vlm` valuable as research input, but it also exposes the main limitations `MLXR` should avoid:

- heavy dependence on `mlx-lm` internals
- heavy dependence on Hugging Face processor conventions
- one-model-at-a-time serving assumptions
- OpenAI-chat-shaped server UX leaking into engine structure
- video and audio support treated as extensions of an image-VLM stack instead of first-class engine concepts

So the answer to "why is `mlx-vlm` separate?" is:

- because `mlx-lm` is fundamentally a text-serving shell
- and `mlx-vlm` layers multimodal model prep on top of that shell

The answer for `MLXR` should be different:

- one repo-owned set of shared execution mechanics for text and multimodal families
- with architecture-specific multimodal composition handled explicitly instead of hidden behind a fake single-path abstraction

## Recommended Package Shape

The shared substrate should live under `packages/core/`.

Recommended new package seams:

- `packages/core/runtime-mlx-engine/`
- `packages/core/runtime-mlx-models/`
- `packages/core/runtime-mlx-ext/`
- `packages/core/runtime-kernels/`

Recommended ownership split:

### `packages/core/runtime-mlx-engine/`

Owns reusable MLX-native execution primitives:

- state and cache abstractions
- prefill and decode mechanics
- samplers and logits processors
- streaming detokenization and output assembly
- session and state reuse policies
- memory-budget policy for interactive and batch workloads

This package should own the engine mechanics, not the entire model zoo.

### `packages/core/runtime-mlx-models/`

Owns the shared architecture registry and the concrete model implementations that do not belong to one family:

- architecture model registry
- config-to-class resolution
- per-architecture model modules
- weight sanitization hooks
- architecture-specific quantization predicates
- multimodal composition helpers such as projector wiring or side-channel setup

This package exists because the real implementation burden is not just one engine loop. It is also the set of concrete architecture modules that today live under `mlx_lm.models.*` and `mlx_vlm.models.*`.

`runtime-mlx-engine` should stay stable and mechanical.
`runtime-mlx-models` should be allowed to evolve with model architectures.

### `packages/core/runtime-mlx-ext/`

Owns profiled native extensions only after benchmark proof:

- custom MLX primitives
- C++ helpers
- native wrappers around durable hotspots

This matches the seam already reserved in:

- [`docs/adr/0005-mlx-extension-and-upstream-roadmap.md`](../adr/0005-mlx-extension-and-upstream-roadmap.md)

### `packages/core/runtime-kernels/`

Owns focused Metal kernels once a hotspot is proven durable enough to keep.

### `packages/families/<family>/`

Continues to own family truth:

- source-role truth
- checkpoint truth
- artifact-conversion truth
- capability truth
- workflow truth
- family-specific fail-closed boundaries

Families should not become the hidden home of shared LM or VLM engine logic.

## Engine Contract

The shared engine should define a stable internal contract that families adapt into, but it should not pretend all multimodal models compose in one linear pipeline.

The earlier sketch of:

- load components
- encode modalities
- build conditioning
- prefill
- decode step

is useful as orientation only. It is not precise enough to be the real contract.

The real ecosystem shape is closer to:

1. architecture registry and model construction
2. multimodal input preparation
3. prefill and decode loop
4. cache and session-state lifecycle

### Architecture Model Registry

The engine needs a shared architecture registry that can:

- resolve `config.json` into a model class
- instantiate the correct text, vision, audio, or projector submodules
- run architecture-specific weight sanitization
- apply architecture-specific quantization predicates

This is the same real problem solved today by:

- [`references/official/mlx-lm/mlx_lm/utils.py`](../../references/official/mlx-lm/mlx_lm/utils.py)
- [`references/ecosystem/mlx-vlm/mlx_vlm/utils.py`](../../references/ecosystem/mlx-vlm/mlx_vlm/utils.py)

Current recommendation:

- keep that registry repo-owned
- put it in `runtime-mlx-models`
- let families decide which architectures and checkpoints they support

### Multimodal Input Preparation Contract

The most important reusable abstraction in the current VLM ecosystem is not `encode_modalities()` alone. It is the broader pattern currently expressed as:

- `get_input_embeddings(...)`
- returning a structured feature bundle such as `InputEmbeddingsFeatures`

This matters because real VLMs compose in materially different ways.

## Multimodal Architecture Composition Patterns

The engine must support at least these patterns.

### Pattern: Token Replacement

Vision features are projected into language-model hidden space and replace placeholder token positions in the embedding sequence before the language model runs.

Examples:

- LLaVA
- Gemma 3 vision variants
- InternVL-style models

### Pattern: Cross-Attention Injection

The vision tower produces `cross_attention_states` and masks that must be available during prefill and every decode step.

Examples:

- `mllama`-style vision models

### Pattern: Multimodal RoPE

Multimodal input preparation modifies the model's positional-encoding state for all subsequent decode steps.

Examples:

- `Qwen2-VL`
- `Qwen2.5-VL`

### Pattern: Per-Layer Inputs

Modality features are routed to each transformer layer separately rather than only through one merged embedding tensor.

Examples:

- `Gemma3n`

### Pattern: Encoder-Decoder Generation

The model does not use the same pure causal-LM loop as a standard text model and may require decoder inputs or encoder outputs as separate state.

Examples:

- `Florence2`-style models

### Implication For The Engine Contract

The engine should define a structured feature bundle contract instead of a fake single-path multimodal API. A more realistic shape is:

```python
class MultimodalFeatures(Protocol):
    inputs_embeds: object | None
    attention_mask: object | None
    attention_mask_4d: object | None
    cross_attention_states: object | None
    cross_attention_mask: object | None
    per_layer_inputs: object | None
    decoder_inputs_embeds: object | None
    metadata: dict[str, object]


class MultimodalModelRuntime(Protocol):
    def prepare_multimodal_inputs(self, request, state) -> MultimodalFeatures: ...
    def make_cache(self, loaded, session_state=None): ...
    def prefill(self, loaded, features, cache, state): ...
    def decode_step(self, loaded, features, cache, sampler_state, state): ...
    def release_state(self, loaded, state): ...
```

The exact types should become repo-owned later. The important point now is that:

- multimodal side-channels must survive through prefill and decode
- the engine contract must not discard architecture-specific state

### Cache Contract

The cache contract must be architecture-aware from the start.

It should explicitly leave room for:

- plain KV cache
- rotating KV cache
- quantized KV cache
- cross-attention state persistence
- array-style state for non-standard sequence models
- prompt-cache trim, filter, extend, and merge operations

That keeps room open for:

- interactive prefix reuse
- continuous batching
- speculative decoding

### Inversion Of Control

The intended ownership split is:

- family adapters own checkpoint truth, capability truth, workflow truth, and fail-closed compatibility boundaries
- the engine owns shared model-runtime mechanics
- architecture model modules own architecture-specific forward composition

So in practice:

- `ModelFamilyAdapter.run_stage()` should call into the shared engine where the work is generic
- but the engine should call architecture-specific runtime modules for multimodal input preparation and model execution details

That inversion-of-control question is real and should remain explicit during implementation.

## Tokenizer And Processor Strategy

The runtime goal is "no `mlx-lm` or `mlx-vlm` runtime dependency." That does not automatically settle the tokenizer and processor question.

### Tokenizer Ownership

The first shared-core tokenizer slice should stay thin and explicit.

The engine should own a repo-owned tokenizer wrapper abstraction that provides:

- loading a local tokenizer artifact
- tokenization into typed `input_ids` and `attention_mask` arrays
- basic detokenization

That is now the implemented policy in `packages/core/runtime-mlx-models/` for
the current Gemma/LTX path. It deliberately does not yet widen into a full
`mlx-lm`-style tokenizer surface with streaming detokenization, chat-template
rendering, EOS-set policy, or tool/structured-output token rules until another
family proves those belong in shared core.

### Processor Ownership

VLM families require inference-time processors for:

- resize
- normalize
- layout conversion
- modality-specific preprocessing

For the first slice, the most pragmatic stance is:

- allow `transformers` as an explicit runtime dependency for tokenizer and processor loading
- keep that separate from the decision to avoid `mlx-lm` and `mlx-vlm`

Current repo truth:

- tokenizer loading is shared-core and repo-owned through a thin wrapper
- processor loading and multimodal preprocessing are still an explicit separate
  decision

Longer-term, `MLXR` may internalize more processor logic. But the first design
doc should be honest that repo-owned model execution does not automatically
mean repo-owned processor implementations on day one.

### Prompt Formatting

Prompt formatting for text and VLM families should not live in hosts and should not be forced into one universal format string.

The better split is:

- engine owns the wrapper abstraction
- families or architecture modules own model-specific prompt-format rules

### Artifact Boundary

Tokenizer and processor artifacts should be treated as part of the portable artifact contract when they are required for truthful local inference.

## Runtime Contract Changes This Implies

The current core contracts are close, but they are still too video-shaped for a shared text and VLM engine.

### 1. Workflow Intent Should Become More Multimodal-Native

Current workflow intent is centered on:

- one `prompt`
- media references by kind

That is a good fit for LTX, but it is awkward for:

- interleaved content parts
- multi-image conversational VLM turns
- text plus image plus audio understanding prompts
- structured conversational state

The likely direction is:

- preserve the simple media path for current LTX workflows
- add a generalized multimodal content-part model for text and VLM families

Relevant current schema:

- [`packages/core/shared-schemas/src/mlxr/core/schemas/workflows.py`](../../packages/core/shared-schemas/src/mlxr/core/schemas/workflows.py)

## Multimodal Content Schema

The content schema should be additive, not an immediate replacement for media-oriented `WorkflowIntent`.

Recommended direction:

```python
class ContentPart(BaseModel):
    type: Literal["text", "image", "audio", "video"]
    text: str | None = None
    input_handle: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: list[ContentPart]


class MultimodalWorkflowIntent(BaseModel):
    model_id: str
    messages: list[ConversationTurn]
    params: dict[str, Any] = Field(default_factory=dict)
    output: JobOutputPolicy = Field(default_factory=JobOutputPolicy)
```

Recommended coexistence rule:

- keep the current `WorkflowIntent` for LTX and other media-generation families
- add a conversation/content-part path for text and VLM families
- let workflow strategy and scheduler class determine which schema shape applies

### 2. Worker Execution Should Become Stage-Graph Driven

Today the worker is still effectively centered on a fixed sequence shaped like:

- `prompt_encode`
- `condition_inputs`
- `generate`
- `encode_output`

That is sufficient for the current LTX proving path, but it is not the right final shape for:

- text generation
- VLM understanding
- speech or omni understanding families

The long-term direction should stay aligned with:

- [`docs/workflow-orchestration-design.md`](../workflow-orchestration-design.md)

Specifically:

- the workflow layer should declare stage graphs
- the worker should interpret family-declared stage graphs
- the shared engine should own reusable stage mechanics below that layer

This is not just a parallel cleanup item. It is a prerequisite for validating the engine on text and VLM families.

### 3. Scheduler Classes Should Expand Beyond Heavy Media Video

The capability schema already has room for interactive text and VLM workloads, but runtime policy is not yet using that breadth truthfully.

Recommended new scheduler classes to validate next:

- `interactive_text`
- `interactive_vlm`
- optional later `omni_understanding`

This matters because:

- concurrency policy differs sharply from video DiT generation
- warm state reuse differs sharply from clip generation
- token streaming and session reuse deserve first-class admission logic

## Worker Topology For Interactive Families

The worker-topology question can no longer stay implicit once the repo adds an interactive text or VLM engine.

Today the runtime direction is:

- control plane plus execution workers
- worker granularity not yet frozen

Relevant current sources:

- [`docs/adr/0004-worker-process-topology.md`](../adr/0004-worker-process-topology.md)
- [`packages/core/runtime-server/src/mlxr/core/server/worker.py`](../../packages/core/runtime-server/src/mlxr/core/server/worker.py)

The current per-job subprocess path is a good default for heavy media generation, but it is a poor fit for interactive text and VLM serving because those workloads want:

- warm model residency
- prompt prefix reuse
- long-lived cache state
- low first-token latency
- optional continuous batching

The design direction this document recommends is:

1. keep a batch-style worker mode for heavy media families
2. add a persistent interactive worker mode for `interactive_text` and `interactive_vlm`
3. drive that distinction from `scheduler_class`, not from host behavior

### 4. Outputs Should Not Be File-Only In Spirit

The current job system already has event streaming and delta concepts, but text and VLM workloads need the runtime contract to treat non-file outputs as first-class:

- token deltas
- structured JSON
- final text
- session-state handles or reusable caches

That should sit alongside artifact outputs, not outside the runtime contract.

### 5. Memory Budget Policy Should Be Engine-Owned

The engine should explicitly own memory-budget and cache-lifecycle policy for interactive and batch workloads:

- `mx.set_wired_limit(...)` policy where appropriate
- `mx.clear_cache()` lifecycle
- per-scheduler-class memory envelopes
- telemetry for active, peak, and cache memory

## Dependency Policy

The runtime goal for this engine should be:

- no `mlx-lm` runtime dependency
- no `mlx-vlm` runtime dependency
- no host-owned inference logic

Recommended interpretation of "no external dependencies except MLX":

- the execution engine itself should be repo-owned and built directly on `mlx`
- checkpoint or source conversion may still use provider tooling during artifact creation when necessary
- but inference-time cache policy, sampling, and model execution should live in repo-owned code
- tokenizer and processor dependencies need an explicit separate decision instead of being blurred into the `mlx-lm` / `mlx-vlm` question

This keeps `MLXR` honest about the difference between:

- source-resolution dependencies
- artifact-conversion tooling
- the actual runtime execution engine

## Low-Level Optimization Path

`MLXR` should explicitly reserve a three-tier optimization ladder.

### Tier 1: Stock MLX First

Start with:

- normal MLX ops
- `mx.compile` on stable inner loops
- explicit streams
- live memory telemetry

This is already compatible with the repo's current architecture doctrine and optimization playbook:

- [`docs/research/05-optimization-playbook.md`](../research/05-optimization-playbook.md)

### Tier 2: Targeted `fast.metal_kernel` Prototypes

When profiling proves a hotspot, move it into targeted custom Metal kernels first.

Likely early candidates for text and multimodal families:

- multimodal gather or scatter
- projector-heavy tensor transforms
- patch packing or unpacking
- modality-specific masking transforms
- cache compaction or cache-layout transforms

### Tier 3: Full MLX Primitives Or Native Extension Packages

Only promote a hotspot into repo-owned C++ MLX primitives when:

- the win is benchmarked
- the kernel is durable across multiple families
- the maintenance cost is justified

The relevant MLX seams are already real today:

- custom extensions
- custom primitives
- custom Metal kernels
- streams
- memory telemetry
- `mlx-c` bridge surfaces

The memory-policy part matters too. The engine should own policy for:

- wired limit usage
- memory limit usage
- cache clearing between phases
- per-scheduler-class warm-state budgets

Useful references:

- [`references/official/mlx/docs/src/dev/extensions.rst`](../../references/official/mlx/docs/src/dev/extensions.rst)
- [`references/official/mlx/docs/src/dev/custom_metal_kernels.rst`](../../references/official/mlx/docs/src/dev/custom_metal_kernels.rst)
- [`references/official/mlx-c/README.md`](../../references/official/mlx-c/README.md)
- [`references/official/mlx-lm/mlx_lm/generate.py`](../../references/official/mlx-lm/mlx_lm/generate.py)

## Quantization Strategy

Quantization needs to be treated as part of the engine architecture, not as a secondary implementation detail.

The engine should explicitly leave room for:

- weight quantization at artifact-conversion time
- architecture-specific quantization predicates
- multimodal exclusions such as keeping some vision or audio towers in higher precision
- runtime KV-cache quantization after a sequence-length threshold
- optional activation quantization on compatible layers

## Continuous Batching And Speculative Decoding

The engine contract should not preclude either of these, even if they land after the first text-only validation rung.

Named future engine capabilities:

- speculative decoding with draft-model verification
- continuous batching with prompt-cache aware in-flight sequence management

This is another reason the cache contract must start life with:

- trim
- filter
- extend
- merge

rather than a single minimal append-only design.

## Fine-Tuning And Training Boundary

This document is inference-first on purpose.

Training and fine-tuning matter, but they should not define the runtime package shape prematurely.

Recommended future direction:

- keep training and fine-tuning out of `runtime-core`
- keep them out of host adapters
- keep them out of the family adapter contract except where artifact compatibility must be declared

If `MLXR` later grows repo-owned training or fine-tuning support, the best home is likely a new top-level role bucket:

- `packages/training/`

That bucket would be allowed to reuse pieces of `runtime-mlx-engine`, such as:

- model-component loading
- tokenizer or processor artifacts
- quantization-aware weight transforms
- some shared module definitions

But the runtime should not inherit:

- trainer lifecycle concerns
- optimizer state semantics
- checkpoint-save policy
- dataset or recipe abstractions

The short rule is:

- inference engine first
- training stack later
- shared modules only where they are truly reusable

## Smallest Truthful First Slice

Recommended first slice should be split into three smaller sub-rungs:

1. text-only engine slice
2. one VLM family on top of that engine slice
3. only then the first LTX prompt-side reuse slice

Practical breakdown:

### Slice 1a: Text-Only Engine

- load one causal LM through repo-owned engine code
- prefill and decode through repo-owned cache and sampler logic
- stream token deltas through the current runtime event path

### Slice 1b: VLM Engine

- add one multimodal architecture
- add multimodal input preparation and processor handling
- validate one image-conditioned understanding path

### Slice 1c: LTX Prompt-Side Reuse

- replace the current `mlx_lm` and `mlx_vlm` prompt-side dependency leak where the repo-owned engine now offers the truer shared primitive
- keep LTX generation itself separate from this scope unless the engine charter expands beyond autoregressive and VLM workloads

Good candidate validating families:

- text: one causal LM family
- VLM: one model such as `Qwen2.5-VL`, `Gemma 3n`, or similar

This stays aligned with the cross-family validation pressure already recorded in:

- [`docs/research/02-model-research-matrix.md`](../research/02-model-research-matrix.md)
- [`docs/research/09-open-questions-and-validation-plan.md`](../research/09-open-questions-and-validation-plan.md)

## Validation Ladder

The engine should not be treated as real until it clears these rungs.

### Rung 1: Text Runtime Parity

- load one text model through repo-owned engine code
- prefill and decode through repo-owned cache and sampler logic
- stream token deltas through the existing runtime event path

### Rung 2: VLM Runtime Parity

- load one VLM family through the same shared engine substrate
- run image-conditioned understanding without `mlx-vlm`
- prove that modality encoding, cache policy, and decode work through the same engine contract

### Rung 3: LTX Prompt-Side Integration

- replace current prompt-side donor imports where the shared engine is now the truer home
- keep the family adapter owning checkpoint truth and fail-closed compatibility checks

### Rung 4: Scheduler Truth

- add truthful scheduler classes for text and VLM
- benchmark warm reuse and memory envelopes
- prove that the control plane can schedule these families without video-specific assumptions

### Rung 5: Low-Level Proof

- benchmark stock MLX
- benchmark compiled inner loops
- benchmark any custom Metal kernel candidate
- promote native work only where measured wins justify it

### Rung 6: Speculative Decoding Parity

- validate draft-model verification on the repo-owned engine without changing the core engine contract later

### Rung 7: Continuous Batching Parity

- validate concurrent decode on a persistent interactive worker with truthful cache and latency behavior

## Relationship To The Current `mlx-video` Dependency

This document is primarily about a shared text and multimodal understanding engine. It should not pretend that this automatically internalizes the current LTX generation-side donor dependency on `mlx-video`.

Relevant current path:

- [`packages/families/ltx/src/mlxr/families/ltx/_generation_backend/runtime_helpers.py`](../../packages/families/ltx/src/mlxr/families/ltx/_generation_backend/runtime_helpers.py)

Current recommendation:

- treat `mlx-video` internalization as a separate but related donor-replacement track
- let the new shared engine first solve text, VLM, and prompt-side reusable model-runtime mechanics
- expand the charter later only if the repo deliberately decides to unify diffusion and autoregressive engine work at a lower layer

## Open Questions That Should Stay Explicit

- Does `MLXR` fully own and maintain a repo-owned model zoo, or does it support a thinner compatibility path for imported model definitions during a transition period?
- Is `transformers` acceptable as a runtime dependency for tokenizer and processor loading in the first slice?
- Exactly where should inversion of control sit between family adapter, architecture runtime module, and shared engine loop?
- Does the long-term engine charter include `mlx-video` replacement, or is that a separate platform track?
- Are speculative decoding and continuous batching in scope for the first stable text-serving slice, or only required as contract-preserving follow-on work?
- Is SSE enough for high-frequency token streaming, or will the interactive path need a different transport later?

## What This Document Recommends Against

Do not:

- make `mlx-vlm` the architecture
- make `mlx-lm` the hidden runtime substrate for every new family
- let host-facing OpenAI or chat compatibility shape the engine internals
- treat Hugging Face processor conventions as the canonical runtime contract
- hide shared execution logic inside one family package
- conflate training-stack needs with inference-engine boundaries before the inference core is real

## Bottom Line

The right next inference-engine move for `MLXR` is:

- one repo-owned multimodal MLX engine and model-registry layer in `packages/core/`
- text and VLM as validating families on top of it
- `mlx-lm` and `mlx-vlm` treated as research inputs, not runtime foundations
- C++ and Metal reserved as profiled extension seams, not the starting point

That gives `MLXR` the thing this repo actually wants:

- full control of the inference engine
- shared runtime logic across families
- the ability to optimize directly into MLX when the measurements justify it
- and a cleaner long-term path than either a text-only shell or a pile of multimodal patches
