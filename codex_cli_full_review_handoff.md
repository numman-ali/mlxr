# Rewrite Brief for GPT-5.4 Codex CLI: Universal MLX Runtime / LTX Platform

**Audience:** the GPT-5.4 Codex CLI agent that authored the current docs
**Purpose:** rewrite brief, not a polite review
**Verified against current public sources:** 2026-03-06

---

## How to use this brief

Treat this document as an instruction set for rewriting the current design pack.

Rules:

1. **Do not defend the current docs.** If this brief says a decision is wrong, weak, incomplete, or prematurely frozen, rewrite it.
2. **Preserve only what survives scrutiny.** There are good architectural instincts in the current pack. Keep them where they remain strong.
3. **Separate platform claims from product claims.** The current pack is mixing a universal runtime platform design with an LTX-first product plan and freezing the wrong things too early.
4. **Downgrade certainty where evidence is missing.** A provisional decision is better than a confident wrong one.
5. **Design for real Apple Silicon constraints, not generic local-serving abstractions.** Unified memory, media engines, shape variability, local packaging, and browser-to-localhost security are not side topics.
6. **Make the rewrite usable.** Update the existing docs, add the missing docs/ADRs, and provide concrete schemas and benchmark criteria.

---

## Reviewed inputs

This brief reviewed the current workspace docs as the primary artifact:

- `README.md`
- `01-product-requirements.md`
- `02-universal-mlx-runtime-design.md`
- `03-phased-delivery-plan.md`
- `01-upstream-baseline.md`
- `02-model-research-matrix.md`
- `03-language-and-runtime-choice.md`
- `04-serving-architecture-and-api.md`
- `05-optimization-playbook.md`
- `06-source-catalog.md`
- `07-ltx-integration-seams.md`
- `08-acceleration-techniques-survey.md`
- `09-open-questions-and-validation-plan.md`

This brief also checked current external evidence, prioritizing primary sources:

- official MLX docs and repositories
- Hugging Face Hub docs and the `Lightricks/LTX-2.3` model card
- official or direct project repositories for ComfyUI/LTX integrations
- Apple documentation and WWDC material relevant to unified memory, Metal, media engines, and video pipelines
- current MLX ecosystem packages where relevant
- current security guidance and concrete localhost/Origin validation evidence
- recent diffusion/video acceleration papers where the docs made forward-looking optimization claims

A source appendix is included at the end.

---

## Executive verdict

### Blunt overall assessment

The current pack is **directionally strong but materially under-specified and prematurely frozen**.

The strongest decisions survive:

- one canonical local runtime rather than per-host inference stacks
- a native job-oriented API instead of OpenAI-compatible text APIs as the source of truth
- Python-first model-family bring-up on top of MLX
- LTX as the first proving workload because it stress-tests the runtime on something genuinely difficult

The weakest decisions do not survive:

- the artifact/cache design conflates portable model artifacts with machine-local compile/build state
- the security model is too weak for a browser-reachable local daemon that accepts mutating requests and raw file paths
- the capability schema is too thin to drive honest host UX or safe runtime validation
- the scheduler section is mostly policy slogans rather than an actual resource model
- the docs overclaim universality while remaining mostly Hugging Face-shaped and LTX-shaped
- the language split undervalues Swift at the host boundary and fails to make “better MLX” an explicit track
- the package/install/service lifecycle story is barely present
- the benchmark plan is not strong enough to justify any “fixed” architecture claim

### What is genuinely impressive

- The pack does not trivialize LTX. Choosing `LTX-2.3` as the proving workload is ambitious in the right way: it forces text conditioning, image conditioning, video generation, optional audio, staged execution, upscalers, large memory behavior, and host integrations all at once.
- The pack correctly resists OpenAI-compatibility cargo culting. A native multimedia job API is the right center of gravity.
- The pack correctly recognizes that “universal” cannot mean “lowest common denominator”; model-family adapters are the right seam.

### What is still too assumption-heavy

- MLX compile/export behavior is treated as more stable and artifact-friendly than current evidence supports.
- Apple Silicon memory behavior is treated as a runtime detail rather than a primary architecture input.
- Hugging Face is treated as both the primary source of weights and almost the only source shape that matters, which quietly prevents real provider universality.
- ComfyUI strategy is based on an outdated value assumption.
- “Swift later” is an outdated simplification.
- “Architecture fixed in Phase 0” is not supportable.

### Bottom line

If implemented literally, the current docs will likely produce a **working LTX-shaped local daemon**.

If rewritten properly, they could produce a **real Apple Silicon local runtime platform**.

---

## Classification legend

Use these labels explicitly in the rewritten docs.

| Classification | Meaning |
| --- | --- |
| **Clearly wrong** | Contradicted by current evidence or structurally unsafe. Replace it. |
| **Likely wrong** | Strong evidence against it, but not mathematically impossible. Default to replacing it. |
| **Incomplete** | Directionally acceptable but missing essential design content. Rewrite substantially. |
| **Reasonable but under-validated** | Plausible, but should remain provisional until benchmarked. |
| **Strong** | Survives scrutiny and should be preserved. |

---

## What survives scrutiny vs what does not

### Strong

- **One runtime, many surfaces.**
- **Native job API as the source of truth.**
- **Python-first model-family bring-up.**
- **LTX first.**
- **Thin host adapters rather than host-owned inference.**
- **Family-aware scheduling instead of one-size-fits-all text-serving assumptions.**

### Reasonable but under-validated

- media mode vs batch mode as a top-level scheduler taxonomy
- one heavy generation job per runtime process as a v1 safety default
- HF-style MLX artifact packaging as a starting point
- family adapter contract and lifecycle
- selective quantization and stage-aware lifecycle control

### Incomplete

- capability schema
- provider abstraction
- source/provenance schema
- cache/GC/quota policy
- packaging/service lifecycle
- multi-process isolation strategy
- benchmark matrix and acceptance criteria
- host parity validation
- security model
- Apple-native media/output path
- explicit MLX extension/upstream roadmap

### Likely wrong

- compile/build state as part of the portable artifact contract
- a runtime-owned copied source cache as the primary source store
- “Swift later” as the default posture
- basic runtime-backed Comfy nodes as the main early Comfy value
- fixed API and architecture before cross-family validation

### Clearly wrong

- loopback HTTP + optional auth as the default security posture for mutating local routes that accept file paths
- “architecture decision-complete” in Phase 0

---

## Top critical findings

The table below should drive the rewrite order.

| Severity | Area | Finding | Why it matters | Exact fix / replacement | Confidence | Applies to |
| --- | --- | --- | --- | --- | --- | --- |
| **Critical** | Artifact/cache | The design conflates **portable converted artifacts** with **machine-local compile/build cache**. | MLX exported/imported functions are experimental and may not remain compatible across MLX versions. Custom Metal kernels also create/build per-kernel libraries. Compile outputs are not a stable portable artifact layer. | Split storage into **(1) source references**, **(2) portable converted artifacts**, and **(3) machine-local build cache**. Remove `compile/` from the portable artifact contract. | High | `02-universal-mlx-runtime-design.md`, `03-phased-delivery-plan.md` |
| **Critical** | Security/API | `127.0.0.1` + optional token is not an acceptable default security model for a browser-reachable mutating daemon, especially with raw filesystem paths in requests. | Loopback binding reduces remote exposure but does not stop browser-originated cross-origin mutation against localhost services. Raw paths raise the blast radius. | Default to **Unix domain socket** on macOS, or require **mandatory auth + Origin/Referer/Sec-Fetch validation** on all mutating HTTP routes. Remove raw file paths from the generic HTTP contract. | High | `02-universal-mlx-runtime-design.md`, `04-serving-architecture-and-api.md` |
| **High** | Capabilities | The capability descriptor is too thin to power host UX, validation, or truthful model behavior reporting. | LTX and similar families have hard input constraints, profile-dependent behavior, artifacts, scheduler needs, and licensing/access state that the current descriptor does not express. | Expand the schema with `constraints`, `profiles_by_task`, `artifacts_out`, `scheduler_class`, `hardware_tiers`, `dependencies`, `license`, `access_state`, `remote_code_required`, and `extensions_schema`. | High | `02-universal-mlx-runtime-design.md` |
| **High** | Scheduler | `media mode` vs `batch mode` is directionally right but not an actual scheduler design. | Apple Silicon unified memory and media workloads need stage reservations, cancellation points, live memory telemetry, and resource-aware admission. A simple estimate-based gate is not enough. | Replace with a **resource-graph scheduler**: per-stage reservations, warm-slot policy, live telemetry inputs, cooperative cancellation points, and explicit admission reasons. | High | `02-universal-mlx-runtime-design.md`, `04-serving-architecture-and-api.md` |
| **High** | Delivery plan | Phase 0 freezes the architecture and API before the key unknowns have been validated. | The validation doc itself admits major unknowns: LTX memory envelope, compile buckets, quantization tolerance, audio parity, source-to-artifact strategy, advanced acceleration survival. | Mark the API and architecture **provisional** until they survive LTX Fast + one image diffusion family + one non-generation family. Remove “decision-complete” / “fixed.” | High | `03-phased-delivery-plan.md`, `09-open-questions-and-validation-plan.md` |
| **High** | Sources/provenance | The source/cache design ignores current Hugging Face cache semantics, selective download, safetensors preflight metadata, gated/private repo flows, and `trust_remote_code` policy. | The runtime will either become wasteful, unsafe, or both if it treats provider content as copied files without provenance or policy. | Build a **preflight source resolver** and treat provider caches as immutable references where possible. Persist revision, license, access state, blob digests, and remote-code policy in manifests. | High | `01-product-requirements.md`, `02-universal-mlx-runtime-design.md`, `09-open-questions-and-validation-plan.md` |
| **High** | Universality/provider scope | The docs aim at broad family coverage but are **not provider-universal**. They are mostly Hugging Face-first plus local-file escape hatches. | “Universal” currently applies to families more than sources. This will limit future product scope and distort the cache/source model. | Add a **`SourceProviderAdapter`** layer parallel to `ModelFamilyAdapter` and explicitly scope supported providers. | High | `01-product-requirements.md`, `02-universal-mlx-runtime-design.md`, `03-phased-delivery-plan.md` |
| **High** | Language/runtime split | The docs frame the decision too much as Python vs Rust and underweight Swift and C++/Metal at the right boundaries. | Python should still own family bring-up, but Swift now matters earlier for embedding, launch/service UX, and Apple-native media/output integration. C++/Metal needs explicit hotspot seams. | Revise the language split: **Python workers**, **Swift host/SDK/launcher/embedded path**, **C++/Metal hotspot packages**, **Rust optional tooling only**. | Medium-high | `03-language-and-runtime-choice.md`, `02-universal-mlx-runtime-design.md` |
| **High** | Comfy strategy | The Comfy adapter plan is partly stale. | LTX support is now in ComfyUI core; custom packages increasingly add advanced controls rather than just baseline access. | Reposition the Comfy strategy around **advanced flows** or a generic **external execution bridge**, not just basic proxy nodes. | High | `07-ltx-integration-seams.md`, `03-phased-delivery-plan.md` |
| **High** | Ops/packaging | The docs barely cover installation, LaunchAgent/service lifecycle, crash recovery, disk quota/GC, codesigning/notarization, media codec dependencies, or sleep/wake behavior. | These issues determine whether the runtime is actually usable on macOS, especially as a shared local service. | Add a first-class **runtime packaging and operations** design section and validation plan. | High | `README.md`, `01-product-requirements.md`, `02-universal-mlx-runtime-design.md` |
| **Medium** | Optimization | The optimization docs miss Apple-specific output-path wins and do not separate DiT/video caching techniques cleanly enough. | For Apple Silicon video workloads, no-copy media output paths and the right DiT-specific cache techniques can matter more than generic serving ideas. | Add **zero-copy output pipeline** work, separate DiT/video methods from generic diffusion methods, and include negative evidence where reuse hurts quality. | Medium | `05-optimization-playbook.md`, `08-acceleration-techniques-survey.md` |
| **Medium** | Better MLX | The current pack treats MLX low-level work as a late escape hatch instead of an explicit “better MLX” track. | The project can create value not just by consuming MLX, but by extending it or upstreaming media-specific improvements. | Add an **MLX extension/upstream roadmap**: local extension package, targeted custom ops, upstream backlog, and benchmark track comparing stock MLX vs extension/fork paths. | High | `03-language-and-runtime-choice.md`, `05-optimization-playbook.md`, new ADR needed |

---

## Universality audit: model families, model types, and providers

### What the current docs are actually universal across

The current docs seriously attempt breadth across **generative and multimodal families**:

- image generation
- video generation
- audio generation
- speech recognition
- text-to-speech
- VLM / multimodal understanding

That is real breadth.

### What they are not universal across

They are **not yet “all model types”** in the broadest sense.

The design still centers on generative/media/VLM workloads. It gestures at text generation, embeddings, and VLM understanding, but it does not require a pure LLM or embedding family in the validation basket and does not prove that the lifecycle/capability/scheduling model works for those workloads.

You have two choices in the rewrite:

1. **Scope honestly.** Say this is a universal **local generative and multimodal runtime** for Apple Silicon, not all possible MLX workloads.
2. **Broaden the validation set.** Add one pure LLM or embedding family to the core architecture validation requirement.

Do not keep the current half-state where the docs imply full universality without proving it.

### Provider universality: current status

The current pack is **not provider-universal**.

It is essentially:

- Hugging Face first
- local-file bundles second
- everything else undefined

That is acceptable as an initial implementation focus, but not as an architecture truth if the word “universal” remains in the docs.

### Exact replacement

Add a provider abstraction that is explicit and first-class.

#### Proposed `SourceProviderAdapter`

```python
class SourceProviderAdapter(Protocol):
    provider_id: str

    def resolve(self, source_ref: SourceRef) -> ResolvedSource: ...
    def inspect(self, resolved: ResolvedSource) -> SourceInspection: ...
    def auth_requirements(self, source_ref: SourceRef) -> AuthRequirements: ...
    def fetch(self, resolved: ResolvedSource, policy: FetchPolicy) -> SourceMaterialization: ...
    def provenance(self, resolved: ResolvedSource) -> ProvenanceRecord: ...
```

#### Supported provider classes the architecture should model now

- **Hugging Face** (primary initial source)
- **GitHub releases/repos**
- **local filesystem bundles**
- **OCI or S3-style artifact registries**
- **future private/internal registries**

#### Proposed source reference schema

```json
{
  "provider": "huggingface",
  "locator": {
    "repo": "Lightricks/LTX-2.3",
    "revision": "pinned-commit-or-tag"
  },
  "materialization": {
    "mode": "provider-cache-ref"
  },
  "auth": {
    "token_ref": "hf-default"
  },
  "policy": {
    "allow_remote_code": false
  }
}
```

### Rewrite instruction

Do **not** say “universal” without clarifying whether you mean:

- universal across **families**
- universal across **providers**
- universal across **host surfaces**
- universal across **task classes**

Right now the docs only meaningfully support the first and third.

---

## Better MLX: direct runtime influence, C++, and the “better-than-stock-MLX” opportunity

The current docs acknowledge C++/Metal and custom kernels, but too passively. They treat them as late escape hatches instead of a **parallel strategic track**.

That is a missed opportunity.

This project can create value in four distinct layers:

1. **Use MLX effectively**
2. **Extend MLX locally**
3. **Contribute or maintain MLX improvements for media-specific pain points**
4. **Build Apple-native output plumbing around MLX that unlocks system-level performance**

The rewrite should say this explicitly.

### Path A: Project-local MLX extension layer

Create a reserved package such as:

```text
packages/
  runtime-kernels/
  runtime-mlx-ext/
```

or similar.

This package should hold only **profiled** hotspots, not speculative kernels.

Likely candidate areas:

- patchify / unpatchify
- latent pack / unpack
- VAE decode / tile decode
- rotary / attention helpers where stock fast ops are not enough
- spectrogram / audio preprocessing
- vocoder-critical kernels
- data layout transformations that create avoidable memory traffic

#### Why this matters

You already know that some families will eventually need more than pure Python MLX. Stop treating that as a late surprise. Reserve the boundary now.

### Path B: Upstream or forked MLX work for media-specific pain

This is the “better MLX” path in the strongest sense.

The rewrite should explicitly call out an **MLX improvement backlog** relevant to media and multimodal workloads:

- better shape-polymorphic compile behavior for variable-resolution / variable-frame-count media graphs
- improved compile-cache / kernel-cache behavior for repeated shape buckets
- video-friendly `fast.*` or fused ops where stock operator composition leaves performance on the table
- better hooks for live memory telemetry and admission-aware execution
- stronger extension ergonomics for app-embedded paths

#### Why this matters

The current docs already identify compile bucketing, shape variability, and custom-kernel escalation as important. The missing step is to state clearly that some of those problems may be solved more effectively by improving the substrate rather than only wrapping it.

### Path C: Apple-native zero-copy output path

For video and some audio/media workflows, the biggest win may not be inside the denoise loop alone.

The architecture should explicitly include a performance track for:

- `MLX / Metal output`
- `IOSurface or CVPixelBuffer-backed surfaces`
- `VideoToolbox / media engine encode path`
- minimal or zero extra frame copies

#### Why this matters

On Apple Silicon, GPU, CPU, and media engines share unified memory. The right output path can remove an entire class of avoidable copies and synchronization overhead.

### Path D: Dedicated worker/process boundaries for low-level performance modules

Even if Python remains the right place for family bring-up, you should consider:

- dedicated worker processes for heavy media execution
- extension modules loaded into those workers
- a stable ABI boundary for high-performance kernels or native helpers

This is not “rewrite the platform in C++.” It is a better way to integrate native performance code into a Python-first runtime.

### Rewrite instruction

Add a dedicated section or ADR called something like:

- `MLX Extension And Upstream Roadmap`
- `Runtime Native Hotspot Strategy`
- `Apple-Native Media Output Optimization`

Do not bury this under a generic “escalation path” bullet list.

---

## Assumption audit

Every major assumption below should either be preserved with evidence, downgraded to provisional, or replaced.

### MLX capabilities and compilation

| Assumption | Status | Assessment |
| --- | --- | --- |
| Python should own v1 family bring-up. | **Validated** | Keep it. The real MLX ecosystem for family bring-up is still Python-first. |
| `mx.compile` is a broad reliable accelerator for media pipelines. | **Weakly supported** | Compile helps repeated stable inner loops, but shape changes trigger recompilation unless shapeless mode is used, and shapeless mode has correctness caveats. |
| Compiled/exported functions can be part of a portable artifact story. | **Probably false** | MLX export/import is explicitly experimental and version compatibility is not guaranteed. |
| Compiled functions can safely wrap orchestration with side effects. | **Probably false** | MLX compile expects purity; side effects or inspection can break or crash. |
| Streams are a meaningful primitive for runtime design. | **Validated** | Keep them, but move from vague mention to explicit execution strategy. |
| Selective quantization by submodule is feasible in MLX. | **Validated** | Strong and underused in the current docs. |
| Custom Metal kernels can wait entirely until late. | **Weakly supported** | Do not implement them blindly, but reserve the seam now. |

### Memory model and hardware tiers

| Assumption | Status | Assessment |
| --- | --- | --- |
| `64GB+` is the honest “works well” tier for LTX-class workloads. | **Weakly supported** | Plausible, but still needs measured proof by workload/profile. |
| `32GB` is a viable degraded tier for LTX Fast. | **Unsupported** | Possible for narrow profiles, but not established. |
| Family estimates plus generic thresholds are enough for admission control. | **Weakly supported** | Static estimates alone are not enough; use live memory telemetry too. |
| Unified memory is mostly a scheduling detail. | **Probably false** | It is a primary architecture input affecting model execution, output paths, and copy strategy. |

### Model conversion, artifact format, and provenance

| Assumption | Status | Assessment |
| --- | --- | --- |
| `source-native` and `artifact-native` should both exist. | **Validated** | Keep this. |
| The runtime should own a copied source cache under its own directory tree. | **Probably false** | Provider caches, especially Hugging Face, already model immutable cached sources well. Favor references and manifests. |
| HF-style package layout is enough as a universal artifact format. | **Weakly supported** | Good starting point, but incomplete without provenance, license, access state, portability class, and dependencies. |
| Compile profile belongs in the same portability class as weights/processors. | **Probably false** | Machine-local build state must be separated from portable artifacts. |
| Local conversion from provider weights should be first-class. | **Validated** | Keep it, but benchmark it against preconverted and policy-aware flows. |
| “Open weights” is sufficient source-policy language. | **Weakly supported** | Replace with explicit license and access-state modeling. |

### Daemon/API design

| Assumption | Status | Assessment |
| --- | --- | --- |
| One long-lived local daemon should be the canonical external surface. | **Validated** | Keep it as the default external mode. |
| SSE is enough for v1 event delivery. | **Validated** | Keep it. |
| Compatibility APIs should be adapters, not the core contract. | **Validated** | Keep this; it is one of the strongest decisions. |
| The current capability schema is sufficient for hosts. | **Unsupported** | Rewrite it. |
| `127.0.0.1` + optional token is good enough for v1. | **Probably false** | Rewrite it. |
| Raw absolute file paths are acceptable in the canonical HTTP job contract. | **Probably false** | Replace with handles or controlled import/export flows. |

### Batching and scheduling

| Assumption | Status | Assessment |
| --- | --- | --- |
| Two scheduler modes (`media`, `batch`) are enough top-level taxonomy. | **Reasonable but under-validated** | Keep the taxonomy, but implement an actual scheduler model under it. |
| One heavy generation job per process is the right default. | **Validated** | Keep as a v1 safety default, not as the full scheduler design. |
| Prefix/prompt caching belongs to text/VLM-oriented families, not media generation by default. | **Validated** | Correct. |
| Text-serving patterns should not dominate video runtime design. | **Validated** | Strong. Keep. |

### LTX feasibility and staging

| Assumption | Status | Assessment |
| --- | --- | --- |
| LTX Fast is the correct first proving slice. | **Validated** | Keep it. |
| The architecture is not secretly LTX-shaped. | **Weakly supported** | Intention is broad; current docs still read too LTX-shaped in key places. |
| Audio can wait without architecture damage. | **Weakly supported** | Only if audio dependencies, constraints, and output semantics are modeled now. |
| A second open video family can wait until late. | **Weakly supported** | Product-wise maybe; architecture-wise it should appear earlier in validation. |

### Comfy/Desktop assumptions

| Assumption | Status | Assessment |
| --- | --- | --- |
| Desktop should become a thin runtime client. | **Validated** | Keep it. |
| Comfy should get a dedicated runtime-backed package. | **Weakly supported** | Maybe, but the value surface should be updated for current Comfy reality. |
| Basic fast T2V/I2V proxy nodes are the best early Comfy surface. | **Weakly supported** | Likely stale. Advanced controls may be the real value surface now. |
| All first-party hosts should only talk to the daemon, not embed the core. | **Weakly supported** | External hosts: yes. First-party Apple apps may need an embedded/shared-core option. |

### Python vs Rust vs Swift vs C++/Metal

| Assumption | Status | Assessment |
| --- | --- | --- |
| Rust should not own the v1 inference core. | **Validated** | Keep it. |
| Python should own family adapters and conversion. | **Validated** | Keep it. |
| Swift can wait until later. | **Probably false** | Rewrite the language split. Swift matters earlier at the host boundary. |
| C++/Metal should be purely reactive and only discussed after profiling. | **Weakly supported** | Profiling should drive implementation, but the architecture should reserve the seam now. |

---

## Missing research and missing design work

The current pack is missing or under-covers several topics that materially affect the architecture.

### 1. MLX compile portability, purity, and shape behavior

The current docs mention compile profiles but do not absorb the actual consequences of current MLX behavior:

- shape changes can trigger recompilation
- shapeless compilation has correctness caveats for shape-dependent graphs
- compiled functions are intended to be pure
- export/import is explicitly experimental and not guaranteed stable across versions

This should have changed the artifact and scheduler design immediately.

### 2. Machine-local kernel build/JIT behavior

The current pack talks about custom Metal kernels as if they are just “later if needed.”
What is missing is the operational implication:

- kernel creation/build/JIT has a cold-path cost
- library reuse and kernel lifetime matter
- build outputs belong in a machine-local cache, not a portable artifact story

### 3. Apple-native media output pipeline

The docs barely discuss:

- `IOSurface`
- `CVPixelBufferPool`
- `CVMetalTextureCache`
- `VideoToolbox`
- GPU-to-media-engine handoff on unified memory

For LTX/video, this is not optional background reading. It is one of the highest-value Apple-specific optimization paths.

### 4. Source-provider and provenance model

The docs should have covered:

- shared provider cache semantics
- selective download
- blob identity and digests
- revision pinning
- gated/private repo flows
- fine-grained tokens
- safetensors metadata preflight
- remote-code trust boundaries

Instead, the current design collapses most of that into “Hugging Face pull + source cache.”

### 5. Localhost threat model

The docs do not treat local HTTP as a real security surface.

That is a serious miss because the proposed daemon:

- is browser-reachable if loopback HTTP is enabled
- has mutating endpoints
- accepts raw path values
- may eventually front multiple hosts

At minimum, the rewrite must cover:

- UDS vs HTTP
- Origin/Referer/Sec-Fetch validation
- auth/token storage
- file-handle model
- browser restrictions
- multi-user and shared-machine behavior

### 6. Worker-process topology

The current docs implicitly push too much into one daemon process.
There is almost no analysis of a control-plane daemon plus per-family or per-job workers, even though that design obviously helps with:

- crash isolation
- memory reclamation
- dependency isolation
- remote-code containment
- native extension lifecycle
- per-family profiling

### 7. Packaging and macOS lifecycle

The docs are missing research on:

- LaunchAgent vs embedded service vs on-demand service
- notarization/codesigning
- app sandbox implications
- crash restart policy
- upgrade migration
- disk quota/GC
- dependency policy for codecs or media tooling
- sleep/wake behavior

### 8. Current Comfy/LTX state

The current Comfy strategy needs to be refreshed against current reality:

- LTX support is in Comfy core
- extra packages increasingly add advanced nodes/workflows
- a “basic runtime-backed node package” may not be the most valuable first move anymore

### 9. DiT/video-specific acceleration research taxonomy

The optimization survey lumps too many things together.

It needs a cleaner separation between:

- production-ready baseline tactics
- plausible MLX-compatible DiT/video methods
- speculative or not-yet-production-ready research
- methods with known quality trade-offs or negative evidence

### 10. Product scope honesty

The current docs should decide whether the project is:

- a universal Apple Silicon **generative and multimodal runtime**, or
- a universal runtime for **all local MLX model classes**

Right now it gestures at both and proves neither completely.

---

## Alternative architectures that should have been explicitly considered

### Alternative 1: Control-plane daemon + per-family Python workers

**Description**
A thin daemon owns model registry, source/artifact indexes, auth, scheduling, and job state. Actual execution happens in family-specific Python workers.

**Better**
- crash isolation
- better memory reclamation
- easier dependency isolation
- cleaner remote-code boundary
- better profiling per family
- easier future sandboxing

**Worse**
- more IPC and orchestration complexity
- slightly higher cold-start overhead

**Recommendation**
**Adopt partially.**
Make this the preferred execution topology, even if a simple in-process mode exists for early bring-up.

---

### Alternative 2: Shared core library + optional daemon wrapper

**Description**
There is one shared runtime core. External tools use the daemon. First-party Apple apps may embed the core directly or through a shared framework.

**Better**
- better native-app integration
- avoids localhost security footguns for first-party apps
- lower latency for some host flows
- better fit for Swift and Apple-native media/output integration

**Worse**
- risk of policy drift if embedded callers bypass daemon-enforced lifecycle rules

**Recommendation**
**Partially adopt.**
Keep the daemon as the canonical external surface, but do not design the core so tightly around “daemon only” that first-party embedding becomes awkward.

---

### Alternative 3: Content-addressable internal store + user-facing HF-like layout

**Description**
Internally store blobs and manifests content-addressably. Externally support familiar HF-style portable artifact folders where useful.

**Better**
- deduplication
- stronger provenance
- safer GC
- better partial reuse
- cleaner future multi-provider support

**Worse**
- more implementation complexity
- less immediately familiar to users than a plain folder story

**Recommendation**
**Adopt internally.**
Do not force users to see the complexity, but stop pretending a simple folder tree is the whole storage model.

---

### Alternative 4: Explicit platform track and product track

**Description**
Separate the universal runtime/platform work from the LTX product integration work.

**Better**
- prevents LTX urgency from freezing wrong platform decisions
- makes benchmarking and validation more honest
- makes “what is universal” vs “what is LTX-specific” explicit

**Worse**
- coordination overhead

**Recommendation**
**Adopt.**
This is the cleanest correction to the current pack.

---

## Language/runtime challenge

The language decision in the current docs is partly right and partly framed too narrowly.

### What is still correct

Python should still own:

- family adapters
- conversion logic
- processor/tokenizer glue
- initial execution bring-up
- semantic parity work
- most runtime workers in v1

That is where the real ecosystem still is.

### What is wrong in the current framing

The current docs act too much as if the decision is mainly:

- Python vs Rust

That is not the real choice.

The real high-value split is:

- **Python** for model semantics and workers
- **Swift** for first-party host integration and Apple-native embedding
- **C++/Metal** for targeted hotspots and MLX extension work
- **Rust** as optional systems tooling, not the core inference runtime

### Revised language split

| Area | Recommended owner | Why |
| --- | --- | --- |
| Model-family adapters | **Python** | fastest bring-up, best ecosystem alignment |
| Conversion + inspection | **Python** | aligns with provider/model tooling |
| Runtime workers | **Python first** | simplest way to leverage MLX today |
| Host SDK / launcher / service wrapper | **Swift** | strong Apple-native fit |
| Embedded first-party access mode | **Swift + shared core** | avoids daemon-only rigidity |
| Native hotspots / custom ops | **C++/Metal** | best place for targeted performance work |
| Systems tooling / optional supervisor | **Rust (optional)** | acceptable for non-core tooling only |

### Rewrite instruction

Replace the current `Why Swift Should Not Own The Core Runtime` framing with:

- **Swift should not own v1 family bring-up**
- **Swift should matter earlier at the host boundary**
- **C++/Metal should be a reserved hotspot and MLX-extension seam**
- **Rust should stay optional and non-core**

Also add a dedicated subsection on **“better MLX”** as part of the language/runtime doc, not just as a generic low-level escape hatch.

---

## Performance and optimization review

### Baseline-good ideas that should stay

- stage-aware lifecycle control
- selective quantization
- compile only stable repeated inner loops
- family-specific scheduling policy
- artifact reuse in principle
- stage splitting for LTX-style pipelines

These are good instincts.

### Ideas that are currently hype or over-framed

- **Compile profiles as if they are durable portable artifact content**
- **Generic diffusion caching as one bucket**
- **Streams as a generic win without a concrete overlap plan**
- **An optimization ladder that sounds cleaner than the current evidence actually supports**

### Important missing acceleration methods or missing framing

The rewrite should explicitly cover:

#### Apple-native system path
- zero-copy output path
- `IOSurface` / `CVPixelBuffer` / `VideoToolbox`
- CPU/GPU/media-engine interaction on unified memory

#### DiT/video-specific methods
- AdaCache
- FasterCache
- PAB
- HiCache
- ProCache
- negative evidence where naive adjacent-step reuse harms quality

#### Native hotspot candidates
- VAE decode / tile decode
- patchify / unpatchify
- attention / rotary helpers
- audio preprocessing / spectrogram paths
- vocoder-critical code paths

### What is not yet production-ready enough to bake into architecture

- experimental paged-attention assumptions on Apple/Metal paths
- very recent DiT/video caching methods without workload-specific validation
- broad claims that a given paper’s speedup will survive an MLX implementation unchanged

### Best breakthrough candidates on Apple Silicon

#### For LTX/video
1. stage-aware scheduler with honest reservations
2. zero-copy output path to media engines
3. shape-bucketed compiled inner loops
4. validated DiT/video temporal/cache reuse on real LTX workloads
5. targeted native kernels only after profiling

#### For speech/audio
1. chunk-aware scheduler
2. reusable conditioning state
3. fused preprocessing kernels
4. output-path separation from containerization/encoding

#### For text/VLM
1. prefix/prompt caching
2. continuous batching where appropriate
3. KV/cache quantization where appropriate
4. dynamic load/unload with stronger cache ownership semantics

### Rewrite instruction

The optimization docs should stop pretending that one ladder fits all families.
Rewrite them into:

- **production baseline**
- **family-specific defaults**
- **MLX-compatible research bets**
- **native hotspot candidates**
- **Apple-specific output and memory-path optimizations**
- **negative evidence / “do not assume” notes**

---

## Validation and benchmarking plan critique

The current validation doc is better than the rest of the pack in one important way: it admits uncertainty.

But the benchmark plan is still too weak.

### What it misses

You need to benchmark at least the following separately:

- cold source pull
- cold preflight inspect
- cold convert
- cold load
- warm load
- warm run
- model switch latency
- compile count / compile cache hit rate
- job cancellation latency
- crash recovery
- mixed-workload interference
- output encode/write time separated from model time
- cache reuse hit rate
- disk growth and GC behavior
- license/gating/remote-code policy paths
- host parity between CLI / REST / Desktop / Comfy

### Required benchmark matrix

| Axis | Minimum coverage |
| --- | --- |
| Families | LTX Fast T2V, LTX Fast I2V, one image diffusion family, Whisper or equivalent, one TTS/audio family, one VLM, and either one pure LLM or a justified explicit exclusion |
| Hardware | at least one 32GB, one 64GB, one 128GB Apple Silicon tier; include an older Max-class machine and a newer Max-class machine if available |
| Run state | cold pull, cold convert, cold load, warm run, model switch, repeated run |
| Concurrency | isolated heavy job, heavy + light, multiple light jobs, cancellation under load |
| Output path | model time only, model + decode, model + encode/mux/write |
| Precision | baseline float path, q8, q6/q4 where meaningful, selective submodule quantization where relevant |
| Shapes | common bucketed shapes plus adversarial off-bucket shapes |

### Acceptance criteria that are too weak

The current acceptance criteria are mostly demo criteria.

Examples that are too weak:

- “can run end to end”
- “same model through CLI and REST”
- “one non-LTX family works”
- “capability metadata is sufficient”

Replace them with criteria that actually prove the platform.

### Mandatory measurements before claims can be trusted

- time to first artifact
- total wall time
- per-stage timings
- active / peak / cached memory
- live device memory telemetry
- compile count and compile cache hit rate
- CPU/GPU/media-engine utilization where measurable
- artifact/source cache hit rate
- quality deltas under quantization or acceleration
- cancellation latency
- failure reason taxonomy

### Recommended acceptance gates for freezing the API

Do **not** declare the API stable until the same lifecycle/capability/job model survives:

1. LTX Fast
2. one image diffusion family
3. one non-generation family (for example Whisper or a VLM understanding family)
4. one provider source flow beyond “HF repo or local path” **or** an explicit provider-scope limitation in the docs

---

## Security, packaging, cache, artifact, and host-integration gaps

This needs to become a first-class section in the rewritten pack.

### Security gaps

- local HTTP threat model is too weak
- no serious treatment of Origin/Referer/Sec-Fetch validation
- no policy for token storage or session rotation
- no real handling of `trust_remote_code`
- no file-handle model to replace raw path injection
- no multi-user or shared-machine story
- no daemon/worker privilege separation story

### Packaging and service gaps

- no LaunchAgent / on-demand service plan
- no upgrade/migration policy
- no codesign/notarization plan
- no crash restart policy
- no sleep/wake handling
- no codec/dependency policy for media output

### Cache and artifact gaps

- no quota/GC policy
- no dedup/content-addressable thinking
- no portability-class separation between sources, artifacts, and build cache
- no robust provenance or license state propagation
- no integrity-verification story beyond “download and convert”

### Host integration gaps

- desktop vs daemon vs embedded mode not clarified
- Comfy strategy not refreshed
- no SDK shape for future host tools
- no honest discussion of host-specific artifact/file-handoff semantics

### Rewrite instruction

Add a dedicated operations/security design doc or expand the current runtime + serving docs to include all of the above.

---

## Concrete file-by-file rewrite instructions

This section is intentionally prescriptive.

### `README.md`

**What to change**
- Stop presenting the architecture as effectively settled.
- Split the docs into **platform** vs **product/LTX** tracks.
- Add a “known unstable decisions” section.
- Add a “last verified” / freshness note.

**Suggested new structure**
1. Workspace purpose
2. Platform track docs
3. Product/LTX track docs
4. Known unstable decisions
5. Current external assumptions verified on 2026-03-06
6. Rewrite targets / ADRs

---

### `01-product-requirements.md`

**What to change**
- Replace loose “open weights” language with explicit license/access-state language.
- State whether the project is universal across all MLX model types or specifically across local generative and multimodal families.
- Add a provider-scope statement.
- Add runtime packaging/service UX requirements.
- Mark hardware tiers as provisional until benchmarked.

**Suggested replacement direction**
- Vision should say: “Build the best local-first generative and multimodal runtime for Apple Silicon” unless you are willing to validate broader scope.
- Add a non-negotiable around provenance, licensing, and remote-code policy.
- Add runtime install and service UX requirements.

**Suggested replacement text for source policy**
> Weights and processors come from hub-hosted or local sources under explicit license and access state. The runtime must preserve source provenance, revision, access state, and remote-code policy. Converted artifacts are not assumed to be redistributable.

---

### `02-universal-mlx-runtime-design.md`

**What to change**
- Rewrite the source/artifact/build-cache model.
- Add a provider abstraction.
- Replace raw-path HTTP job payloads with handle-based or imported asset flows.
- Replace the minimal capability descriptor with a real schema.
- Rewrite scheduling into a resource-graph model.
- Replace the local security section with a real threat model.
- Add worker-process topology.
- Add an explicit MLX extension / native-hotspot seam.
- Clarify embedded mode vs daemon mode.

**Suggested new sections**
1. Source provider model
2. Portability classes: source refs vs portable artifacts vs build cache
3. Worker/process topology
4. Expanded capability schema
5. File and artifact handle model
6. Transport and security model
7. Scheduler resource model
8. Native hotspot / MLX extension seam
9. Embedded vs daemon access mode

**Suggested replacement storage layout**
```text
$MLX_RUNTIME_HOME/
  config/
  logs/
  jobs/
  temp/
  sources-ref/
    huggingface/
    github/
    local/
    oci/
  artifacts-portable/
    <family>/<model>/<artifact_digest>/
  build-cache-local/
    <mlx_version>/<macos_build>/<chip_family>/<adapter_version>/<shape_bucket>/<precision>/
```

---

### `03-phased-delivery-plan.md`

**What to change**
- Remove “decision-complete” / “API fixed” from Phase 0.
- Split the plan into **platform track** and **product track**.
- Insert a validation checkpoint before API freeze.
- Move provider abstraction, security, and worker-process topology earlier.
- Add an explicit MLX-extension/upstream exploration track.

**Suggested new phase logic**
- Phase A: platform skeleton + security + source/provider + artifact portability classes
- Phase B: LTX Fast proof path
- Phase C: second family + non-generation family validation
- Phase D: Desktop/Comfy integrations
- Phase E: native hotspot / MLX extension passes
- API freeze only after Phase C or D, not Phase 0

---

### `01-upstream-baseline.md`

**What to change**
- Refresh current ecosystem state:
  - current LTX-2.3 model card
  - current Comfy/LTX state
  - MLX Swift / MLX C / MLX Swift LM maturity
  - active MLX VLM / audio packages
- Add explicit notes about stale or uncertain ecosystem observations.

**Suggested new subsections**
- LTX 2.3 current model reality
- Comfy core vs extra nodes
- MLX language surfaces
- MLX extension ergonomics
- Serving stack freshness and caveats

---

### `02-model-research-matrix.md`

**What to change**
- Add an explicit “provider/source pressure” column.
- Either add a pure LLM or embedding family to the validation matrix or explicitly scope it out.
- Add a “scheduler class” and “artifact/output class” column.
- Add “host integration pressure” and “native hotspot likelihood.”

**Suggested matrix columns**
- family
- workload class
- provider/source pressure
- scheduler class
- capability pressure
- artifact/output pressure
- memory pressure
- likely native hotspots
- validation priority

---

### `03-language-and-runtime-choice.md`

**What to change**
- Stop treating Swift as mostly future.
- Add the “better MLX” section.
- Reframe the decision away from Python vs Rust.
- Define the revised language ownership table.

**Suggested replacement structure**
1. What Python must own
2. Why Rust should remain non-core
3. Why Swift matters earlier than previously stated
4. Why C++/Metal must be reserved now
5. Better MLX: local extension path and upstream backlog
6. Revised language split

---

### `04-serving-architecture-and-api.md`

**What to change**
- Replace the auth section completely.
- Add UDS vs loopback HTTP transport policy.
- Add a handle-based file model.
- Add daemon/worker topology.
- Expand capability semantics.
- Add policy propagation for license/access/remote-code state.

**Suggested new sections**
- threat model
- transport modes
- auth and browser-origin protections
- file handle and artifact handle model
- worker execution model
- policy propagation and provenance
- SDK/facade policy

---

### `05-optimization-playbook.md`

**What to change**
- Add Apple-native output-path optimization.
- Separate production-ready methods from speculative DiT/video methods more aggressively.
- Add likely native hotspot candidates explicitly.
- Add negative evidence where methods can hurt quality.
- Add MLX extension benchmark criteria.

**Suggested new structure**
1. baseline platform optimizations
2. Apple-specific memory/output optimizations
3. family-specific defaults
4. MLX-compatible research bets
5. native hotspot candidate shortlist
6. validation rules

---

### `06-source-catalog.md`

**What to change**
- Add freshness/status metadata.
- Distinguish official vs community vs speculative sources.
- Add security/policy sources and Apple/media-system sources.
- Add a “what this source proves” note for each important entry.

**Suggested extra columns**
- source type
- last verified
- freshness risk
- what it proves
- action relevance

---

### `07-ltx-integration-seams.md`

**What to change**
- Refresh Comfy assumptions.
- Add audio/output-path seams, not just baseline fast-path seams.
- Add constraint propagation requirements.
- Add notes on where runtime-managed handles should replace direct path assumptions.
- Add an “LTX-specific vs universal seam” distinction.

**Suggested new sections**
- constraint propagation
- audio branch readiness
- output and encode seams
- advanced-flow Comfy opportunities
- universalizable seams vs LTX-only seams

---

### `08-acceleration-techniques-survey.md`

**What to change**
- Reclassify techniques by readiness and family fit.
- Add Apple-specific output-path work.
- Add AdaCache / FasterCache / PAB / HiCache / ProCache explicitly.
- Add negative-evidence notes.
- Mark which methods are plausible for MLX now vs later.

**Suggested new tiers**
- Tier A: production baseline
- Tier B: family-specific proven candidates
- Tier C: plausible MLX-compatible research bets
- Tier D: speculative / not production-ready

---

### `09-open-questions-and-validation-plan.md`

**What to change**
- Add open questions for:
  - provider abstraction
  - artifact portability classes
  - security model
  - worker-process topology
  - Apple-native output path
  - packaging/ops
  - MLX extension/upstream roadmap
- Upgrade the benchmark plan into a real matrix with metrics and acceptance gates.
- Add “freeze criteria” for when architecture/API claims can become stable.

**Suggested new open questions**
1. Which provider abstraction is enough for v1?
2. What exactly is portable vs machine-local in artifacts/build state?
3. What daemon/worker topology gives the best correctness/performance trade-off?
4. Which Apple-native output path should be canonical for video?
5. Which MLX extension or upstream changes are worth carrying?

---

## New docs / ADRs that should be added

The current doc set is missing several architecture records that should exist before implementation starts.

### Required new ADRs

1. **`docs/adr/0001-source-provider-adapter.md`**
   Why provider abstraction exists, what v1 supports, and how provenance/auth/license are modeled.

2. **`docs/adr/0002-artifact-vs-build-cache.md`**
   Defines portability classes and explicitly states that build cache / compile cache are machine-local.

3. **`docs/adr/0003-local-security-model.md`**
   Defines transport modes, UDS vs HTTP, auth requirements, browser-origin validation, file-handle model, and `trust_remote_code` policy.

4. **`docs/adr/0004-worker-process-topology.md`**
   Defines control-plane daemon vs execution workers and failure/isolation behavior.

5. **`docs/adr/0005-mlx-extension-and-upstream-roadmap.md`**
   Defines when to use pure MLX, `mx.compile`, `fast.metal_kernel`, custom MLX extensions, or upstream MLX work.

### Recommended new non-ADR docs

- `docs/capability-schema.md`
- `docs/benchmark-matrix.md`
- `docs/operations-and-packaging.md`
- `docs/provider-and-provenance-model.md`

---

## Concrete replacement schemas and structures

These are not the only possible shapes, but the rewritten docs need something at least this concrete.

### Improved capability descriptor

```json
{
  "model_id": "ltx-2.3-fast-local",
  "family": "ltx",
  "family_variant": "fast",
  "tasks": ["video.generate", "video.condition.image"],
  "modalities_in": ["text", "image"],
  "modalities_out": ["video", "audio"],
  "constraints": {
    "width": { "multiple_of": 32 },
    "height": { "multiple_of": 32 },
    "num_frames": { "formula": "8n+1" }
  },
  "conditioning": {
    "image": true,
    "video": false,
    "audio": false,
    "lora": true
  },
  "profiles_by_task": {
    "video.generate": ["fp16", "q8"],
    "video.condition.image": ["fp16", "q8"]
  },
  "streaming": {
    "progress_events": true,
    "partial_artifacts": true,
    "token_deltas": false,
    "segment_events": true
  },
  "artifacts_out": ["mp4", "mov", "wav"],
  "scheduler_class": "media_video_dit",
  "hardware_tiers": [
    {
      "tier": "recommended",
      "memory_gb": 64,
      "notes": "full fast profile target"
    },
    {
      "tier": "degraded",
      "memory_gb": 32,
      "notes": "reduced profiles only; benchmark-gated"
    }
  ],
  "dependencies": {
    "media_encode": {
      "required": true,
      "policy": "runtime-managed"
    }
  },
  "policy": {
    "license": "declared-in-source-manifest",
    "access_state": "declared-in-source-manifest",
    "remote_code_required": false
  },
  "extensions_schema": {
    "namespace": "ltx",
    "version": "1"
  }
}
```

### Improved source / artifact / build-cache separation

```text
$MLX_RUNTIME_HOME/
  config/
  logs/
  jobs/
  temp/
  sources-ref/
    huggingface/
      <provider-source-manifest>.json
    github/
      <provider-source-manifest>.json
    local/
      <provider-source-manifest>.json
    oci/
      <provider-source-manifest>.json
  artifacts-portable/
    <family>/<model_id>/<artifact_digest>/
      manifest.json
      weights/
      processors/
      metadata/
  build-cache-local/
    <mlx_version>/<macos_build>/<chip_family>/<adapter_version>/<shape_bucket>/<precision>/
```

### Improved job/input model

For generic HTTP or SDK clients, do **not** accept raw absolute input paths by default.

Use handles or explicit import flows.

```json
{
  "model_id": "ltx-2.3-fast-local",
  "task": "video.generate",
  "inputs": {
    "prompt": "cinematic slow dolly shot of a fox in snow",
    "images": [
      {
        "input_handle": "inp_01H...",
        "frame_index": 0,
        "strength": 1.0
      }
    ]
  },
  "params": {
    "width": 768,
    "height": 512,
    "num_frames": 97,
    "fps": 24,
    "seed": 42
  },
  "output": {
    "artifact_format": "mp4",
    "destination": {
      "mode": "runtime_managed"
    }
  },
  "extensions": {
    "ltx": {}
  }
}
```

For trusted local CLI or embedded callers, a direct-path mode may exist as a **separate trusted surface**, not the universal default HTTP contract.

### Preferred runtime topology

```text
Hosts (CLI / REST / Desktop / Comfy / SDK)
        |
        v
Control-plane daemon
  - auth / transport
  - registry
  - provider resolution
  - artifact index
  - scheduler
  - job store
  - capability service
        |
        +--> Family worker: LTX (Python + MLX + optional native ext)
        |
        +--> Family worker: image diffusion
        |
        +--> Family worker: Whisper/VLM/etc.
```

---

## Final recommendation

### Recommendation label

**Split into separate platform and product tracks, and revise the architecture significantly before implementation claims are frozen.**

This is stronger than “minor revision” but not a full pivot away from the current direction.

### What to keep

- one canonical runtime
- native job API
- Python-first family bring-up
- LTX as first proving workload
- thin adapters

### What to change now

- provider abstraction
- artifact vs build-cache split
- security model
- capability schema
- scheduler design
- worker-process topology
- packaging/ops design
- Swift role at the host boundary
- explicit MLX extension/upstream track
- benchmark/acceptance rigor

### The clearest next 5 actions

1. **Rewrite `02-universal-mlx-runtime-design.md` and `04-serving-architecture-and-api.md` first.**
   Those two docs currently contain the most consequential wrong or incomplete decisions.

2. **Split the plan into platform and product tracks.**
   Do not let LTX product urgency freeze the universal runtime design prematurely.

3. **Add the missing ADRs before code is written.**
   Provider abstraction, security model, artifact/build-cache split, worker topology, MLX extension roadmap.

4. **Refresh the upstream/research docs against current reality.**
   Especially LTX 2.3, Comfy core state, MLX Swift/C, and current MLX ecosystem packages.

5. **Replace the current benchmark plan with a real matrix and freeze criteria.**
   The architecture is not “fixed” until it survives that matrix.

---

## Exact rewrite order

If you only have enough time to rewrite a subset of the pack first, do it in this order:

1. `02-universal-mlx-runtime-design.md`
2. `04-serving-architecture-and-api.md`
3. `03-phased-delivery-plan.md`
4. `03-language-and-runtime-choice.md`
5. `09-open-questions-and-validation-plan.md`
6. `01-product-requirements.md`
7. `07-ltx-integration-seams.md`
8. `05-optimization-playbook.md`
9. `08-acceleration-techniques-survey.md`
10. `01-upstream-baseline.md`
11. `02-model-research-matrix.md`
12. `06-source-catalog.md`
13. `README.md`

---

## Non-negotiables for the rewrite

Do not ship a rewritten pack that still does any of the following:

- declares the architecture/API fixed before cross-family validation
- treats compile/build state as portable artifact content
- uses optional auth as the default security posture for mutating local HTTP routes
- leaves raw absolute file paths in the universal HTTP contract
- claims provider universality without a provider abstraction
- treats Swift as mostly future-only
- lacks a packaging/service-lifecycle design
- lacks an explicit MLX extension/upstream track
- keeps demo-level acceptance criteria

---

## Appendix A: external evidence to use in the rewrite

These are not exhaustive; they are the highest-value checked sources.

### MLX core / official

- MLX repo (official overview, language APIs):
  https://github.com/ml-explore/mlx

- MLX compile docs (shape changes, shapeless mode, purity caveats):
  https://ml-explore.github.io/mlx/build/html/usage/compile.html

- MLX export/import docs (explicitly experimental compatibility):
  https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.exporter.html

- MLX custom extensions / primitives:
  https://ml-explore.github.io/mlx/build/html/dev/extensions.html

- MLX custom Metal kernels:
  https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html

- MLX streams and unified memory model:
  https://ml-explore.github.io/mlx/build/html/usage/using_streams.html

- MLX Metal memory APIs:
  https://ml-explore.github.io/mlx/build/html/python/metal.html

### MLX language surfaces

- MLX C repo:
  https://github.com/ml-explore/mlx-c

- MLX Swift examples / Apple-native examples:
  https://github.com/ml-explore/mlx-swift-examples

- MLX Swift LM:
  https://github.com/ml-explore/mlx-swift-examples/tree/main/Applications/MLXLMExample
  and current releases / packages under the MLX Swift examples repo

### Hugging Face / provenance / source policy

- `hf_hub_download` cache semantics:
  https://huggingface.co/docs/huggingface_hub/guides/download

- gated model access docs:
  https://huggingface.co/docs/hub/models-gated

- safetensors metadata parsing without full download:
  https://huggingface.co/docs/safetensors/en/metadata_parsing

- Transformers `trust_remote_code` guidance:
  https://huggingface.co/docs/transformers/models

### LTX / Comfy

- `Lightricks/LTX-2.3` model card:
  https://huggingface.co/Lightricks/LTX-2.3

- official or current LTX project repos:
  https://github.com/Lightricks/LTX-Video

- ComfyUI-LTXVideo current state:
  https://github.com/Lightricks/ComfyUI-LTXVideo

### Apple system / performance

- Apple unified memory and media-engine workflow guidance:
  https://developer.apple.com/videos/play/wwdc2021/10153/

- Metal `recommendedMaxWorkingSetSize`:
  https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize

- Metal `currentAllocatedSize`:
  https://developer.apple.com/documentation/metal/mtldevice/currentallocatedsize

### Security / localhost

- NVD localhost/Origin-validation example (illustrative current risk class):
  https://nvd.nist.gov/vuln/detail/CVE-2026-26317

- OWASP CSRF cheat sheet:
  https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html

### Serving ecosystem / current packages

- `mlx-vlm`:
  https://github.com/Blaizzy/mlx-vlm

- `mlx-audio`:
  https://github.com/Blaizzy/mlx-audio

- `vllm-metal`:
  https://github.com/vllm-project/vllm-metal

### Acceleration research to track, not blindly assume

- AdaCache
- FasterCache
- PAB (Pyramid Attention Broadcast)
- HiCache
- ProCache

Use current papers and their caveats; do not turn paper headlines into architecture facts without benchmark evidence.

---

## Appendix B: direct instructions to yourself when rewriting

1. Rewrite the docs as if they will be read by engineers who will implement them literally.
2. Remove phrasing that sounds settled when it is not.
3. Separate “architecture principle” from “implementation choice” from “measured fact.”
4. Keep the strong decisions.
5. Add the missing ADRs and schemas.
6. Make the pack honest about scope, provider model, and Apple-specific realities.
7. Do not let “universal” become marketing language.

---

## Deliverable expectation for the rewrite

Your revised output should include:

- updated versions of the existing docs
- the new ADRs listed above
- a revised benchmark matrix
- an expanded capability schema
- a corrected storage/provenance model
- a corrected security model
- a clearer language/runtime split
- an explicit MLX extension/upstream roadmap

If you cannot support a “fixed” claim with evidence, mark it **provisional** and define the validation needed to upgrade it.

That is the standard the current docs do not yet meet.
