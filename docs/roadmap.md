# MLXR Roadmap

Last updated: 2026-03-12

## Summary

The roadmap now has two product horizons:

- first: open-source runtime + CLI + simple native Mac app
- later: `MLXR Studio` as a deeper AI-native generative suite

The runtime remains the center of gravity. Every client should stay thin over
the same capability, job, provenance, and artifact model.

## Now

### 1. Open-source foundation

- consolidate current truth into public-facing status docs
- refresh the README around what is real today
- add license, contributing, support, security, and conduct docs
- define the public maintenance and release posture
- get the repo back to a clean release-ready validation state

### 2. Runtime and CLI public polish

- keep `mlxr generate` as the current canonical thin client flow
- document the intended CLI growth for:
  - `mlxr models *`
  - `mlxr jobs *`
  - `mlxr serve`
- keep the CLI agent-friendly and machine-readable
- avoid drifting family-specific inference logic into the CLI

## Next

### 3. Simple native Mac app

The first app is not `MLXR Studio`.

It is a simple macOS app that:

- talks to the shared runtime
- starts or attaches to the local runtime cleanly
- exposes real + supported image and video flows
- keeps advanced or unpromoted rows visibly separate from recommended defaults
- treats prompt enhancement as an optional host-side helper

### 4. Capability closure

The next capability work after the OSS foundation is:

- LTX `dev` closure: standard two-stage, HQ, retake, interpolation, broader
  IC-LoRA validation
- Qwen review and promotion coverage for already-real rows
- Z-Image public positioning and base-row maturation
- FLUX.2 multi-reference, base-row, and later `dev` closure

## Later

### 5. Thin adapter expansion

- `ltx-desktop` compatibility adapter
- Comfy adapter
- clearer embedded first-party host boundary

### 6. MLXR Studio discovery

`MLXR Studio` is a separate product track.

It should only start after the simpler first-party app proves:

- the runtime story is stable enough
- the first-party client ergonomics are understood
- the capability map is public and trustworthy

Studio should be treated as a researched product design effort, not as a
rename of the first simple app.

## Ordering

1. OSS foundation and repo truth
2. CLI public polish
3. simple Mac app v1
4. capability closure and adapter growth
5. Studio discovery

## Not In The First Milestone

- full desktop parity
- fully closed LTX `dev` family
- prompt enhancement as a core runtime dependency
- every possible family-specific advanced control in the shared public surface
- Studio-level UX or orchestration
