# MLXR Roadmap

Last updated: 2026-03-14

## Summary

The roadmap now has one clear app direction:

- open-source runtime + CLI as the platform
- one first-party native Mac app evolving into the unified studio shell

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

### 3. First-party Mac app

The current state is:

- native Swift package scaffold is real
- runtime bridge, model install, advanced import, and shared workflow seams are
  real
- the app is already moving toward one unified studio shell with a shared
  composer, project-first library, and rail-based activity
- the next work is hardening, polish, and release posture, not architecture
  invention

The next app tranche is:

- talks to the shared runtime
- starts or attaches to the local runtime cleanly
- exposes real + supported image and video flows through one coherent consumer
  app
- keeps advanced or unpromoted rows visibly separate from recommended defaults
- treats prompt enhancement as an optional host-side helper
- improves preview, onboarding, validation, documentation, and release
  packaging

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

## Ordering

1. OSS foundation and repo truth
2. CLI public polish
3. first-party Mac app hardening
4. capability closure and adapter growth
5. broader adapter and product expansion

## Not In The First Milestone

- full desktop parity
- fully closed LTX `dev` family
- prompt enhancement as a core runtime dependency
- every possible family-specific advanced control in the shared public surface
- a second host-owned studio architecture
