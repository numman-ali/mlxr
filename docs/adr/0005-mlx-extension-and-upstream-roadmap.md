# ADR-0005: MLX Extension And Upstream Roadmap

Status: proposed

## Decision

Reserve an explicit low-level path for:

- profiled local MLX extensions
- custom Metal kernels
- upstream or forked MLX improvements that materially help media workloads

Suggested package seams:

- `packages/runtime-mlx-ext/`
- `packages/runtime-kernels/`

## Why

The earlier docs treated low-level MLX work as a generic late escape hatch. That was too passive.

This project can create value by:

1. using MLX correctly
2. extending MLX locally where profiling proves it matters
3. pushing or carrying targeted MLX improvements for media workloads

## Consequences

### Positive

- low-level work becomes a deliberate roadmap instead of a panic move
- hotspot ownership becomes clearer
- Apple-specific output and memory-path optimizations get a real home

### Negative

- more maintenance burden once native code exists
- greater packaging and notarization complexity

## Notes

No hotspot moves into native code without benchmark evidence. This ADR reserves the seam and the roadmap; it does not authorize speculative kernel work.
