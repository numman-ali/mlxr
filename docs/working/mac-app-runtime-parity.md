# MLXR Mac App Runtime Hardening Plan

Status: active working plan for Mac app runtime hardening

Last updated: 2026-03-14

## Summary

This is the current working plan for hardening the Mac app against the shared
runtime contract.

The durable boundary rules now live in:

- [mac-app-runtime-contract.md](/Users/numman/Repos/mlxr/docs/mac-app-runtime-contract.md)

This file is for active tranche execution only.

## Product Shape

Top-level surfaces stay fixed:

- `Home`
- `Library`
- `Models`
- `Settings`

`Activity` stays secondary, and the shared composer plus project-first
`Library` stay the visible creation path.

## Current Concrete Direction

The current hardening path is:

1. enrich runtime planning readiness and capability normalization
2. keep CLI `--plan-only` aligned with the same readiness truth
3. keep the shared composer runtime-led and draft-persistent
4. keep Activity secondary and truthful
5. keep Library fast, modal, and asset-first
6. keep docs aligned with the real app/runtime contract

## Done Definition

This tranche is only done when all of these hold together:

- `swift test --package-path packages/clients/mlxr-mac-app`
- targeted runtime/schema tests for planning and validation
- `uv run python scripts/dev.py verify`
- staged dev `.app` validation with Peekaboo
- runtime logs show no duplicate daemon spawning or invalid-request spam
