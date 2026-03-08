# Test Layout

Status: canonical test-organization guide for `MLXR`.

## Purpose

`MLXR` is no longer small enough for giant top-level unit files to remain the default.

The test layout should mirror ownership:

- package-specific unit tests live with the package
- repo-level integration and end-to-end tests stay at the top level
- fixtures stay shared unless they belong clearly to one package

This keeps the codebase easier to navigate, easier to extend, and easier for an agent to maintain without growing monolithic test files.

## Canonical Layout

Use this structure by default:

```text
packages/
  core/
    runtime-server/
      src/...
      tests/
        unit/...
    runtime-core/
      src/...
      tests/
        unit/...
  families/
    ltx/
      src/...
      tests/
        unit/
          adapter/...
          generation/...
          prompt/...
          workflows/...
  clients/
    runtime-cli/
      src/...
      tests/
        unit/...

tests/
  integration/...
  e2e/...
  fixtures/...
  runtime_test_support.py
```

## Rules

Package-local unit tests should live under the package root, adjacent to `src/`, not inside `src/`.

That means:

- yes: `packages/families/ltx/tests/unit/generation/test_audio_runtime.py`
- no: `packages/families/ltx/src/ltx/_generation_backend/test_audio_runtime.py`

Root-level `tests/` should be reserved for:

- integration tests spanning multiple packages
- end-to-end and workflow tests
- shared fixtures and test support
- legacy files that have not been migrated yet

## Naming

Use standard Python names:

- `test_audio_runtime.py`
- `test_workflows.py`
- `test_cli_generate.py`

Do not use TypeScript-style `*.test.py` naming as the repo default. Python tooling and `unittest`/`pytest` conventions are smoother with `test_*.py`.

## Migration Policy

When touching a large monolithic test file:

1. stop adding unrelated tests to it
2. move the changed subsystem's tests into a package-local unit file
3. leave integration coverage in the root-level integration suite

Do not try to migrate every old test file in one shot. Migrate in coherent slices.

## What Belongs Where

Examples for this repo:

- LTX generation helpers, audio processing, VAE loaders, prompt components:
  package-local unit tests under `packages/families/ltx/tests/unit/...`
- runtime server request/response contracts and workflow HTTP behavior:
  root integration tests under `tests/`
- CLI end-to-end generation behavior against the daemon:
  root integration tests under `tests/`
- shared image bytes, prompt scenarios, or reusable helper code:
  `tests/fixtures/` and `tests/runtime_test_support.py`

## Harness Contract

The repo harness must run both:

- top-level repo tests under `tests/`
- package-local tests under `packages/*/*/tests/`

If you add package-local tests, do not assume the old `unittest discover -s tests` path is enough. Keep the harness and docs in sync.
