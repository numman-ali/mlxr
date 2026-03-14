# Dev Harness

Status: local-first quality gate for `MLXR`.

## Purpose

This repo is optimized for a Codex-run local feedback loop.

The default gate is a repo-owned harness command, not manual one-off tool invocation.

## Primary Commands

Use the repo harness:

```bash
uv run python scripts/dev.py fix
uv run python scripts/dev.py verify
uv run python scripts/dev.py logs
uv run pre-commit run --all-files
```

## Harness Contract

### `fix`

Runs the fast mutation-safe cleanup loop:

- `ruff format packages tests scripts`
- `ruff check --fix packages tests scripts`

Use this during iteration.

### `verify`

Runs the required pre-commit gate:

- `ruff format --check packages tests scripts`
- `ruff check packages tests scripts`
- `python scripts/run_mypy.py`
- `python scripts/check_type_escapes.py`
- `coverage run scripts/run_unittests.py -v`
- `coverage report` with the current repo floor set to `80%` line coverage across package code under `packages/`
- `uv build --all-packages`

Do not commit without a green `verify`.

### `pre-commit`

`pre-commit` mirrors fast hygiene checks:

- trailing whitespace cleanup
- EOF normalization
- `ruff check --fix`
- `ruff format`
- `python scripts/check_type_escapes.py`

It is a convenience mirror, not the primary acceptance gate.

### `logs`

Reads runtime logs from `$MLX_RUNTIME_HOME/logs/`.

Use it when:

- the daemon lifecycle changed
- source registration or artifact conversion behavior changed
- a runtime error needs confirmation from actual logs

## Tooling Policy

- `ruff` is the formatter and linter
- `scripts/run_mypy.py` is the type gate; it checks the shared `mlxr.*` namespace packages as packages and then checks `tests/` and `scripts/` as file targets
- `check_type_escapes.py` forbids `typing.cast` and explicit `Any` in the agent-facing runtime and test surfaces covered by the script
- `unittest` is the current test runner
- `scripts/run_unittests.py` discovers both repo-level `tests/` and package-local `packages/*/*/tests/`
- package-local parity tests that depend on optional heavyweight reference frameworks such as `torch`, `transformers`, or `diffusers` must skip cleanly when those dependencies are absent from the default dev environment
- `coverage.py` enforces the current line-coverage floor for repo-owned Python package code; test files are not counted toward the package coverage floor
- `uv build --all-packages` is the build gate

Do not add parallel quality tools without a concrete reason.

## Why This Exists

The repo should be legible and mechanically checkable by an agent.

The harness exists to:

- reduce drift
- shorten the feedback loop
- make quality standards executable
- keep the default path obvious
