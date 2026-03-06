# MEMORY.md

This file is the durable cross-session memory for `MLXR`.

Use it to preserve stable repo-operating learnings so fresh Codex sessions do not have to rediscover them.

## How To Use This File

### Tier 1 — Must Read

- Read this section at the start of every session, immediately after `AGENTS.md`.
- Keep it short and durable.
- Promote only facts that should shape most future sessions.

### Tier 2 — Lookup Log

- Do not read this end to end by default.
- Search it when working on a related area.
- Append durable learnings here at the end of a session.

## Memory Update Rule

- Add durable learnings to Tier 2.
- Promote only cross-session critical items to Tier 1.
- Keep Tier 1 focused on durable repo facts and sharp edges, not doctrine already stated in `AGENTS.md`.
- If a rule belongs in repo operating doctrine, put it in `AGENTS.md` instead of repeating it here.
- Archive obsolete items instead of silently deleting them.
- Do not put transient task notes or diary-style entries here.

## Tier 1 — Must Read

- The default local quality gate is `uv run python scripts/dev.py verify`.
- `pre-commit` is a fast hygiene mirror, not the main acceptance gate.
- If runtime behavior changes, inspect `$MLX_RUNTIME_HOME/logs/control-plane.log` as part of validation.
- `verify` and `pre-commit` run `scripts/check_type_escapes.py`; avoid `typing.cast` and explicit `Any` in the scanned runtime and test surfaces.
- Placeholder package folders stay outside the `uv` workspace until they gain a real `pyproject.toml`.

## Tier 2 — Lookup Log

- (2026-03-06) [HARNESS] The repo uses explicit `uv` workspace membership and `tool.uv.sources`; root `uv sync`, `uv lock`, and `uv run` should work without extra flags. — refs: `pyproject.toml`, `uv.lock`
- (2026-03-06) [QUALITY] The current local quality stack is `ruff`, `mypy --strict`, `unittest`, and `uv build --all-packages`. — refs: `pyproject.toml`, `scripts/dev.py`
- (2026-03-06) [LOGS] The control plane logs to `$MLX_RUNTIME_HOME/logs/control-plane.log`, and runtime log inspection is part of the default dev loop. — refs: `packages/runtime-server/src/mlx_runtime_server/logging.py`, `docs/dev-harness.md`
- (2026-03-06) [SKILLS] The repo-owned specialized validation overlay is `skills/validation/`; it is narrow by design and not a replacement for the common startup flow. — refs: `skills/validation/SKILL.md`, `AGENTS.md`
- (2026-03-06) [COVERAGE] `verify` enforces an `85%` package coverage floor via `coverage.py`; the floor applies to `packages/`, not repo scripts. — refs: `pyproject.toml`, `scripts/dev.py`
- (2026-03-06) [INSPECTION] Family preflight inspection now runs from `ResolvedSource` without provider fetch; selective provider fetch happens at conversion time through the family adapter’s fetch-policy seam. Hugging Face provenance also preserves `remote_code_approved` from source policy. — refs: `packages/runtime-core/src/mlx_runtime_core/catalog.py`, `packages/runtime-core/src/mlx_runtime_core/contracts.py`, `packages/runtime-core/src/mlx_runtime_core/providers.py`, `packages/model-family-ltx/src/mlx_runtime_family_ltx/adapter.py`
- (2026-03-06) [ARTIFACTS] LTX Phase B now uses a family-generic multi-source conversion contract: `ArtifactConversionRequest` accepts exactly one of `source_id` or `source_bindings`, portable artifacts can carry typed `components`, and the first truthful LTX slice materializes copied payloads for `checkpoint`, `spatial_upsampler`, and `text_encoder` under `artifacts-portable/.../payload/`. — refs: `packages/shared-schemas/src/mlx_runtime_schemas/models.py`, `packages/runtime-core/src/mlx_runtime_core/catalog.py`, `packages/model-family-ltx/src/mlx_runtime_family_ltx/adapter.py`, `docs/research/07-ltx-integration-seams.md`
- (2026-03-06) [PROMPT-ENCODE] The next truthful LTX execution seam is now real: artifactized LTX models lazily create a strict-local MLX Gemma prompt encoder at `prompt_encode`, store prompt context in worker-local runtime state, and keep denoise/generate scaffolded until a later slice. Job inputs with non-empty `negative_prompt` now fail clearly for the fast path instead of being ignored. — refs: `packages/model-family-ltx/src/mlx_runtime_family_ltx/prompt_encoding.py`, `packages/model-family-ltx/src/mlx_runtime_family_ltx/adapter.py`, `packages/model-family-ltx/README.md`
- (2026-03-06) [PROMPT-ENCODE-HARNESS] The typed public prompt-encode seam now lives in `prompt_encoding.py`, while the large MLX implementation is isolated in `_prompt_encoding_backend.py` and omitted from repo coverage until it has real asset-backed tests. API job tests patch `mlx_runtime_server.jobs._job_process_context()` to an in-process threaded context so prompt-encoder patches survive worker execution without changing production spawn behavior. — refs: `packages/model-family-ltx/src/mlx_runtime_family_ltx/prompt_encoding.py`, `packages/model-family-ltx/src/mlx_runtime_family_ltx/_prompt_encoding_backend.py`, `tests/runtime_test_support.py`, `pyproject.toml`
- (2026-03-06) [LTX-EXECUTION] The first non-placeholder LTX runtime path now resolves imported image handles inside the worker, emits per-stage timing plus MLX memory telemetry, narrows truthful fast-path outputs to runtime-managed `mp4` video, and generates encoded preview video artifacts without claiming checkpoint-faithful denoise yet. — refs: `packages/model-family-ltx/src/mlx_runtime_family_ltx/adapter.py`, `packages/model-family-ltx/src/mlx_runtime_family_ltx/generation.py`, `packages/runtime-server/src/mlx_runtime_server/worker.py`, `docs/research/07-ltx-integration-seams.md`
