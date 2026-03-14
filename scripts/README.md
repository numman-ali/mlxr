# Scripts

This directory is reserved for project-owned automation and helper scripts.

Keep scripts here only if they are part of this repo's actual implementation workflow.

When adding a new script, keep two things obvious:

- what class of work it belongs to
- whether it is part of the default development loop or specialist tooling

The flat namespace is intentional for now, so the README and top-of-file
docstring need to do the classification work clearly.

## Core Harness

- `dev.py`: the local development harness used by Codex for fix, verify, build, log inspection, and staged Mac app launch loops
- `run_mypy.py`: the repo-owned mypy gate over packages, tests, and scripts
- `run_unittests.py`: discovers both repo-level `tests/` and package-local `packages/*/*/tests/`
- `check_type_escapes.py`: forbids broad type escapes in the covered runtime and test surfaces; the current enforced scope is `runtime-server`, `runtime-workflows`, `flux2`, `ltx`, `qwen-image`, `z-image`, `runtime-cli`, and repo-level tests

## Runtime And Packaging Utilities

- `benchmark_ltx.py`: compatibility wrapper for the repo-owned LTX benchmark CLI entrypoint
- `runtime_clone_home.py`: clone an MLXR runtime home by reusing installed artifacts and manifests for isolated validation or benchmarking
- `runtime_register_bundle.py`: inspect or register a trusted local model bundle through the runtime API and convert it into an MLXR artifact
- `runtime_link_bundle.py`: register a trusted local bundle directly into a runtime home by linking payload files instead of copying them

## Validation And Showcase Tooling

- `ltx_debug_smoke.py`: the repo-owned real-weight LTX fidelity smoke/debug runner with named profile presets, reviewer stills, stage-debug manifests, and hard safety guardrails; this is transitional engine-diagnostic tooling and should shrink over time as equivalent debug controls move into the canonical `mlxr` CLI surface
- `ltx_showcase_generate.py`: the repo-owned showcase runner that loads scenario definitions from `tests/fixtures/ltx/prompt_scenarios.json`, shapes text-first prompts, runs `ltx_debug_smoke.py`, captures `ffprobe` receipts, and uses the Gemini XML review helper as a promotion gate for showcase scenes
- `gemini_describe_video.py`: a headless Gemini CLI wrapper that stages local video files into a review workspace, extracts a mono WAV review track when audio is present, requests one strict XML review over the video and audio together, and saves parsed validation receipts when `--save-dir` or `--keep-staged-copy` is used
- `gemini_review_image.py`: a headless Gemini CLI wrapper that stages a generated still plus optional reference images, requests one strict XML review over prompt alignment and reference alignment, and saves parsed image-review receipts
- `gemini_review_conditioned_video.py`: a stricter Gemini reviewer for conditioned clips that compares the produced video against prompt semantics and optional start/end/keyframe reference images, with explicit fields for temporal coherence, opening/closing frame match, and stretch-or-squash / aspect-ratio issues
- `qwen_official_eval.py`: the repo-owned heavyweight Qwen evaluator that runs high-resolution base-first `Qwen-Image-2512` generation and `Qwen-Image-Edit-2511` editing through the canonical `mlxr` CLI path, then records manifests, timing receipts, and Gemini image-review sidecars for the promoted comparison runs
- `qwen_decode_oracle.py`: the repo-owned Qwen large-image decode oracle that saves final denoised latents and compares owned MLX decode against official diffusers decode on the exact same latents; use this before guessing at future full-resolution Qwen regressions
- `qwen_showcase_matrix.py`: a focused Qwen profile-comparison runner kept for targeted showcase tuning; it is useful, but it is not part of the default dev or release gate

## Repo Asset Generation

- `generate_readme_assets.py`: regenerate the README logo, hero, showcase grid, and promo reel assets

## Placement Rules

- default dev-loop and repo-gate scripts belong in `Core Harness`
- trusted local runtime utilities should stay narrow and explicit about side
  effects
- validation and showcase tools should stay clearly specialist and must not
  quietly become the public product surface
- any new script should be documented here in the same change that adds it
