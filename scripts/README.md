# Scripts

This directory is reserved for project-owned automation and helper scripts.

Keep scripts here only if they are part of this repo's actual implementation workflow.

Current primary entrypoints:

- `dev.py`: the local development harness used by Codex for fix, verify, build, and log inspection loops
- `benchmark_ltx.py`: the repo-owned LTX benchmark harness
- `ltx_debug_smoke.py`: the repo-owned real-weight LTX fidelity smoke/debug runner with named profile presets, reviewer stills, stage-debug manifests, and hard safety guardrails; this is transitional engine-diagnostic tooling and should shrink over time as equivalent debug controls move into the canonical `mlxr` CLI surface
- `gemini_describe_video.py`: a headless Gemini CLI wrapper that stages local video files into a review workspace, extracts a mono WAV review track when audio is present, requests one strict XML review over the video and audio together, and saves parsed validation receipts when `--save-dir` or `--keep-staged-copy` is used
- `gemini_review_image.py`: a headless Gemini CLI wrapper that stages a generated still plus optional reference images, requests one strict XML review over prompt alignment and reference alignment, and saves parsed image-review receipts
- `gemini_review_conditioned_video.py`: a stricter Gemini reviewer for conditioned clips that compares the produced video against prompt semantics and optional start/end/keyframe reference images, with explicit fields for temporal coherence, opening/closing frame match, and stretch-or-squash / aspect-ratio issues
- `qwen_official_eval.py`: the repo-owned heavyweight Qwen evaluator that runs high-resolution base-first `Qwen-Image-2512` generation and `Qwen-Image-Edit-2511` editing through the canonical `mlxr` CLI path, then records manifests, timing receipts, and Gemini image-review sidecars for the promoted comparison runs
- `qwen_decode_oracle.py`: the repo-owned Qwen large-image decode oracle that saves final denoised latents and compares owned MLX decode against official diffusers decode on the exact same latents; use this before guessing at future full-resolution Qwen regressions
- `ltx_showcase_generate.py`: the repo-owned showcase runner that loads scenario definitions from `tests/fixtures/ltx/prompt_scenarios.json`, shapes text-first prompts, runs `ltx_debug_smoke.py`, captures `ffprobe` receipts, and uses the Gemini XML review helper as a promotion gate for showcase scenes
