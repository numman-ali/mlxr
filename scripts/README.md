# Scripts

This directory is reserved for project-owned automation and helper scripts.

Keep scripts here only if they are part of this repo's actual implementation workflow.

Current primary entrypoints:

- `dev.py`: the local development harness used by Codex for fix, verify, build, and log inspection loops
- `benchmark_ltx.py`: the repo-owned LTX benchmark harness
- `ltx_debug_smoke.py`: the repo-owned real-weight LTX fidelity smoke/debug runner with named profile presets, reviewer stills, and stage-debug manifests
- `gemini_describe_video.py`: a headless Gemini CLI wrapper that stages local video files into a review workspace, requests a strict XML review, and saves parsed validation receipts when `--save-dir` or `--keep-staged-copy` is used
