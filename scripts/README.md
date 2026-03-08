# Scripts

This directory is reserved for project-owned automation and helper scripts.

Keep scripts here only if they are part of this repo's actual implementation workflow.

Current primary entrypoints:

- `dev.py`: the local development harness used by Codex for fix, verify, build, and log inspection loops
- `benchmark_ltx.py`: the repo-owned LTX benchmark harness
- `ltx_debug_smoke.py`: the repo-owned real-weight LTX fidelity smoke/debug runner with named profile presets, reviewer stills, and stage-debug manifests
- `gemini_describe_video.py`: a headless Gemini CLI wrapper that stages local video files into a review workspace, extracts a mono WAV review track when audio is present, requests one strict XML review over the video and audio together, and saves parsed validation receipts when `--save-dir` or `--keep-staged-copy` is used
- `ltx_showcase_generate.py`: the repo-owned showcase runner that shapes text-first prompts, runs `ltx_debug_smoke.py`, captures `ffprobe` receipts, and uses the Gemini XML review helper as a promotion gate for showcase scenes
