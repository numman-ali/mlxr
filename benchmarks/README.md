# Benchmarks

This directory is reserved for profiling harnesses, benchmark inputs, and measured results owned by this repo.

Do not place third-party repos or ad hoc scratch work here.

## First Harness

The first repo-owned LTX benchmark entrypoint is:

```bash
uv run python scripts/benchmark_ltx.py --bundle-path /path/to/ltx-bundle
```

If you want the harness to pull directly from Hugging Face instead of preparing a local bundle first, run:

```bash
uv run python scripts/benchmark_ltx.py --source-mode huggingface --scenario t2v
```

Defaults:

- transport: Unix domain socket against the real daemon
- source mode: trusted local bundle
- scenarios: `t2v` and `i2v`
- conditioning fixture: `benchmarks/fixtures/conditioning.ppm`

Generated outputs stay local and gitignored:

- structured JSON results: `benchmarks/results/<session-label>/`
- runtime homes, downloaded artifacts, and daemon stdio logs: `benchmarks/output/<session-label>/`

`local-bundle` means a local folder that already contains the LTX checkpoint, spatial upsampler, and text encoder in the layout expected by the family adapter. Most users should prefer `--source-mode huggingface` unless they already have a trusted local export.

When `--source-mode huggingface` is used, the harness follows standard Hugging Face auth behavior:

- if you have already run `hf auth login`, the saved machine login is reused
- `HF_TOKEN` still works as the explicit override
- the harness isolates Hub download caches with `HF_HUB_CACHE` and `HF_XET_CACHE`
- the harness does not replace `HF_HOME`, so a normal CLI login stays visible to the runtime

The benchmark daemon may use a short explicit Unix socket path under `/tmp/mlxr-bench-uds/` to avoid macOS `AF_UNIX path too long` failures when the benchmark output directory is nested deeply.

## Metric Semantics

- `route_wall_ms`: end-to-end client wall clock for a single route call
- `job_wait_ms`: time from an accepted submit response until the job reaches a terminal state
- `download_ms`: time spent on the runtime-managed artifact download route only
- `timings_ms` on inspect and convert responses: server-side breakdowns returned by the runtime
- `stage_metrics[*].duration_ms`: worker-side stage timing from `job.metrics`

When a metric is not implemented yet, benchmark results record it under `unavailable_metrics` instead of inferring or fabricating a value.
