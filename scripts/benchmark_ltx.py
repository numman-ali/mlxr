"""Compatibility wrapper for the repo-owned LTX benchmark CLI entrypoint."""

from __future__ import annotations

from mlxr.clients.cli.benchmark_ltx import main

if __name__ == "__main__":
    raise SystemExit(main())
