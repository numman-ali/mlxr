from __future__ import annotations

import os


def debug_trace_enabled() -> bool:
    return os.environ.get("MLXR_Z_IMAGE_DEBUG_TRACE") == "1"


def debug_trace_sync_enabled() -> bool:
    return os.environ.get("MLXR_Z_IMAGE_DEBUG_TRACE_SYNC") == "1"
