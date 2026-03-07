# mypy: ignore-errors
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
_REFERENCE_MLX_VIDEO_ROOT = _REPO_ROOT / "references" / "ecosystem" / "mlx-video"


@contextmanager
def _reference_path_on_sys_path() -> object:
    if not _REFERENCE_MLX_VIDEO_ROOT.is_dir():
        raise RuntimeError(
            "LTX real generation requires the repo-local reference checkout at "
            f"'{_REFERENCE_MLX_VIDEO_ROOT}'"
        )
    reference_path = str(_REFERENCE_MLX_VIDEO_ROOT)
    already_present = reference_path in sys.path
    if not already_present:
        sys.path.insert(0, reference_path)
    try:
        yield
    finally:
        if not already_present:
            try:
                sys.path.remove(reference_path)
            except ValueError:
                return None
