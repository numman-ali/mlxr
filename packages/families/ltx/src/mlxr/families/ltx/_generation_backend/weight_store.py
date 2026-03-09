from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias

import mlx.core as mx

MLXArray: TypeAlias = mx.array


class CheckpointWeightStore:
    """Lifecycle-owned checkpoint weight broker for one generator run."""

    def __init__(self, checkpoint_path: Path) -> None:
        self._checkpoint_path = checkpoint_path
        self._weights: dict[str, MLXArray] | None = None

    def all_weights(self) -> Mapping[str, MLXArray]:
        if self._weights is None:
            loaded = mx.load(str(self._checkpoint_path))
            if not isinstance(loaded, dict):
                raise RuntimeError(
                    f"LTX checkpoint '{self._checkpoint_path}' did not load into a weight mapping"
                )
            self._weights = loaded
        return self._weights

    def prefixed_weights(self, prefixes: tuple[str, ...]) -> dict[str, MLXArray]:
        weights = self.all_weights()
        selected = {
            key: value for key, value in weights.items() if key.startswith(prefixes)
        }
        if not selected:
            raise RuntimeError(
                f"LTX checkpoint '{self._checkpoint_path}' is missing required prefixed weights for {prefixes!r}"
            )
        return selected

    def release(self) -> None:
        if self._weights is not None:
            self._weights = None
            mx.clear_cache()
