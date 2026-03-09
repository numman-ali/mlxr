from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Protocol, TypeAlias

import mlx.core as mx
import numpy as np
from safetensors import safe_open

MLXArray: TypeAlias = mx.array


class _SafeOpenHandle(Protocol):
    def __enter__(self) -> "_SafeOpenHandle": ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None: ...
    def keys(self) -> list[str]: ...
    def metadata(self) -> dict[str, str] | None: ...
    def get_tensor(self, key: str) -> np.ndarray: ...


def _safe_open_numpy(checkpoint_path: Path) -> _SafeOpenHandle:
    return safe_open(str(checkpoint_path), framework="numpy")  # type: ignore[no-untyped-call]


class CheckpointIndex:
    """Lightweight checkpoint metadata and key index without tensor residency."""

    def __init__(self, checkpoint_path: Path) -> None:
        self._checkpoint_path = checkpoint_path
        self._keys: frozenset[str] | None = None
        self._metadata: dict[str, str] | None = None

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    def keys(self) -> frozenset[str]:
        if self._keys is None:
            with _safe_open_numpy(self._checkpoint_path) as handle:
                self._keys = frozenset(handle.keys())
        return self._keys

    def metadata(self) -> dict[str, str]:
        if self._metadata is None:
            with _safe_open_numpy(self._checkpoint_path) as handle:
                self._metadata = dict(handle.metadata() or {})
        return self._metadata


class CheckpointReader:
    """Lifecycle-owned staged checkpoint reader with scoped tensor residency."""

    def __init__(self, checkpoint_path: Path) -> None:
        self._index = CheckpointIndex(checkpoint_path)
        self._resident: dict[str, MLXArray] = {}

    @property
    def checkpoint_path(self) -> Path:
        return self._index.checkpoint_path

    def keys(self) -> frozenset[str]:
        return self._index.keys()

    def metadata(self) -> dict[str, str]:
        return self._index.metadata()

    def load_exact(
        self,
        keys: Iterable[str],
        *,
        required: bool = True,
    ) -> dict[str, MLXArray]:
        requested = tuple(dict.fromkeys(keys))
        missing = [key for key in requested if key not in self.keys()]
        if missing and required:
            sample = ", ".join(sorted(missing)[:5])
            raise RuntimeError(
                f"LTX checkpoint '{self.checkpoint_path}' is missing required weights: {sample}"
            )
        selected = [key for key in requested if key not in missing]
        if not selected:
            return {}
        self._ensure_loaded(selected)
        return {key: self._resident[key] for key in selected}

    def load_prefixes(
        self,
        prefixes: tuple[str, ...],
        *,
        required: bool = True,
    ) -> dict[str, MLXArray]:
        selected = tuple(key for key in self.keys() if key.startswith(prefixes))
        if not selected and required:
            raise RuntimeError(
                f"LTX checkpoint '{self.checkpoint_path}' is missing required prefixed weights for {prefixes!r}"
            )
        if not selected:
            return {}
        self._ensure_loaded(selected)
        return {key: self._resident[key] for key in selected}

    def release(self, *, keys: Iterable[str] | None = None) -> None:
        if keys is None:
            if self._resident:
                self._resident.clear()
                mx.clear_cache()
            return
        released = False
        for key in tuple(keys):
            if key in self._resident:
                del self._resident[key]
                released = True
        if released:
            mx.clear_cache()

    def _ensure_loaded(self, keys: Iterable[str]) -> None:
        missing = [key for key in keys if key not in self._resident]
        if not missing:
            return
        with _safe_open_numpy(self.checkpoint_path) as handle:
            for key in missing:
                self._resident[key] = mx.array(handle.get_tensor(key))


class CheckpointReaderView(Mapping[str, MLXArray]):
    """Scoped mapping view over a reader-owned set of resident checkpoint keys."""

    def __init__(self, reader: CheckpointReader, keys: Iterable[str]) -> None:
        self._reader = reader
        self._keys = tuple(dict.fromkeys(keys))

    def __getitem__(self, key: str) -> MLXArray:
        if key not in self._keys:
            raise KeyError(key)
        return self._reader.load_exact((key,))[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def materialize(self) -> dict[str, MLXArray]:
        return self._reader.load_exact(self._keys)

    def release(self) -> None:
        self._reader.release(keys=self._keys)
