from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from .._nn_compat import Dropout, Linear, Module, RMSNorm, gelu_approx
from . import runtime

if TYPE_CHECKING:
    mx = runtime.mx
    np = runtime.np

    class TextConfig(Protocol):
        hidden_size: int
        num_hidden_layers: int
        sliding_window: int
        sliding_window_pattern: int

    class _EmbedTokens(Protocol):
        def __call__(self, inputs: mx.array) -> mx.array: ...
        def as_linear(self, hidden: mx.array) -> mx.array: ...

    class _GemmaLayer(Protocol):
        def __call__(
            self,
            hidden: mx.array,
            attention_mask: mx.array | str | None,
            cache: object | None,
        ) -> mx.array: ...

    class _GemmaNorm(Protocol):
        def __call__(self, hidden: mx.array) -> mx.array: ...

    class Gemma3Model(Module):
        embed_tokens: _EmbedTokens
        layers: list[_GemmaLayer]
        norm: _GemmaNorm

        def __init__(self, config: object) -> None: ...

    class SafeOpenHandle(Protocol):
        def __enter__(self) -> SafeOpenHandle: ...
        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None: ...
        def keys(self) -> list[str]: ...
        def metadata(self) -> dict[str, str] | None: ...

    def safe_open(path: str, *, framework: str) -> SafeOpenHandle: ...
else:
    mx = runtime.mx
    np = runtime.np
    TextConfig = runtime.TextConfig
    Gemma3Model = runtime.Gemma3Model
    safe_open = runtime.safe_open


__all__ = [
    "Dropout",
    "Gemma3Model",
    "Linear",
    "Module",
    "RMSNorm",
    "TextConfig",
    "gelu_approx",
    "mx",
    "np",
    "safe_open",
]
