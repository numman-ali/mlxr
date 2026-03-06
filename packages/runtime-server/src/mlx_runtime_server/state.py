from __future__ import annotations

from dataclasses import dataclass, field

from mlx_runtime_core import (
    LocalFileProviderAdapter,
    RuntimeCatalog,
    RuntimeHome,
    RuntimeRegistry,
)
from mlx_runtime_family_ltx import LTXFamilyAdapter


def _default_registry() -> RuntimeRegistry:
    registry = RuntimeRegistry()
    registry.register_provider(LocalFileProviderAdapter())
    registry.register_family(LTXFamilyAdapter())
    return registry


@dataclass
class RuntimeState:
    registry: RuntimeRegistry = field(default_factory=_default_registry)
    runtime_home: RuntimeHome = field(default_factory=RuntimeHome.from_env)
    catalog: RuntimeCatalog = field(init=False)

    def __post_init__(self) -> None:
        self.catalog = RuntimeCatalog(self.registry, self.runtime_home)
