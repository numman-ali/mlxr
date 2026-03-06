from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mlx_runtime_core import (
    LocalFileProviderAdapter,
    RuntimeCatalog,
    RuntimeHome,
    RuntimeRegistry,
)
from mlx_runtime_family_ltx import LTXFamilyAdapter

from .logging import configure_control_plane_logging, get_control_plane_logger


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
    log_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.catalog = RuntimeCatalog(self.registry, self.runtime_home)
        self.log_path = configure_control_plane_logging(self.runtime_home)
        get_control_plane_logger().info(
            "Control-plane state initialized runtime_home=%s log_path=%s",
            self.runtime_home.root,
            self.log_path,
        )
