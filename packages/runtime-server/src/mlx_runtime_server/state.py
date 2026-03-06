from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from mlx_runtime_core import RuntimeRegistry
from mlx_runtime_schemas import JobRecord, ModelRecord, RuntimeEvent


@dataclass
class RuntimeState:
    registry: RuntimeRegistry = field(default_factory=RuntimeRegistry)
    models: dict[str, ModelRecord] = field(default_factory=dict)
    jobs: dict[str, JobRecord] = field(default_factory=dict)
    events: dict[str, list[RuntimeEvent]] = field(default_factory=lambda: defaultdict(list))
