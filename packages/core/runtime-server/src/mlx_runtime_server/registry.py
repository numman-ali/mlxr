from __future__ import annotations

from ltx import LTXFamilyAdapter, LTXWorkflowStrategy
from mlx_runtime_core import (
    HuggingFaceProviderAdapter,
    LocalFileProviderAdapter,
    RuntimeRegistry,
)
from mlx_runtime_workflows import WorkflowPlanner, WorkflowStrategyRegistry


def default_runtime_registry() -> RuntimeRegistry:
    registry = RuntimeRegistry()
    registry.register_provider(LocalFileProviderAdapter())
    registry.register_provider(HuggingFaceProviderAdapter())
    registry.register_family(LTXFamilyAdapter())
    return registry


def default_workflow_planner() -> WorkflowPlanner:
    registry = WorkflowStrategyRegistry()
    registry.register(LTXWorkflowStrategy())
    return WorkflowPlanner(registry)
