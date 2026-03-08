from __future__ import annotations

from mlxr.core.runtime import (
    HuggingFaceProviderAdapter,
    LocalFileProviderAdapter,
    RuntimeRegistry,
)
from mlxr.core.workflows import WorkflowPlanner, WorkflowStrategyRegistry
from mlxr.families.ltx import LTXFamilyAdapter, LTXWorkflowStrategy


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
