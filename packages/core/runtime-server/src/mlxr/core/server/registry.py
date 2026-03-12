from __future__ import annotations

from mlxr.core.runtime import (
    HuggingFaceProviderAdapter,
    LocalFileProviderAdapter,
    RuntimeRegistry,
)
from mlxr.core.workflows import WorkflowPlanner, WorkflowStrategyRegistry
from mlxr.families.flux2 import Flux2FamilyAdapter, Flux2WorkflowStrategy
from mlxr.families.ltx import LTXFamilyAdapter, LTXWorkflowStrategy
from mlxr.families.qwen_image import QwenImageFamilyAdapter, QwenImageWorkflowStrategy
from mlxr.families.z_image import ZImageFamilyAdapter, ZImageWorkflowStrategy


def default_runtime_registry() -> RuntimeRegistry:
    registry = RuntimeRegistry()
    registry.register_provider(LocalFileProviderAdapter())
    registry.register_provider(HuggingFaceProviderAdapter())
    registry.register_family(Flux2FamilyAdapter())
    registry.register_family(LTXFamilyAdapter())
    registry.register_family(QwenImageFamilyAdapter())
    registry.register_family(ZImageFamilyAdapter())
    return registry


def default_workflow_planner() -> WorkflowPlanner:
    registry = WorkflowStrategyRegistry()
    registry.register(Flux2WorkflowStrategy())
    registry.register(LTXWorkflowStrategy())
    registry.register(QwenImageWorkflowStrategy())
    registry.register(ZImageWorkflowStrategy())
    return WorkflowPlanner(registry)
