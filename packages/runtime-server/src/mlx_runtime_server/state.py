from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mlx_runtime_core import (
    RuntimeCatalog,
    RuntimeHome,
    RuntimeRegistry,
)
from mlx_runtime_workflows import WorkflowPlanner

from .jobs import JobManager
from .logging import configure_control_plane_logging, get_control_plane_logger
from .registry import default_runtime_registry, default_workflow_planner
from .settings import ServerSettings
from .store import InputStore, JobStore, OutputStore
from .workflows import WorkflowService


@dataclass
class RuntimeState:
    registry: RuntimeRegistry = field(default_factory=default_runtime_registry)
    runtime_home: RuntimeHome = field(default_factory=RuntimeHome.from_env)
    settings: ServerSettings = field(default_factory=ServerSettings.from_env)
    workflow_planner: WorkflowPlanner = field(default_factory=default_workflow_planner)
    catalog: RuntimeCatalog = field(init=False)
    input_store: InputStore = field(init=False)
    output_store: OutputStore = field(init=False)
    job_store: JobStore = field(init=False)
    job_manager: JobManager = field(init=False)
    workflow_service: WorkflowService = field(init=False)
    log_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.catalog = RuntimeCatalog(self.registry, self.runtime_home)
        self.input_store = InputStore(self.runtime_home)
        self.output_store = OutputStore(self.runtime_home)
        self.job_store = JobStore(self.runtime_home)
        self.job_manager = JobManager(
            catalog=self.catalog,
            job_store=self.job_store,
            input_store=self.input_store,
            output_store=self.output_store,
            settings=self.settings,
        )
        self.workflow_service = WorkflowService(
            catalog=self.catalog,
            planner=self.workflow_planner,
            job_manager=self.job_manager,
        )
        self.log_path = configure_control_plane_logging(self.runtime_home)
        get_control_plane_logger().info(
            "Control-plane state initialized runtime_home=%s log_path=%s http_enabled=%s",
            self.runtime_home.root,
            self.log_path,
            self.settings.http_enabled,
        )
