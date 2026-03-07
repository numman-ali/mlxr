"""CLI package scaffold for MLXR."""

from .benchmark_ltx import (
    DEFAULT_CONDITIONING_IMAGE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULTS_DIR,
    BenchmarkEnvironment,
    JobRunResult,
    RuntimeApi,
    ScenarioBenchmarkResult,
    ScenarioConfig,
    SourceSetupResult,
    StageMetricResult,
    build_default_scenarios,
    main,
    run_scenario_with_api,
    running_runtime_daemon,
    write_benchmark_result,
)
from .cli import main as cli_main

__all__ = [
    "BenchmarkEnvironment",
    "DEFAULT_CONDITIONING_IMAGE",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_RESULTS_DIR",
    "cli_main",
    "JobRunResult",
    "RuntimeApi",
    "ScenarioBenchmarkResult",
    "ScenarioConfig",
    "SourceSetupResult",
    "StageMetricResult",
    "build_default_scenarios",
    "main",
    "run_scenario_with_api",
    "running_runtime_daemon",
    "write_benchmark_result",
]
