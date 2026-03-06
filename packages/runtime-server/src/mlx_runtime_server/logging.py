from __future__ import annotations

import logging
import sys
from pathlib import Path

from mlx_runtime_core import RuntimeHome

LOGGER_NAME = "mlxr.control_plane"


def configure_control_plane_logging(runtime_home: RuntimeHome) -> Path:
    runtime_home.ensure_layout()
    log_path = runtime_home.logs_dir / "control-plane.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return log_path


def get_control_plane_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)
