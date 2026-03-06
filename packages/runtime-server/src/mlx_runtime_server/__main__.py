from __future__ import annotations

import os

import uvicorn
from mlx_runtime_core import RuntimeHome

from .app import app


def main() -> None:
    http_host = os.environ.get("MLX_RUNTIME_HTTP_HOST")
    http_port = int(os.environ.get("MLX_RUNTIME_HTTP_PORT", "46321"))
    if http_host:
        uvicorn.run(app, host=http_host, port=http_port)
        return

    runtime_home = RuntimeHome.from_env()
    runtime_home.ensure_layout()
    socket_dir = runtime_home.temp_dir
    socket_dir.mkdir(parents=True, exist_ok=True)
    socket_path = socket_dir / "control-plane.sock"
    if socket_path.exists():
        socket_path.unlink()
    uvicorn.run(app, uds=str(socket_path))


if __name__ == "__main__":
    main()
