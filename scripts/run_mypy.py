from __future__ import annotations

import subprocess
import sys

PACKAGE_TARGETS = (
    "mlxr.core.schemas",
    "mlxr.core.runtime",
    "mlxr.core.server",
    "mlxr.core.workflows",
    "mlxr.families.ltx",
    "mlxr.clients.cli",
)
FILE_TARGETS = ("tests", "scripts")


def run_command(command: list[str]) -> None:
    print(f"+ {' '.join(command)}")
    subprocess.run(command, check=True)


def main() -> None:
    package_command = [sys.executable, "-m", "mypy", "--strict"]
    for package in PACKAGE_TARGETS:
        package_command.extend(["-p", package])
    run_command(package_command)
    run_command([sys.executable, "-m", "mypy", "--strict", *FILE_TARGETS])


if __name__ == "__main__":
    main()
