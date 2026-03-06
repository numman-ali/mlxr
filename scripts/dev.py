from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

TARGETS = ("packages", "tests", "scripts")


def runtime_home() -> Path:
    raw_root = os.environ.get("MLX_RUNTIME_HOME", str(Path.home() / ".mlx-runtime"))
    return Path(raw_root).expanduser()


def run_command(command: Sequence[str]) -> None:
    print(f"+ {' '.join(command)}")
    subprocess.run(command, check=True)


def read_tail(path: Path, lines: int) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()[-lines:]


def cmd_fix() -> None:
    run_command(["ruff", "format", *TARGETS])
    run_command(["ruff", "check", "--fix", *TARGETS])


def cmd_lint() -> None:
    run_command(["ruff", "check", *TARGETS])


def cmd_typecheck() -> None:
    run_command(["mypy", "--strict", *TARGETS])


def cmd_test() -> None:
    run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])


def cmd_build() -> None:
    run_command(["uv", "build", "--all-packages"])


def cmd_verify() -> None:
    run_command(["ruff", "format", "--check", *TARGETS])
    run_command(["ruff", "check", *TARGETS])
    run_command(["mypy", "--strict", *TARGETS])
    run_command([sys.executable, "scripts/check_type_escapes.py"])
    Path("tmp").mkdir(parents=True, exist_ok=True)
    run_command(["coverage", "erase"])
    run_command(["coverage", "run", "-m", "unittest", "discover", "-s", "tests", "-v"])
    run_command(["coverage", "report"])
    run_command(["uv", "build", "--all-packages"])


def cmd_logs(list_only: bool, tail: int, errors_only: bool) -> None:
    logs_dir = runtime_home() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_paths = sorted(path for path in logs_dir.glob("*.log") if path.is_file())

    print(f"Runtime home: {runtime_home()}")
    if not log_paths:
        print("No log files found.")
        return

    print("Available logs:")
    for path in log_paths:
        print(f"- {path.name}")

    if list_only:
        return

    primary_log = logs_dir / "control-plane.log"
    if not primary_log.exists():
        print("Primary log 'control-plane.log' does not exist yet.")
        return

    print(f"\nTail of {primary_log.name}:")
    lines = read_tail(primary_log, tail)
    if errors_only:
        lines = [
            line
            for line in lines
            if any(level in line for level in (" ERROR ", " WARNING ", " CRITICAL "))
        ]
    if not lines:
        print("(no matching log lines)")
        return
    for line in lines:
        print(line)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local development harness for MLXR")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("fix", help="Run formatting and auto-fix linting")
    subparsers.add_parser("lint", help="Run ruff lint checks")
    subparsers.add_parser("typecheck", help="Run strict mypy checks")
    subparsers.add_parser("test", help="Run the unittest suite")
    subparsers.add_parser("build", help="Build all workspace packages")
    subparsers.add_parser("verify", help="Run the full local quality gate")

    logs_parser = subparsers.add_parser("logs", help="Inspect runtime logs")
    logs_parser.add_argument(
        "--list", action="store_true", help="Only list available logs"
    )
    logs_parser.add_argument(
        "--tail",
        type=int,
        default=80,
        help="Number of lines to show from the control-plane log",
    )
    logs_parser.add_argument(
        "--errors-only",
        action="store_true",
        help="Show only warning/error/critical lines",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "fix":
        cmd_fix()
    elif args.command == "lint":
        cmd_lint()
    elif args.command == "typecheck":
        cmd_typecheck()
    elif args.command == "test":
        cmd_test()
    elif args.command == "build":
        cmd_build()
    elif args.command == "verify":
        cmd_verify()
    elif args.command == "logs":
        cmd_logs(list_only=args.list, tail=args.tail, errors_only=args.errors_only)
    else:
        raise ValueError(f"Unknown command '{args.command}'")


if __name__ == "__main__":
    main()
