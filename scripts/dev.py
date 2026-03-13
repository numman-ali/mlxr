from __future__ import annotations

import argparse
import os
import plistlib
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

TARGETS = ("packages", "tests", "scripts")
REPO_ROOT = Path(__file__).resolve().parent.parent
MAC_APP_PACKAGE_DIR = REPO_ROOT / "packages" / "clients" / "mlxr-mac-app"
MAC_APP_NAME = "MLXRMacApp"
MAC_APP_BUNDLE_ID = "com.numman-ali.mlxr.dev"


def runtime_home() -> Path:
    raw_root = os.environ.get("MLX_RUNTIME_HOME", str(Path.home() / ".mlx-runtime"))
    return Path(raw_root).expanduser()


def run_command(command: Sequence[str]) -> None:
    print(f"+ {' '.join(command)}")
    subprocess.run(command, check=True)


def capture_command(command: Sequence[str]) -> str:
    print(f"+ {' '.join(command)}")
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, cwd=REPO_ROOT
    )
    return result.stdout.strip()


def read_tail(path: Path, lines: int) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()[-lines:]


def cmd_fix() -> None:
    run_command(["ruff", "format", *TARGETS])
    run_command(["ruff", "check", "--fix", *TARGETS])


def cmd_lint() -> None:
    run_command(["ruff", "check", *TARGETS])


def cmd_typecheck() -> None:
    run_command([sys.executable, "scripts/run_mypy.py"])


def cmd_test() -> None:
    run_command([sys.executable, "scripts/run_unittests.py", "-v"])


def cmd_build() -> None:
    run_command(["uv", "build", "--all-packages"])


def cmd_verify() -> None:
    run_command(["ruff", "format", "--check", *TARGETS])
    run_command(["ruff", "check", *TARGETS])
    run_command([sys.executable, "scripts/run_mypy.py"])
    run_command([sys.executable, "scripts/check_type_escapes.py"])
    Path("tmp").mkdir(parents=True, exist_ok=True)
    run_command(["coverage", "erase"])
    run_command(["coverage", "run", "scripts/run_unittests.py", "-v"])
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


def cmd_mac_app(
    *,
    launch: bool,
    quit_existing: bool,
    bundle_dir: str | None,
    runtime_home_override: str | None,
    python_override: str | None,
    uds_path_override: str | None,
    print_path: bool,
) -> None:
    package_dir = MAC_APP_PACKAGE_DIR
    if not package_dir.exists():
        raise FileNotFoundError(f"Mac app package not found at {package_dir}")

    run_command(["swift", "build", "--package-path", str(package_dir)])
    bin_path = Path(
        capture_command(
            ["swift", "build", "--package-path", str(package_dir), "--show-bin-path"]
        )
    )
    built_executable = bin_path / MAC_APP_NAME
    if not built_executable.exists():
        raise FileNotFoundError(
            f"Built executable was not found at {built_executable}"
        )

    bundle_root = (
        Path(bundle_dir).expanduser()
        if bundle_dir is not None
        else package_dir / ".build" / "dev-app"
    )
    app_bundle = bundle_root / f"{MAC_APP_NAME}.app"
    contents_dir = app_bundle / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_dir = contents_dir / "Resources"
    launcher_path = macos_dir / MAC_APP_NAME
    bundled_binary = resources_dir / f"{MAC_APP_NAME}-bin"

    if app_bundle.exists():
        shutil.rmtree(app_bundle)
    resources_dir.mkdir(parents=True, exist_ok=True)
    macos_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(built_executable, bundled_binary)
    bundled_binary.chmod(0o755)

    runtime_home_value = (
        runtime_home_override
        or os.environ.get("MLXR_MAC_APP_RUNTIME_HOME")
        or os.environ.get("MLX_RUNTIME_HOME")
        or str(Path.home() / ".mlx-runtime")
    )
    python_value = (
        python_override
        or os.environ.get("MLXR_MAC_APP_PYTHON")
        or str(REPO_ROOT / ".venv" / "bin" / "python")
    )
    uds_path_value = (
        uds_path_override or os.environ.get("MLXR_MAC_APP_UDS_PATH") or ""
    )

    launcher_script = _mac_app_launcher_script(
        repo_root=str(REPO_ROOT),
        python_path=python_value,
        runtime_home=runtime_home_value,
        uds_path=uds_path_value,
        bundled_binary=bundled_binary,
    )
    launcher_path.write_text(launcher_script, encoding="utf-8")
    launcher_path.chmod(0o755)

    info_plist = {
        "CFBundleDevelopmentRegion": "en",
        "CFBundleExecutable": MAC_APP_NAME,
        "CFBundleIdentifier": MAC_APP_BUNDLE_ID,
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleName": MAC_APP_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "0.1-dev",
        "CFBundleVersion": capture_command(["git", "rev-parse", "--short", "HEAD"])
        or "dev",
        "LSMinimumSystemVersion": "15.0",
        "NSHighResolutionCapable": True,
        "NSPrincipalClass": "NSApplication",
    }
    with (contents_dir / "Info.plist").open("wb") as handle:
        plistlib.dump(info_plist, handle, sort_keys=False)
    (contents_dir / "PkgInfo").write_text("APPL????", encoding="ascii")

    if quit_existing:
        subprocess.run(
            ["pkill", "-f", str(launcher_path)],
            check=False,
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["pkill", "-f", str(built_executable)],
            check=False,
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    if launch:
        run_command(["open", "-na", str(app_bundle)])

    if print_path or not launch:
        print(app_bundle)

    print(f"Dev app bundle ready at {app_bundle}")
    print("Peekaboo:")
    print(f"  peekaboo app switch --to {MAC_APP_NAME}")
    print(f"  peekaboo see --app {MAC_APP_NAME} --json --annotate")


def _mac_app_launcher_script(
    *,
    repo_root: str,
    python_path: str,
    runtime_home: str,
    uds_path: str,
    bundled_binary: Path,
) -> str:
    lines = [
        "#!/bin/zsh",
        "set -euo pipefail",
        f"export MLXR_MAC_APP_REPO_ROOT={shlex.quote(repo_root)}",
        f"export MLXR_MAC_APP_PYTHON={shlex.quote(python_path)}",
        f"export MLXR_MAC_APP_RUNTIME_HOME={shlex.quote(runtime_home)}",
        f"export MLX_RUNTIME_HOME={shlex.quote(runtime_home)}",
    ]
    if uds_path:
        quoted_uds = shlex.quote(uds_path)
        lines.append(f"export MLXR_MAC_APP_UDS_PATH={quoted_uds}")
        lines.append(f"export MLX_RUNTIME_UDS_PATH={quoted_uds}")
    lines.append(f"exec {shlex.quote(str(bundled_binary))}")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local development harness for MLXR")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("fix", help="Run formatting and auto-fix linting")
    subparsers.add_parser("lint", help="Run ruff lint checks")
    subparsers.add_parser("typecheck", help="Run strict mypy checks")
    subparsers.add_parser("test", help="Run the unittest suite")
    subparsers.add_parser("build", help="Build all workspace packages")
    subparsers.add_parser("verify", help="Run the full local quality gate")

    mac_app_parser = subparsers.add_parser(
        "mac-app",
        help="Build a dev MLXRMacApp.app bundle and optionally launch it",
    )
    mac_app_parser.add_argument(
        "--launch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch the bundled app after building it",
    )
    mac_app_parser.add_argument(
        "--quit-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Kill existing dev MLXRMacApp processes before launch",
    )
    mac_app_parser.add_argument(
        "--bundle-dir",
        help="Override the directory where the dev .app bundle is staged",
    )
    mac_app_parser.add_argument(
        "--runtime-home",
        help="Embed a specific runtime home into the dev app launcher",
    )
    mac_app_parser.add_argument(
        "--python",
        help="Embed a specific Python executable into the dev app launcher",
    )
    mac_app_parser.add_argument(
        "--uds-path",
        help="Embed a specific UDS socket path into the dev app launcher",
    )
    mac_app_parser.add_argument(
        "--print-path",
        action="store_true",
        help="Print the staged .app bundle path after building",
    )

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
    elif args.command == "mac-app":
        cmd_mac_app(
            launch=args.launch,
            quit_existing=args.quit_existing,
            bundle_dir=args.bundle_dir,
            runtime_home_override=args.runtime_home,
            python_override=args.python,
            uds_path_override=args.uds_path,
            print_path=args.print_path,
        )
    elif args.command == "logs":
        cmd_logs(list_only=args.list, tail=args.tail, errors_only=args.errors_only)
    else:
        raise ValueError(f"Unknown command '{args.command}'")


if __name__ == "__main__":
    main()
