from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_TEST_DIR = REPO_ROOT / "tests"


def _ensure_repo_on_path() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _package_test_dirs() -> list[Path]:
    return sorted(
        path for path in REPO_ROOT.glob("packages/*/*/tests") if path.is_dir()
    )


def _load_module_from_path(path: Path) -> ModuleType:
    relative = path.relative_to(REPO_ROOT).with_suffix("")
    module_name = "__mlxr_tests__." + "_".join(relative.parts)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_suite() -> unittest.TestSuite:
    _ensure_repo_on_path()
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(
        loader.discover(
            start_dir=str(ROOT_TEST_DIR),
            pattern="test*.py",
            top_level_dir=str(REPO_ROOT),
        )
    )
    for test_root in _package_test_dirs():
        for test_path in sorted(test_root.rglob("test*.py")):
            module = _load_module_from_path(test_path)
            suite.addTests(loader.loadTestsFromModule(module))
    return suite


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MLXR unittest suites from both repo-level and package-local test roots."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase test runner verbosity.",
    )
    args = parser.parse_args()

    verbosity = 2 if args.verbose else 1
    suite = build_suite()
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
