"""Forbid broad type escapes in agent-facing runtime and test surfaces."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

TARGET_ROOTS = (
    Path("packages/core/runtime-workflows"),
    Path("packages/core/runtime-server"),
    Path("packages/families/flux2"),
    Path("packages/families/ltx"),
    Path("packages/families/qwen-image"),
    Path("packages/families/z-image"),
    Path("packages/clients/runtime-cli"),
    Path("tests"),
)


@dataclass(frozen=True, slots=True)
class Violation:
    path: Path
    line: int
    column: int
    message: str


class TypeEscapeVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.typing_aliases: set[str] = set()
        self.any_aliases: set[str] = set()
        self.cast_aliases: set[str] = set()
        self.violations: list[tuple[int, int, str]] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "typing":
                self.typing_aliases.add(alias.asname or "typing")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module in {"typing", "typing_extensions"}:
            for alias in node.names:
                bound_name = alias.asname or alias.name
                if alias.name == "Any":
                    self.any_aliases.add(bound_name)
                elif alias.name == "cast":
                    self.cast_aliases.add(bound_name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in self.cast_aliases:
            self.violations.append(
                (
                    node.lineno,
                    node.col_offset,
                    "Avoid typing.cast; fix the seam instead",
                )
            )
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in self.typing_aliases
            and node.func.attr == "cast"
        ):
            self.violations.append(
                (
                    node.lineno,
                    node.col_offset,
                    "Avoid typing.cast; fix the seam instead",
                )
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id in self.any_aliases:
            self.violations.append(
                (
                    node.lineno,
                    node.col_offset,
                    "Avoid explicit Any in agent-facing runtime code",
                )
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.value.id in self.typing_aliases
            and node.attr == "Any"
        ):
            self.violations.append(
                (
                    node.lineno,
                    node.col_offset,
                    "Avoid explicit Any in agent-facing runtime code",
                )
            )
        self.generic_visit(node)


def collect_python_files() -> list[Path]:
    files: list[Path] = []
    for root in TARGET_ROOTS:
        files.extend(sorted(path for path in root.rglob("*.py") if path.is_file()))
    return files


def find_violations(path: Path) -> list[Violation]:
    visitor = TypeEscapeVisitor()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    visitor.visit(tree)
    return [
        Violation(path=path, line=line, column=column, message=message)
        for line, column, message in visitor.violations
    ]


def main() -> int:
    violations: list[Violation] = []
    for path in collect_python_files():
        violations.extend(find_violations(path))

    if not violations:
        print("No forbidden type escapes found.")
        return 0

    for violation in violations:
        print(
            f"{violation.path}:{violation.line}:{violation.column + 1}: {violation.message}"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
