from __future__ import annotations

from textwrap import dedent

from .constants import DOCS_URL, FEEDBACK_URL


def _top_level_brief_help() -> str:
    return dedent(
        f"""\
        MLXR is a local-first MLX runtime for Apple Silicon.

        Start here:
          mlxr models list
          mlxr models install ltx-2.3-fast-local
          mlxr generate --model-id ltx-2.3-fast-local --prompt "golden retriever in a park" --wait --export-path out.mp4

        For full help, run: mlxr --help
        Docs: {DOCS_URL}
        Feedback: {FEEDBACK_URL}
        """
    )


def _top_level_help_epilog() -> str:
    return dedent(
        f"""\
        Examples:
          mlxr models list
          mlxr models install ltx-2.3-fast-local
          mlxr generate --model-id ltx-2.3-fast-local --prompt "golden retriever in a park" --wait --export-path out.mp4

        Docs: {DOCS_URL}
        Feedback: {FEEDBACK_URL}
        """
    )


def _generate_help_epilog() -> str:
    return dedent(
        f"""\
        Examples:
          mlxr generate --model-id ltx-2.3-fast-local --prompt "golden retriever in a park" --wait --export-path out.mp4
          mlxr generate --model-id flux2-klein-9b-local --prompt "cinematic portrait" --artifact-format png --wait --export-path out.png

        Docs: {DOCS_URL}
        """
    )
