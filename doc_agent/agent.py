"""DocAgent: Uses Claude to generate and maintain API documentation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import anthropic

from .extractor import DocExtractor, ModuleInfo

# ─── Tool definitions ────────────────────────────────────────────────────────

READ_FILE_TOOL: dict[str, Any] = {
    "name": "read_file",
    "description": (
        "Read the full contents of a source file from disk. "
        "Use this to inspect implementation details before generating docs."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file to read.",
            }
        },
        "required": ["path"],
    },
}

LIST_FILES_TOOL: dict[str, Any] = {
    "name": "list_files",
    "description": (
        "List files in a directory that match a glob pattern. "
        "Use this to discover Python source files before processing."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "dir": {
                "type": "string",
                "description": "Directory path to search in.",
            },
            "pattern": {
                "type": "string",
                "description": "Glob pattern to filter files, e.g. '**/*.py'.",
                "default": "**/*.py",
            },
        },
        "required": ["dir"],
    },
}

WRITE_FILE_TOOL: dict[str, Any] = {
    "name": "write_file",
    "description": (
        "Write content to a file, creating parent directories as needed. "
        "Use this to save generated documentation to disk."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to write.",
            },
            "content": {
                "type": "string",
                "description": "Text content to write to the file.",
            },
        },
        "required": ["path", "content"],
    },
}

TOOLS = [READ_FILE_TOOL, LIST_FILES_TOOL, WRITE_FILE_TOOL]


# ─── Tool execution ───────────────────────────────────────────────────────────


def _run_read_file(path: str) -> str:
    """Read a file from disk and return its contents."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"ERROR: File not found: {path}"
    except PermissionError:
        return f"ERROR: Permission denied reading: {path}"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


def _run_list_files(dir: str, pattern: str = "**/*.py") -> str:
    """List files matching a glob pattern within a directory."""
    base = Path(dir)
    if not base.exists():
        return f"ERROR: Directory not found: {dir}"
    matches = sorted(str(p) for p in base.glob(pattern) if p.is_file())
    if not matches:
        return "No files found."
    return "\n".join(matches)


def _run_write_file(path: str, content: str) -> str:
    """Write content to a file, creating directories as needed."""
    try:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        return f"OK: wrote {len(content)} chars to {path}"
    except PermissionError:
        return f"ERROR: Permission denied writing: {path}"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


def _dispatch_tool(name: str, inputs: dict[str, Any]) -> str:
    """Route a tool call to the appropriate implementation."""
    if name == "read_file":
        return _run_read_file(inputs["path"])
    if name == "list_files":
        return _run_list_files(inputs["dir"], inputs.get("pattern", "**/*.py"))
    if name == "write_file":
        return _run_write_file(inputs["path"], inputs["content"])
    return f"ERROR: Unknown tool: {name}"


# ─── DocAgent ─────────────────────────────────────────────────────────────────


class DocAgent:
    """Agentic documentation generator powered by Claude.

    Uses an agentic loop with three tools (read_file, list_files, write_file)
    to inspect Python source code and produce Markdown or RST documentation
    that stays in sync with the public API.

    Parameters
    ----------
    api_key:
        Anthropic API key. Falls back to the ``ANTHROPIC_API_KEY`` environment
        variable when *None*.
    model:
        Claude model to use for generation. Defaults to ``claude-sonnet-4-6``.
    """

    MODEL = "claude-sonnet-4-6"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = MODEL,
    ) -> None:
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = model
        self._extractor = DocExtractor()

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        src_dir: str | Path,
        output_dir: str | Path,
        fmt: str = "markdown",
    ) -> list[Path]:
        """Generate documentation for all Python files under *src_dir*.

        For each ``.py`` file found the agent:

        1. Extracts class/function signatures and docstrings via AST.
        2. Sends the structured API info plus source context to Claude.
        3. Writes the produced documentation to *output_dir*.

        Parameters
        ----------
        src_dir:
            Root directory containing Python source files.
        output_dir:
            Directory where documentation files will be written.
        fmt:
            Output format — ``"markdown"`` (default) or ``"rst"``.

        Returns
        -------
        list[Path]
            Paths of documentation files that were written.
        """
        src_dir = Path(src_dir).resolve()
        output_dir = Path(output_dir).resolve()
        fmt = fmt.lower()

        py_files = sorted(src_dir.rglob("*.py"))
        written: list[Path] = []

        for py_file in py_files:
            if py_file.name.startswith("_") and py_file.name != "__init__.py":
                continue
            doc_path = self._doc_path(py_file, src_dir, output_dir, fmt)
            content = self._generate_for_file(py_file, fmt)
            if content:
                doc_path.parent.mkdir(parents=True, exist_ok=True)
                doc_path.write_text(content, encoding="utf-8")
                written.append(doc_path)

        return written

    def check(
        self,
        src_dir: str | Path,
        output_dir: str | Path,
        fmt: str = "markdown",
    ) -> list[tuple[Path, str]]:
        """Check whether existing documentation is up to date with the source.

        Generates fresh documentation in memory and compares it with what is
        already on disk.  Returns a list of ``(doc_path, reason)`` tuples for
        any file that has drifted.

        Parameters
        ----------
        src_dir:
            Root directory containing Python source files.
        output_dir:
            Directory containing existing documentation files.
        fmt:
            Documentation format to check against.

        Returns
        -------
        list[tuple[Path, str]]
            Pairs of (path, drift-reason) for files that are out of date.
            An empty list means everything is in sync.
        """
        src_dir = Path(src_dir).resolve()
        output_dir = Path(output_dir).resolve()
        fmt = fmt.lower()

        py_files = sorted(src_dir.rglob("*.py"))
        drifted: list[tuple[Path, str]] = []

        for py_file in py_files:
            if py_file.name.startswith("_") and py_file.name != "__init__.py":
                continue
            doc_path = self._doc_path(py_file, src_dir, output_dir, fmt)

            fresh = self._generate_for_file(py_file, fmt)
            if not fresh:
                continue

            if not doc_path.exists():
                drifted.append((doc_path, "documentation file missing"))
                continue

            existing = doc_path.read_text(encoding="utf-8")
            if existing.strip() != fresh.strip():
                drifted.append((doc_path, "documentation is out of sync with source"))

        return drifted

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _doc_path(
        self,
        py_file: Path,
        src_dir: Path,
        output_dir: Path,
        fmt: str,
    ) -> Path:
        """Compute the destination documentation file path."""
        rel = py_file.relative_to(src_dir).with_suffix(".md" if fmt == "markdown" else ".rst")
        return output_dir / rel

    def _generate_for_file(self, py_file: Path, fmt: str) -> str:
        """Run the agentic loop for a single Python file and return doc content."""
        info: ModuleInfo = self._extractor.extract(py_file)
        if not info.classes and not info.functions:
            return ""

        system_prompt = self._build_system_prompt(fmt)
        user_message = self._build_user_message(py_file, info, fmt)

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_message}
        ]

        # Agentic loop — keep going until Claude stops requesting tool calls
        for _ in range(20):  # safety cap
            with self._client.messages.stream(
                model=self.model,
                max_tokens=8192,
                system=system_prompt,
                tools=TOOLS,  # type: ignore[arg-type]
                messages=messages,  # type: ignore[arg-type]
            ) as stream:
                response = stream.get_final_message()

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                # Extract the final text block as the documentation
                for block in response.content:
                    if hasattr(block, "type") and block.type == "text":
                        return block.text
                return ""

            if response.stop_reason != "tool_use":
                break

            # Execute all requested tool calls
            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if not (hasattr(block, "type") and block.type == "tool_use"):
                    continue
                result = _dispatch_tool(block.name, block.input)  # type: ignore[arg-type]
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )

            messages.append({"role": "user", "content": tool_results})

        return ""

    @staticmethod
    def _build_system_prompt(fmt: str) -> str:
        ext_name = "Markdown" if fmt == "markdown" else "reStructuredText (RST)"
        return (
            f"You are an expert technical writer who generates {ext_name} API "
            "documentation for Python libraries.\n\n"
            "You will be given structured information extracted from a Python source "
            "file — classes, methods, functions, their signatures, and docstrings. "
            "You also have tools to read additional source files if you need more context.\n\n"
            "Your task:\n"
            f"1. Write clear, accurate {ext_name} documentation for every public API "
            "element (classes, methods, functions).\n"
            "2. Include the full signature for each item.\n"
            "3. Include a usage example (in a code block) for each class and "
            "standalone function.\n"
            "4. Preserve and expand on any existing docstrings.\n"
            "5. Do NOT document private items (names starting with '_').\n\n"
            "Output ONLY the documentation content — no preamble, no explanation."
        )

    @staticmethod
    def _build_user_message(py_file: Path, info: ModuleInfo, fmt: str) -> str:
        api_summary = json.dumps(
            {
                "module": info.module_name,
                "docstring": info.docstring,
                "classes": [
                    {
                        "name": cls.name,
                        "docstring": cls.docstring,
                        "bases": cls.bases,
                        "methods": [
                            {
                                "name": m.name,
                                "signature": m.signature,
                                "docstring": m.docstring,
                                "decorators": m.decorators,
                            }
                            for m in cls.methods
                        ],
                    }
                    for cls in info.classes
                ],
                "functions": [
                    {
                        "name": fn.name,
                        "signature": fn.signature,
                        "docstring": fn.docstring,
                        "decorators": fn.decorators,
                    }
                    for fn in info.functions
                ],
            },
            indent=2,
        )

        ext = "md" if fmt == "markdown" else "rst"
        return (
            f"Generate {fmt} documentation for the Python module at `{py_file}`.\n\n"
            f"Here is the extracted public API (JSON):\n```json\n{api_summary}\n```\n\n"
            "You can call `read_file` with the source path above if you need to inspect "
            "implementation details or understand usage patterns.\n\n"
            f"Write the documentation as a self-contained `{ext}` file."
        )
