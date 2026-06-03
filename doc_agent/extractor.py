"""DocExtractor: Parse Python source files via AST to extract public API info."""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class FunctionInfo:
    """Information about a function or method extracted from the AST."""

    name: str
    signature: str
    docstring: str
    decorators: list[str] = field(default_factory=list)


@dataclass
class ClassInfo:
    """Information about a class extracted from the AST."""

    name: str
    docstring: str
    bases: list[str] = field(default_factory=list)
    methods: list[FunctionInfo] = field(default_factory=list)


@dataclass
class ModuleInfo:
    """Information about a Python module extracted from the AST."""

    module_name: str
    docstring: str
    classes: list[ClassInfo] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)


class DocExtractor:
    """Extract public API information from Python source files using the AST.

    Only public symbols are extracted — names beginning with an underscore are
    skipped, with the exception of ``__init__`` and ``__call__`` which are
    included because they are part of the observable interface.

    Example
    -------
    >>> extractor = DocExtractor()
    >>> info = extractor.extract("mypackage/core.py")
    >>> for cls in info.classes:
    ...     print(cls.name, [m.name for m in cls.methods])
    """

    #: Methods to include even though they start with underscores.
    INCLUDE_DUNDER = {"__init__", "__call__", "__enter__", "__exit__",
                     "__repr__", "__str__", "__len__", "__iter__",
                     "__next__", "__getitem__", "__setitem__", "__delitem__",
                     "__contains__", "__eq__", "__hash__"}

    def extract(self, path: str | Path) -> ModuleInfo:
        """Parse *path* and return a :class:`ModuleInfo` for the module.

        Parameters
        ----------
        path:
            Path to a Python ``.py`` file.

        Returns
        -------
        ModuleInfo
            Structured API information suitable for documentation generation.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        SyntaxError
            If *path* contains invalid Python syntax.
        """
        path = Path(path)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))

        module_name = path.stem
        module_doc = ast.get_docstring(tree) or ""

        classes: list[ClassInfo] = []
        functions: list[FunctionInfo] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.ClassDef,)):
                cls_info = self._extract_class(node, source)
                if cls_info is not None:
                    classes.append(cls_info)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_info = self._extract_function(node, source)
                if fn_info is not None:
                    functions.append(fn_info)

        return ModuleInfo(
            module_name=module_name,
            docstring=module_doc,
            classes=classes,
            functions=functions,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _is_public(self, name: str) -> bool:
        """Return True if *name* should be included in the public API."""
        return not name.startswith("_") or name in self.INCLUDE_DUNDER

    def _extract_class(self, node: ast.ClassDef, source: str) -> ClassInfo | None:
        """Extract a :class:`ClassInfo` from a class AST node."""
        if not self._is_public(node.name):
            return None

        docstring = ast.get_docstring(node) or ""
        bases = [self._unparse(base) for base in node.bases]

        methods: list[FunctionInfo] = []
        for item in ast.iter_child_nodes(node):
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn = self._extract_function(item, source)
                if fn is not None:
                    methods.append(fn)

        return ClassInfo(
            name=node.name,
            docstring=docstring,
            bases=bases,
            methods=methods,
        )

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        source: str,
    ) -> FunctionInfo | None:
        """Extract a :class:`FunctionInfo` from a function/method AST node."""
        if not self._is_public(node.name):
            return None

        docstring = ast.get_docstring(node) or ""
        signature = self._build_signature(node)
        decorators = [self._unparse(d) for d in node.decorator_list]

        return FunctionInfo(
            name=node.name,
            signature=signature,
            docstring=docstring,
            decorators=decorators,
        )

    def _build_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """Build a human-readable function signature from an AST node."""
        args = node.args
        parts: list[str] = []

        # positional-only args (before /)
        pos_only = args.posonlyargs
        regular = args.args
        defaults_offset = len(regular) - len(args.defaults)
        pos_only_offset = len(pos_only) - max(
            0, len(args.defaults) - len(regular)
        )

        for i, arg in enumerate(pos_only):
            default_index = i - pos_only_offset
            part = self._arg_str(arg)
            if default_index >= 0 and default_index < len(args.defaults) - len(regular):
                part += f" = {self._unparse(args.defaults[default_index])}"
            parts.append(part)
        if pos_only:
            parts.append("/")

        for i, arg in enumerate(regular):
            default_index = i - defaults_offset
            part = self._arg_str(arg)
            if default_index >= 0:
                part += f" = {self._unparse(args.defaults[default_index])}"
            parts.append(part)

        if args.vararg:
            parts.append(f"*{self._arg_str(args.vararg)}")
        elif args.kwonlyargs:
            parts.append("*")

        for i, arg in enumerate(args.kwonlyargs):
            kw_default = args.kw_defaults[i]
            part = self._arg_str(arg)
            if kw_default is not None:
                part += f" = {self._unparse(kw_default)}"
            parts.append(part)

        if args.kwarg:
            parts.append(f"**{self._arg_str(args.kwarg)}")

        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        ret = ""
        if node.returns is not None:
            ret = f" -> {self._unparse(node.returns)}"

        return f"{prefix} {node.name}({', '.join(parts)}){ret}:"

    @staticmethod
    def _arg_str(arg: ast.arg) -> str:
        """Format a single argument with optional type annotation."""
        if arg.annotation:
            return f"{arg.arg}: {DocExtractor._unparse(arg.annotation)}"
        return arg.arg

    @staticmethod
    def _unparse(node: ast.AST) -> str:
        """Convert an AST expression node back to source text."""
        return ast.unparse(node)
