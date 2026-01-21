from spider.utils.console import info, warn


# Standardized stdout helper
def _log(*parts, section: str = "TOOLS", **_kwargs) -> None:
    msg = " ".join(str(p) for p in parts)
    low = msg.strip().lower()
    if low.startswith("warning") or low.startswith("error"):
        warn(msg, section=section)
    else:
        info(msg, section=section)

"""
Static reachability audit for the `spider` package.

Goal
----
Help maintainers spot "stale" modules that are not reachable from the primary
entrypoints (CLI + package import) via *import edges*.

This is intentionally conservative:
- It only understands `import ...` and `from ... import ...` statements.
- It does NOT evaluate runtime imports (importlib, __import__, plugin patterns).
- "Unreachable" here means: not imported (directly or transitively) from the
  chosen roots. Such modules may still be used via dynamic imports or notebooks.

Usage
-----
  python -m spider.tools.audit_reachability
  python -m spider.tools.audit_reachability --roots spider.cli spider.__main__
"""

from __future__ import annotations

import argparse
import ast
import os
from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Sequence


@dataclass(frozen=True)
class ModuleInfo:
    module: str
    path: str


def _iter_py_modules(package_dir: str, package_name: str) -> Iterator[ModuleInfo]:
    for root, _dirs, files in os.walk(package_dir):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, package_dir)
            mod = rel[:-3].replace(os.sep, ".")  # strip .py
            # Canonicalize package `__init__.py` to the package name.
            # - spider/__init__.py          -> spider
            # - spider/core/__init__.py     -> spider.core
            if mod == "__init__":
                mod = ""
            if mod.endswith(".__init__"):
                mod = mod[: -len(".__init__")]
            module = f"{package_name}.{mod}" if mod else package_name
            yield ModuleInfo(module=module, path=path)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _imports_from_ast(tree: ast.AST, *, current_module: str, is_package: bool) -> set[str]:
    """
    Return a set of imported *module names* as strings.

    Notes:
    - For `from X import Y`, we include both `X` and `X.Y` as potential edges,
      because either might resolve to a module within the package.
    - For relative imports, we resolve them against current_module.
    """
    out: set[str] = set()

    def _resolve_relative(module: str | None, level: int) -> str | None:
        if level <= 0:
            return module
        parts = current_module.split(".")
        # For regular modules, relative imports are resolved from the *containing package*.
        # For package __init__.py modules, they are resolved from the package itself.
        base_parts = parts if is_package else parts[:-1]
        if not base_parts:
            return None
        # In Python, level=1 means "current package", level=2 means "parent", etc.
        keep = len(base_parts) - (level - 1)
        if keep <= 0:
            return None
        prefix = ".".join(base_parts[:keep])
        if module:
            return f"{prefix}.{module}"
        return prefix or None

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name:
                    out.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            mod = _resolve_relative(node.module, int(node.level or 0))
            if mod:
                out.add(mod)
                for a in node.names:
                    if a.name and a.name != "*":
                        out.add(f"{mod}.{a.name}")
    return out


def _build_import_graph(mods: Sequence[ModuleInfo]) -> dict[str, set[str]]:
    by_module: dict[str, ModuleInfo] = {m.module: m for m in mods}
    graph: dict[str, set[str]] = {m.module: set() for m in mods}

    for m in mods:
        try:
            tree = ast.parse(_read_text(m.path), filename=m.path)
        except SyntaxError:
            # Skip files that aren't parseable in this environment; they are still "present".
            continue
        imports = _imports_from_ast(tree, current_module=m.module, is_package=os.path.basename(m.path) == "__init__.py")
        # Keep only edges into known modules (within this package snapshot)
        for imp in imports:
            if imp in by_module:
                graph[m.module].add(imp)
            else:
                # Also accept "package imports" for modules that correspond to a package __init__.
                # E.g. importing "spider.core" should mark "spider.core" (package) reachable.
                if imp in graph:
                    graph[m.module].add(imp)
    return graph


def _reachable(graph: Mapping[str, set[str]], roots: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    stack = [r for r in roots if r in graph]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for nxt in graph.get(cur, set()):
            if nxt not in seen:
                stack.append(nxt)
    return seen


def _promote_parent_packages(mods: Iterable[str]) -> set[str]:
    """
    Given a set of reachable modules, also mark their parent packages as reachable.

    This reduces false positives where a submodule is imported (e.g. spider.optim.sgld)
    but the package `spider.optim` isn't explicitly imported anywhere.
    """
    out: set[str] = set(mods)
    for m in list(out):
        parts = m.split(".")
        # Keep promoting `a.b.c` -> `a.b` -> `a`
        for k in range(len(parts) - 1, 0, -1):
            out.add(".".join(parts[:k]))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Static import reachability audit for the spider package")
    parser.add_argument(
        "--package-dir",
        default=os.path.join(os.path.dirname(__file__), ".."),
        help="Filesystem path to the spider package directory (default: parent of this file)",
    )
    parser.add_argument(
        "--package-name",
        default="spider",
        help="Top-level package name (default: spider)",
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["spider.__main__", "spider.cli", "spider"],
        help="Root modules to treat as entrypoints (default: spider.__main__ spider.cli spider)",
    )
    parser.add_argument(
        "--ignore-prefix",
        nargs="*",
        default=["spider.tools"],
        help="Module prefixes to exclude from the unreachable list (default: spider.tools)",
    )
    args = parser.parse_args(argv)

    package_dir = os.path.abspath(args.package_dir)
    mods = list(_iter_py_modules(package_dir, args.package_name))
    graph = _build_import_graph(mods)
    reach = _promote_parent_packages(_reachable(graph, args.roots))

    ignore_prefixes = tuple(str(x).strip() for x in (args.ignore_prefix or []) if str(x).strip())

    unreachable = []
    for m in sorted(graph.keys()):
        if m in reach:
            continue
        if ignore_prefixes and m.startswith(ignore_prefixes):
            continue
        unreachable.append(m)

    _log(f"[audit] package_dir={package_dir}")
    _log(f"[audit] modules_total={len(graph)} roots={args.roots} reachable={len(reach)} unreachable={len(unreachable)}")
    if unreachable:
        _log("\n[audit] Unreachable modules (by static import graph):")
        for m in unreachable:
            _log(f"- {m}")
    else:
        _log("\n[audit] No unreachable modules detected (from chosen roots).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

