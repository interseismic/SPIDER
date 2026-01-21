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
Clean common local artifacts from a SPIDER worktree (safe, opt-in).

This is intended for developer hygiene when the repository directory collects:
- wandb logs
- plots / checkpoints
- build artifacts like *.egg-info
- Python caches like __pycache__

By default this runs in --dry-run mode.
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Target:
    relpath: str
    kind: str  # "dir" or "glob"


DEFAULT_TARGETS: list[Target] = [
    Target("wandb", "dir"),
    Target("plots", "dir"),
    Target("checkpoints", "dir"),
    Target("spider.egg-info", "dir"),
    Target("__pycache__", "dir"),
    Target("None", "dir"),
    # Recursive python caches
    Target("**/__pycache__", "glob"),
    Target("**/*.pyc", "glob"),
    Target("**/*.pyo", "glob"),
]


def _rm_dir(p: Path, *, dry_run: bool) -> None:
    if not p.exists():
        return
    if not p.is_dir():
        return
    if dry_run:
        _log(f"[dry-run] rmdir -r {p}")
        return
    shutil.rmtree(p)
    _log(f"deleted dir {p}")


def _rm_file(p: Path, *, dry_run: bool) -> None:
    if not p.exists():
        return
    if p.is_dir():
        return
    if dry_run:
        _log(f"[dry-run] rm {p}")
        return
    p.unlink()
    _log(f"deleted file {p}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Remove common local artifacts from a SPIDER worktree")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Worktree root to clean (default: repository root inferred from this file location)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete files/dirs (default: dry-run only)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive glob targets (**/__pycache__, **/*.pyc, etc.)",
    )
    args = parser.parse_args(argv)

    # Infer repo root: spider/tools/clean_worktree.py -> repo root is ../../
    here = Path(__file__).resolve()
    repo_root = Path(args.root).expanduser().resolve() if args.root else here.parents[2]

    dry_run = not bool(args.yes)
    _log(f"[clean] root={repo_root} mode={'dry-run' if dry_run else 'DELETE'}")

    targets = DEFAULT_TARGETS
    if args.no_recursive:
        targets = [t for t in targets if not (t.kind == "glob" and t.relpath.startswith("**/"))]

    # First remove explicit directories (so later globs don't traverse them)
    for t in targets:
        if t.kind != "dir":
            continue
        _rm_dir(repo_root / t.relpath, dry_run=dry_run)

    # Then remove globs (files + dirs)
    for t in targets:
        if t.kind != "glob":
            continue
        for hit in repo_root.glob(t.relpath):
            # Avoid deleting outside root via weird symlinks
            try:
                hit.resolve().relative_to(repo_root.resolve())
            except Exception:
                continue
            if hit.is_dir():
                _rm_dir(hit, dry_run=dry_run)
            else:
                _rm_file(hit, dry_run=dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

