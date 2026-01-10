"""
Lightweight, consistent console logging for SPIDER.

Goal: make terminal output easy to scan and grep.
Format:
  [spider][LEVEL][SECTION] message

We intentionally do not depend on the Python `logging` module here to keep callsites
simple and avoid configuration surprises in CLI environments.
"""

from __future__ import annotations

from typing import Optional
import os


def _fmt(level: str, section: str, msg: str) -> str:
    lvl = (level or "INFO").strip().upper()
    sec = (section or "GEN").strip().upper()
    return f"[spider][{lvl}][{sec}] {msg}"


def _is_main_rank() -> bool:
    """
    Best-effort detection of torchrun/DDP rank to avoid duplicated console output.

    Default behavior:
      - If WORLD_SIZE<=1: print normally.
      - If WORLD_SIZE>1: only rank0 prints INFO/kv lines.

    Override:
      - Set SPIDER_LOG_ALL_RANKS=1 to print from all ranks.
    """
    try:
        if str(os.environ.get("SPIDER_LOG_ALL_RANKS", "0")).strip().lower() in {"1", "true", "yes", "y"}:
            return True
    except Exception:
        pass

    # torchrun sets these even before torch.distributed init.
    try:
        ws = int(os.environ.get("WORLD_SIZE", "1") or "1")
    except Exception:
        ws = 1
    if int(ws) <= 1:
        return True

    # Prefer torch.distributed rank if initialized; else fall back to env.
    rk = None
    try:
        import torch.distributed as dist  # type: ignore

        if bool(dist.is_available()) and bool(dist.is_initialized()):
            rk = int(dist.get_rank())
    except Exception:
        rk = None
    if rk is None:
        try:
            rk = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0") or "0") or 0)
        except Exception:
            rk = 0
    return int(rk) == 0


def info(msg: str, *, section: str = "GEN") -> None:
    if _is_main_rank():
        print(_fmt("INFO", section, msg), flush=True)


def warn(msg: str, *, section: str = "GEN") -> None:
    # Warnings are important for debugging; keep them on all ranks.
    print(_fmt("WARN", section, msg), flush=True)


def error(msg: str, *, section: str = "GEN") -> None:
    # Errors are important for debugging; keep them on all ranks.
    print(_fmt("ERROR", section, msg), flush=True)


def kv(*, section: str = "GEN", level: str = "INFO", prefix: Optional[str] = None, **items) -> None:
    """
    Print key=value items in a consistent one-liner.
    """
    parts = []
    if prefix:
        parts.append(str(prefix))
    for k, v in items.items():
        parts.append(f"{k}={v}")
    # Treat kv like INFO: avoid duplicates under torchrun by default.
    if str(level or "INFO").strip().upper() == "INFO":
        if not _is_main_rank():
            return
    print(_fmt(level, section, " ".join(parts)), flush=True)


