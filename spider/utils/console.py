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


def _fmt(level: str, section: str, msg: str) -> str:
    lvl = (level or "INFO").strip().upper()
    sec = (section or "GEN").strip().upper()
    return f"[spider][{lvl}][{sec}] {msg}"


def info(msg: str, *, section: str = "GEN") -> None:
    print(_fmt("INFO", section, msg))


def warn(msg: str, *, section: str = "GEN") -> None:
    print(_fmt("WARN", section, msg))


def error(msg: str, *, section: str = "GEN") -> None:
    print(_fmt("ERROR", section, msg))


def kv(*, section: str = "GEN", level: str = "INFO", prefix: Optional[str] = None, **items) -> None:
    """
    Print key=value items in a consistent one-liner.
    """
    parts = []
    if prefix:
        parts.append(str(prefix))
    for k, v in items.items():
        parts.append(f"{k}={v}")
    print(_fmt(level, section, " ".join(parts)))


