from __future__ import annotations


def _fmt(msg: str, section: str | None = None) -> str:
    if section:
        return f"[{section}] {msg}"
    return str(msg)


def info(msg: str, *, section: str | None = None) -> None:
    print(_fmt(msg, section=section), flush=True)


def warn(msg: str, *, section: str | None = None) -> None:
    print(_fmt(f"WARNING: {msg}", section=section), flush=True)
