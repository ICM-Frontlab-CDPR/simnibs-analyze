"""
Logging utilities for the SimNIBS analysis pipeline.

Usage in every module:
    from _logging import get_logger
    logger = get_logger(__name__)

    logger.info("message")
    logger.warning("message")
    logger.error("message")
    logger.step("SECTION TITLE")   # rich rule across the terminal width
"""

from __future__ import annotations

import datetime
import logging

import rich.console
import rich.theme

_THEME = rich.theme.Theme(
    {
        "asctime": "green",
        "name": "dim cyan",
        "debug": "dim",
        "info": "white",
        "warning": "yellow",
        "error": "bold red",
        "step": "bold cyan",
    }
)

# Shared console — lazy so pytest capture still works
_console: rich.console.Console | None = None


def _get_console() -> rich.console.Console:
    global _console
    if _console is None:
        _console = rich.console.Console(soft_wrap=True, theme=_THEME)
    return _console


class _PipelineLogger:
    """Lightweight rich-based logger."""

    def __init__(self, name: str = "", level: int = logging.INFO) -> None:
        self.name = name
        self.level = level

    def _emit(self, kind: str, msg: str) -> None:
        if getattr(logging, kind.upper()) < self.level:
            return
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        name_part = f"[name]{self.name}[/]  " if self.name else ""
        _get_console().print(f"[asctime]{ts}[/]  {name_part}[{kind}]{msg}[/]")

    def debug(self, msg: str) -> None:
        self._emit("debug", msg)

    def info(self, msg: str) -> None:
        self._emit("info", msg)

    def warning(self, msg: str) -> None:
        self._emit("warning", f"⚠  {msg}")

    def error(self, msg: str) -> None:
        self._emit("error", f"✗  {msg}")

    def step(self, title: str) -> None:
        """Print a full-width section banner (rich rule)."""
        _get_console().rule(f"[step] {title} [/]", style="cyan")


def get_logger(name: str = "", level: int = logging.INFO) -> _PipelineLogger:
    """Return a pipeline logger bound to *name*."""
    return _PipelineLogger(name=name, level=level)
