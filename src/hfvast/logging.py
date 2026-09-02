"""Console/logging setup with secret redaction at every output path."""

from __future__ import annotations

import sys

from rich.console import Console

from hfvast.utils.redact import redact


def make_console(verbose: int = 0) -> Console:
    return Console(
        file=sys.stdout,
        soft_wrap=True,
        highlight=False,
        quiet=False,
    )


def make_error_console() -> Console:
    return Console(file=sys.stderr, highlight=False)


class SafeConsole:
    """Thin wrapper ensuring no output can leak registered secrets."""

    def __init__(self, console: Console) -> None:
        self._console = console

    def print(self, *args: object, **kwargs: object) -> None:
        rendered = " ".join(str(a) for a in args)
        self._console.print(redact(rendered), **kwargs)  # type: ignore[arg-type]

    @property
    def raw(self) -> Console:
        return self._console
