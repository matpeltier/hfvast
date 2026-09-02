"""Central secret redaction.

Credentials are registered here the moment they are resolved; every output path
(console, logs, exception formatting) funnels text through ``redact()`` so that a
known secret string can never reach the terminal, a log file, or a traceback.
"""

from __future__ import annotations

import re

_REDACTED = "***REDACTED***"

# Patterns that look like credentials even when not registered explicitly.
_BUILTIN_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-hfvast-[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?i)vast[ _-]?api[ _-]?key[=: ]+\S+"),
)


class SecretRedactor:
    """Replaces registered secrets and credential-shaped strings with a marker."""

    def __init__(self) -> None:
        self._secrets: set[str] = set()

    def register(self, *secrets: str | None) -> None:
        for secret in secrets:
            if secret and len(secret) >= 8:
                self._secrets.add(secret)

    def redact(self, text: str) -> str:
        if not text:
            return text
        for secret in self._secrets:
            if secret in text:
                text = text.replace(secret, _REDACTED)
        for pattern in _BUILTIN_PATTERNS:
            text = pattern.sub(_REDACTED, text)
        return text


_global = SecretRedactor()


def get_redactor() -> SecretRedactor:
    return _global


def register_secrets(*secrets: str | None) -> None:
    _global.register(*secrets)


def redact(text: object) -> str:
    """Redact a string (or any object's str()) before it is printed or logged."""
    return _global.redact(str(text))
