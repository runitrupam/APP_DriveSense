"""File and payload hashing used for input identity and provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CHUNK_SIZE = 1024 * 1024


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file, streamed in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """SHA-256 of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_json_hash(payload: Any) -> str:
    """SHA-256 of a JSON-serialisable payload, independent of key order."""
    dumped = json.dumps(payload, sort_keys=True, default=str)
    return sha256_text(dumped)


def short_hash(value: str, length: int = 8) -> str:
    """Truncate a hex digest for display."""
    return value[:length]
