"""Integrity verification helpers to detect unexpected tampering."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "bybit_app" / "_data" / "integrity.json"


class IntegrityError(RuntimeError):
    """Raised when integrity verification fails."""


def load_manifest(path: Path | None = None) -> dict[str, str]:
    manifest_path = Path(path) if path is not None else DEFAULT_MANIFEST
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("integrity manifest must be a mapping")
    result: dict[str, str] = {}
    for rel_path, digest in payload.items():
        if not isinstance(rel_path, str) or not isinstance(digest, str):
            raise ValueError("manifest entries must be string -> string pairs")
        result[rel_path] = digest.lower()
    return result


def calculate_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_integrity(
    manifest_path: Path | None = None,
    *,
    root: Path | None = None,
) -> Dict[str, str]:
    """Return a mapping of path -> actual digest for mismatched files."""

    manifest = load_manifest(manifest_path)
    if not manifest:
        return {}

    base = Path(root) if root is not None else PROJECT_ROOT
    mismatches: Dict[str, str] = {}

    for rel_path, expected in manifest.items():
        target = (base / rel_path).resolve()
        if not target.exists():
            mismatches[rel_path] = "missing"
            continue
        actual = calculate_digest(target)
        if actual.lower() != expected.lower():
            mismatches[rel_path] = actual
    return mismatches


def assert_integrity(manifest_path: Path | None = None) -> None:
    """Raise :class:`IntegrityError` when manifest verification fails."""

    if should_skip_integrity():
        return

    mismatches = verify_integrity(manifest_path)
    if mismatches:
        summary = ", ".join(f"{path}: {digest}" for path, digest in mismatches.items())
        raise IntegrityError(f"integrity check failed: {summary}")


def should_skip_integrity() -> bool:
    marker = os.getenv("BYBITBOT_SKIP_INTEGRITY", "").strip().lower()
    return marker in {"1", "true", "yes", "on"}


__all__ = [
    "IntegrityError",
    "assert_integrity",
    "calculate_digest",
    "load_manifest",
    "should_skip_integrity",
    "verify_integrity",
]
