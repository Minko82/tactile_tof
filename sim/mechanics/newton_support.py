"""Pinned Newton compatibility and runtime provenance helpers."""

from __future__ import annotations

import inspect
from pathlib import Path
import subprocess
from typing import Any


SUPPORTED_NEWTON_GIT_SHA = "8baee876dc5f001c66f1cbafec16246a3fb6f6f6"
SUPPORTED_NEWTON_VERSION = "1.2.0.dev0"


def git_revision(path: str | Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(path)), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def verify_newton_revision(active_sha: str, *, strict: bool) -> bool:
    matches = active_sha == SUPPORTED_NEWTON_GIT_SHA
    if matches:
        return True
    message = (
        "Newton compatibility check failed: active Git SHA "
        f"{active_sha!r}, supported SHA {SUPPORTED_NEWTON_GIT_SHA!r}. "
        "sim/mechanics/contact.py reconstructs reactions from internal Newton "
        "contact arrays and must be revalidated after every Newton update."
    )
    if strict:
        raise RuntimeError(message)
    print("\n" + "!" * 78)
    print(f"[mechanics] WARNING: {message}")
    print("!" * 78 + "\n")
    return False


def deterministic_constructor_kwargs(
    factory: Any, requested: bool
) -> tuple[dict[str, bool], bool]:
    """Return a supported deterministic keyword, if the active API exposes one."""

    if not requested:
        return {}, False
    parameters = inspect.signature(factory).parameters
    for name in ("deterministic", "enable_determinism"):
        if name in parameters:
            return {name: True}, True
    return {}, False
