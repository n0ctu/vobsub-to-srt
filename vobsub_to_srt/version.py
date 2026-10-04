"""The app version, from the most specific source available: the VTS_VERSION environment variable
(set from the git tag when the Docker image is built), `git describe` when running from a checkout,
else the installed package's metadata. Glyph sets record it as `learned_with`."""
from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def app_version() -> str:
    env = os.environ.get("VTS_VERSION", "").strip()
    if env:
        return env
    root = Path(__file__).resolve().parent.parent
    if (root / ".git").exists():
        try:
            out = subprocess.run(["git", "-C", str(root), "describe", "--tags", "--always", "--dirty"],
                                 capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip().removeprefix("v")
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        from importlib.metadata import version
        return version("vobsub-to-srt")
    except Exception:  # not installed (e.g. a plain source copy)
        return "unknown"
