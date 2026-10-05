"""Seed a data directory's glyph memory from the image's baseline (docker entrypoint).

A shipped set that is missing is copied in. A set the image ships in a newer version than the
one this copy came from (`seeded_from`) is replaced: the old copy moves to `superseded/` with its
version in the name, so what the instance learned in between can still be harvested
(tools/memory_import.py) and merged as a delta against the version it started from. Copies with
no `seeded_from` at all (from before 0.2.1) are treated like an older version.

usage: python -m vobsub_to_srt.seed BASELINE_DIR DATA_GLYPH_DIR
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from .version import app_version


def seed(baseline: Path, target: Path, version: str | None = None) -> list[str]:
    version = version or app_version()
    target.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    for src in sorted(baseline.glob("*.json")):
        dst = target / src.name
        if dst.exists():
            try:
                have = json.loads(dst.read_text(encoding="utf-8")).get("seeded_from")
            except (OSError, ValueError):
                have = None
            if have == version:
                continue
            archive = target / "superseded"
            archive.mkdir(exist_ok=True)
            shutil.move(str(dst), str(archive / f"{src.stem}.{have or 'unversioned'}.json"))
            notes.append(f"{src.name}: replaced (was seeded from {have or 'an unversioned image'}; old copy archived)")
        else:
            notes.append(f"{src.name}: installed")
        d = json.loads(src.read_text(encoding="utf-8"))
        d["seeded_from"] = version
        dst.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return notes


def main() -> None:
    for n in seed(Path(sys.argv[1]), Path(sys.argv[2])):
        print("seed:", n)


if __name__ == "__main__":
    main()
