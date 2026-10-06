"""Seed a data directory's glyph memory from the image's baseline (docker entrypoint).

A shipped set that is missing is copied in. A set the image ships in a newer version than the
one this copy came from (`seeded_from`) is replaced: the old copy moves to `superseded/` with its
version in the name, so what the instance learned in between can still be harvested
(tools/memory_import.py) and merged as a delta against the version it started from. Copies with
no `seeded_from` at all (from before 0.2.1) are treated like an older version.

A set an earlier image shipped (it carries `seeded_from`) that this image no longer ships - merged
into another set, or dropped - is archived the same way, with its private sidecar (interim
learning, learned image hashes) when a word-memory dir is given: left in place, a font split over
two sets keeps the comparison flipping between them. Sets learned on the instance itself carry no
`seeded_from` and stay.

usage: python -m vobsub_to_srt.seed BASELINE_DIR DATA_GLYPH_DIR [DATA_WORD_MEMORY_DIR]
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from .version import app_version


def seed(baseline: Path, target: Path, version: str | None = None, private: Path | None = None) -> list[str]:
    version = version or app_version()
    target.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    shipped = {src.name for src in baseline.glob("*.json")}
    for dst in sorted(target.glob("*.json")):
        if dst.name in shipped:
            continue
        try:
            have = json.loads(dst.read_text(encoding="utf-8")).get("seeded_from")
        except (OSError, ValueError):
            continue
        if not have:
            continue                      # learned on this instance: not ours to retire
        archive = target / "superseded"
        archive.mkdir(exist_ok=True)
        shutil.move(str(dst), str(archive / f"{dst.stem}.{have}.json"))
        side = private / dst.name if private is not None else None
        if side is not None and side.exists():
            (private / "superseded").mkdir(exist_ok=True)
            shutil.move(str(side), str(private / "superseded" / f"{dst.stem}.{have}.json"))
        notes.append(f"{dst.name}: retired (no longer shipped; seeded from {have}; archived)")
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
    for n in seed(Path(sys.argv[1]), Path(sys.argv[2]), private=Path(sys.argv[3]) if len(sys.argv) > 3 else None):
        print("seed:", n)


if __name__ == "__main__":
    main()
