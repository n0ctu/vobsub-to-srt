"""Scan corpus/ and (re)write corpus/index.csv: one row per .idx track.

Generated columns: release (folder), stem, languages (id: lines), cues (timestamp lines), size,
ground_truth (a <stem>.gt.srt next to the track). Hand-written columns (source, style, notes)
of an existing index.csv are kept by (release, stem).

usage: python tools/corpus_index.py [corpus-dir]
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

GENERATED = ["release", "stem", "languages", "cues", "size", "ground_truth"]
MANUAL = ["source", "style", "notes"]


def scan_idx(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    langs = re.findall(r"^id:\s*([A-Za-z-]+)", text, re.M)
    size = re.search(r"^size:\s*(\d+x\d+)", text, re.M)
    return {"release": path.parent.name, "stem": path.stem, "languages": " ".join(dict.fromkeys(langs)),
            "cues": str(len(re.findall(r"^timestamp:", text, re.M))), "size": size.group(1) if size else "",
            "ground_truth": "yes" if path.with_suffix(".gt.srt").exists() else "no"}


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("corpus")
    index = root / "index.csv"
    manual: dict[tuple[str, str], dict] = {}
    if index.exists():
        with index.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                manual[(row.get("release", ""), row.get("stem", ""))] = {k: row.get(k, "") for k in MANUAL}
    rows = []
    for idx in sorted(p for p in root.rglob("*.idx") if p.parent != root):
        if not idx.with_suffix(".sub").exists():
            print(f"skipped (no .sub): {idx.relative_to(root)}", file=sys.stderr)
            continue
        row = scan_idx(idx)
        row.update(manual.get((row["release"], row["stem"]), {k: "" for k in MANUAL}))
        rows.append(row)
    with index.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=GENERATED + MANUAL)
        w.writeheader()
        w.writerows(rows)
    releases = len({r["release"] for r in rows})
    gt = sum(1 for r in rows if r["ground_truth"] == "yes")
    print(f"{len(rows)} tracks in {releases} releases, {sum(int(r['cues']) for r in rows)} cues, {gt} with ground truth -> {index}")


if __name__ == "__main__":
    main()
