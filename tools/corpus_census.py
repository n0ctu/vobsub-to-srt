"""Census of the benchmark corpus without a vision model: segment every track, fingerprint its
glyph set, group tracks by font + raster, and say which groups the glyph memory already knows.

Per track (cached under .cache/census/, keyed by file size + mtime):
  size, cues, glyph occurrences, unique bitmaps, once-share (>= 0.5 on >= 200 bitmaps means a
  rescaled track), the dominant glyph height (~ x-height) and the bitmaps seen at least twice.

Grouping: tracks are taken largest first; a track joins the first group (same frame size) whose
bitmaps cover at least --threshold of its glyph occurrences, else it starts a group. Exact bitmaps
only, so a rescaled font forms a looser group than a crisp one (its frequent letters still repeat).

usage: python tools/corpus_census.py [--corpus corpus] [--release SUBSTR] [--limit N] [--jobs N]
                                     [--threshold 0.3] [--refresh]
writes corpus/census.csv (tracks) and corpus/census_groups.csv (groups).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CACHE = ROOT / ".cache" / "census"
TRACK_COLS = ["release", "stem", "size", "languages", "tracks", "cues", "glyphs", "unique", "once_share", "rescaled",
              "xheight", "group", "known_db", "known_coverage", "error"]
GROUP_COLS = ["group", "size", "xheight", "tracks", "releases", "cues", "glyphs", "unique", "rescaled_tracks",
              "known_db", "known_coverage", "example", "example_cues"]


def fingerprint(args: tuple[str, str]) -> dict:
    """Worker: segment one track; returns the cached JSON-able record."""
    release, idx = args
    idx_path = Path(idx)
    sub_path = idx_path.with_suffix(".sub")
    stamp = f"{idx_path.stat().st_size}-{idx_path.stat().st_mtime_ns}-{sub_path.stat().st_size}-{sub_path.stat().st_mtime_ns}"
    cpath = CACHE / f"{release}__{idx_path.stem}.json"
    if cpath.exists():
        try:
            rec = json.loads(cpath.read_text())
            if rec.get("stamp") == stamp:
                return rec
        except ValueError:
            pass
    rec = {"release": release, "stem": idx_path.stem, "stamp": stamp, "error": ""}
    try:
        from vobsub_to_srt.segment import fill_mask, segment
        from vobsub_to_srt.vobsub import load_vobsub, parse_idx
        idx_meta = parse_idx(idx_path)
        _, cues = load_vobsub(idx_path, 0)
        keyfreq: Counter = Counter()
        heights: Counter = Counter()
        for c in cues:
            for line in segment(fill_mask(c)):
                for g in line.glyphs:
                    keyfreq[g.key] += 1
                    heights[int(g.bits.shape[0])] += 1
        total = sum(keyfreq.values())
        rec.update({
            "size": "%dx%d" % idx_meta.size, "languages": " ".join(dict.fromkeys(t.lang for t in idx_meta.tracks)),
            "tracks": len(idx_meta.tracks), "cues": len(cues), "glyphs": total, "unique": len(keyfreq),
            "once_share": round(sum(1 for n in keyfreq.values() if n == 1) / max(1, len(keyfreq)), 3),
            "xheight": heights.most_common(1)[0][0] if heights else 0,
            "repeated": {k: n for k, n in keyfreq.items() if n >= 2},
        })
    except Exception as e:  # noqa: BLE001 - a broken track must not stop the census
        rec.update({"size": "", "languages": "", "tracks": 0, "cues": 0, "glyphs": 0, "unique": 0, "once_share": 0,
                    "xheight": 0, "repeated": {}, "error": f"{type(e).__name__}: {e}"[:200]})
    CACHE.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps(rec))
    return rec


def group_tracks(recs: list[dict], threshold: float) -> list[dict]:
    """Greedy font grouping on exact bitmaps (largest tracks seed the groups)."""
    groups: list[dict] = []
    owner: dict[str, set[int]] = defaultdict(set)        # bitmap key -> groups holding it
    for rec in sorted(recs, key=lambda r: -r["glyphs"]):
        if rec["error"] or not rec["glyphs"]:
            rec["group"] = ""
            continue
        hits: Counter = Counter()
        for k, n in rec["repeated"].items():
            for gid in owner.get(k, ()):
                hits[gid] += n
        best = None
        for gid, n in hits.most_common():
            g = groups[gid]
            if g["size"] == rec["size"] and n / rec["glyphs"] >= threshold:
                best = gid
                break
        if best is None:
            best = len(groups)
            groups.append({"id": best, "size": rec["size"], "keys": Counter(), "tracks": [], "xheights": Counter()})
        g = groups[best]
        g["tracks"].append(rec)
        g["xheights"][rec["xheight"]] += rec["glyphs"]
        for k, n in rec["repeated"].items():
            g["keys"][k] += n
            owner[k].add(best)
        rec["group"] = best
    return groups


def known_coverage(groups: list[dict], memory: Path) -> None:
    """Exact coverage of each group's bitmaps by the glyph sets in memory (occurrence-weighted)."""
    from vobsub_to_srt.glyphdb import GlyphDB
    dbs = []
    for p in sorted(memory.glob("*.json")) if memory.is_dir() else []:
        try:
            dbs.append(GlyphDB.load(p))
        except Exception:  # noqa: BLE001
            continue
    for g in groups:
        total = sum(g["keys"].values()) or 1
        best, cov = "", 0.0
        for db in dbs:
            c = sum(n for k, n in g["keys"].items() if k in db.shapes) / total
            if c > cov:
                best, cov = db.name, c
        g["known_db"], g["known_coverage"] = (best, round(cov, 3)) if cov >= 0.05 else ("", 0.0)   # below that it is noise (shared punctuation)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=ROOT / "corpus")
    ap.add_argument("--memory", type=Path, default=ROOT / "glyph-memory")
    ap.add_argument("--release", default=None, help="only releases whose folder name contains this")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--refresh", action="store_true", help="ignore cached fingerprints")
    a = ap.parse_args()
    if a.refresh and CACHE.exists():
        for p in CACHE.glob("*.json"):
            p.unlink()
    tracks = []
    for idx in sorted(a.corpus.rglob("*.idx")):
        if idx.parent == a.corpus or not idx.with_suffix(".sub").exists():
            continue
        if a.release and a.release.lower() not in idx.parent.name.lower():
            continue
        tracks.append((idx.parent.name, str(idx)))
    if a.limit:
        tracks = tracks[:a.limit]
    t0 = time.time()
    with Pool(a.jobs) as pool:
        recs = []
        for i, rec in enumerate(pool.imap_unordered(fingerprint, tracks, chunksize=1), 1):
            recs.append(rec)
            if i % 25 == 0 or i == len(tracks):
                print(f"  {i}/{len(tracks)} tracks fingerprinted ({time.time() - t0:.0f}s)", file=sys.stderr)
    groups = group_tracks(recs, a.threshold)
    known_coverage(groups, a.memory)
    for g in groups:
        for rec in g["tracks"]:
            rec["known_db"], rec["known_coverage"] = g["known_db"], g["known_coverage"]
    recs.sort(key=lambda r: (r["release"], r["stem"]))
    with (a.corpus / "census.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRACK_COLS, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            r["rescaled"] = "yes" if r["unique"] >= 200 and r["once_share"] >= 0.5 else "no"
            r.setdefault("known_db", ""); r.setdefault("known_coverage", "")
            w.writerow(r)
    rows = []
    for g in groups:
        ts = g["tracks"]
        ex = max(ts, key=lambda r: r["cues"])
        rows.append({"group": g["id"], "size": g["size"], "xheight": g["xheights"].most_common(1)[0][0],
                     "tracks": len(ts), "releases": len({r["release"] for r in ts}),
                     "cues": sum(r["cues"] for r in ts), "glyphs": sum(r["glyphs"] for r in ts),
                     "unique": len(g["keys"]), "rescaled_tracks": sum(1 for r in ts if r["rescaled"] == "yes"),
                     "known_db": g["known_db"], "known_coverage": g["known_coverage"],
                     "example": f'{ex["release"]}/{ex["stem"]}', "example_cues": ex["cues"]})
    rows.sort(key=lambda r: -r["cues"])
    with (a.corpus / "census_groups.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=GROUP_COLS)
        w.writeheader()
        w.writerows(rows)
    errors = [r for r in recs if r["error"]]
    known = [r for r in rows if r["known_coverage"] and r["known_coverage"] >= 0.5]
    print(f"{len(recs)} tracks, {len(errors)} unreadable, {len(groups)} font groups "
          f"({len(known)} covered >= 50% by the glyph memory), {time.time() - t0:.0f}s")
    print(f"frame sizes: {dict(Counter(r['size'] for r in recs).most_common())}")
    print(f"rescaled tracks: {sum(1 for r in recs if r['rescaled'] == 'yes')}")
    print("largest groups (cues, tracks, releases, size, x-height, known):")
    for r in rows[:12]:
        print(f"  #{r['group']:<4} {r['cues']:>7} {r['tracks']:>4} {r['releases']:>3}  {r['size']:<10} {r['xheight']:>3}px"
              f"  {r['known_db'] or '-'} {r['known_coverage'] or ''}  e.g. {r['example']}")
    for r in errors[:10]:
        print(f"  unreadable: {r['release']}/{r['stem']}: {r['error']}")
    print(f"-> {a.corpus / 'census.csv'}, {a.corpus / 'census_groups.csv'}")


if __name__ == "__main__":
    main()
