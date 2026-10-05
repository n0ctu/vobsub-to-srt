"""Bring glyph sets a web instance has learned into the repository's baseline (glyph-memory/).

    pull  URL DIR            download every set of a running instance (GET /api/glyphs/<name>/download)
    plan  DIR [--repo R]     say what each pulled set would do to the baseline, and check the new glyphs
    sheet DIR OUT.png        contact sheet of the glyphs that are new to the baseline, with their labels
    apply DIR [--repo R]     write the accepted sets into the baseline (refuses flagged sets unless --force)
    audit [--repo R]         check every shipped set against itself (conflicts, dissent, labels too wide for their glyph)

A pulled set is one of: an *update* of a shipped set (same name: the instance started from the
shipped file and added votes), a *delta* (same name, but the copy records the shipped version it was
seeded from: only the votes it added since then are merged, so nothing is counted twice; the
server's superseded/<name>.<version>.json archives are read the same way), a set to *merge* into a
shipped set (another name, but the same font at the same size: most of its letters are bitmaps the
shipped set already knows), or a *new* set. Only confirmed labels (the same rule the converter trusts) count. Every new confirmed cluster
is checked against the set it joins: a nearer cluster carrying a different label is a conflict,
labels with a dissenting read, from the confusable classes with few reads, or spelling several
letters (a fused pair) are marked for a closer look. Sets learned by a version older than --min-version are refused unless
--allow-unversioned (sets from before 0.1.0 carry no version at all).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vobsub_to_srt.glyphdb import CLUSTER_TOL, MIN_VOTES, GlyphDB, is_strict, tol_for, trusted_label  # noqa: E402

REPO = Path(__file__).resolve().parent.parent / "glyph-memory"
MERGE_SHARE = 0.5        # share of a pulled set's letter occurrences a shipped set knows bit for bit
_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


# ---------- pull ----------
def pull(url: str, out: Path) -> None:
    import httpx
    out.mkdir(parents=True, exist_ok=True)
    base = url.rstrip("/")
    with httpx.Client(timeout=60) as c:
        sets = c.get(f"{base}/api/glyphs").raise_for_status().json()["sets"]
        for s in sets:
            name = s["name"]
            if not _NAME.match(name):
                continue
            body = c.get(f"{base}/api/glyphs/{name}/download").raise_for_status().content
            (out / f"{name}.json").write_bytes(body)
            print(f"pulled {name:24} {len(body) // 1024:5} KB  shapes {s.get('shapes', '?'):6}  learned_with {s.get('learned_with') or '-'}"
                  + ("  (shipped set, grown)" if s.get("baseline") else ""))


# ---------- inspection ----------
@dataclass
class Cluster:
    key: str
    label: str
    bits: np.ndarray
    top_rel: int
    votes: Counter
    n: int                      # occurrences over all files
    flags: list[str] = field(default_factory=list)


def confirmed_clusters(db: GlyphDB) -> dict[str, Cluster]:
    """Canonical key -> the cluster's confirmed label (best variant), trusted by the converter's rule."""
    occ: Counter = Counter()
    for s in db.shapes.values():
        occ[s.cluster] += s.n
    out: dict[str, Cluster] = {}
    for key, s in db.shapes.items():
        if s.cluster != key or not s.variants:
            continue
        v = max(s.variants, key=lambda v: sum(v.votes.values()))
        label = trusted_label(v.votes)[0]
        if label:
            out[key] = Cluster(key, label, s.bits, v.top_rel, +v.votes, int(occ[key]))
    return out


def version_tuple(v: str | None) -> tuple[int, ...] | None:
    if not v:
        return None
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", v)
    return tuple(int(x) for x in m.groups()) if m else None


def check_cluster(c: Cluster, target: GlyphDB, own_key: str | None) -> None:
    """Flags for a confirmed cluster that is new to `target`: a dissenting read, a fused pair of
    letters, a nearer cluster in the target carrying a different label (a conflict at the
    converter's own tolerance), and for the confusable classes (I l 1, digits, g) a differently
    labelled confusable cluster within the ordinary clustering distance."""
    if len(c.label) > 1:
        c.flags.append("fused letters")
    if len(c.votes) > 1:
        c.flags.append("dissent " + ",".join(f"{k}:{n}" for k, n in c.votes.most_common()))
    nearest_other = None            # nearest cluster whose *confirmed* label differs (strays are what votes are for)
    for r, key, dh in target.near_candidates(c.bits, max_ratio=CLUSTER_TOL):
        if key == own_key or dh:
            continue
        canon = target.shapes[target.shapes[key].cluster]
        v = canon.variant(c.top_rel, target.pos_tol)
        lab = trusted_label(v.votes)[0] if v is not None else None
        if lab and lab != c.label and (nearest_other is None or r < nearest_other[0]):
            nearest_other = (r, lab)
    if nearest_other is None:
        return
    r, lab = nearest_other
    if r <= tol_for(c.label, lab):
        c.flags.append(f"conflict: {lab!r} at {r:.3f}")
    elif is_strict(c.label) and is_strict(lab):
        c.flags.append(f"close to {lab!r} ({r:.3f})")


@dataclass
class Plan:
    name: str
    path: Path
    action: str                 # update | merge | new | skip
    target: str | None          # shipped set it updates or merges into
    learned_with: str | None
    new: list[Cluster]          # confirmed clusters the baseline does not have yet
    changed: list[tuple[str, str, str]]   # (key, old label, new label) on updates
    seed: str | None = None     # delta merges: the shipped version the copy started from
    lost: int = 0               # shipped confirmed labels the update no longer confirms
    ambiguous: int = 0          # ... of which I/l clusters that became a word-decided ambiguity
    refused: list[str] = field(default_factory=list)

    @property
    def flagged(self) -> list[Cluster]:
        return [c for c in self.new if c.flags]


_SUPERSEDED = re.compile(r"^(?P<name>[a-z]+-[a-z]+-[0-9a-f]{4})\.(?P<ver>.+)$")


def base_dir(repo: Path, version: str) -> Path | None:
    """The shipped sets as of release `version`, extracted from the repository's history."""
    import subprocess
    try:
        top = Path(subprocess.run(["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
                                  capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        top = repo.parent
    out = top / ".cache" / "bench" / "baseline-at" / version
    if out.is_dir():
        return out
    tag = version if version.startswith("v") else "v" + version
    try:
        names = subprocess.run(["git", "-C", str(top), "ls-tree", "--name-only", f"{tag}:glyph-memory"],
                               capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return None
    out.mkdir(parents=True, exist_ok=True)
    for n in names:
        if n.endswith(".json"):
            body = subprocess.run(["git", "-C", str(top), "show", f"{tag}:glyph-memory/{n}"],
                                  capture_output=True, text=True, check=True).stdout
            (out / n).write_text(body, encoding="utf-8")
    return out


def make_plan(path: Path, repo: Path, min_version: str, allow_unversioned: bool, seed_version: str | None = None) -> Plan:
    pulled = GlyphDB.load(path)
    d = json.loads(path.read_text())
    learned_with = d.get("learned_with")
    name = path.stem
    m = _SUPERSEDED.match(name)
    if m:                                    # superseded/<name>.<version>.json from the server archive
        name = m.group("name")
        if not pulled.seeded_from and m.group("ver") != "unversioned":
            pulled.seeded_from = m.group("ver")
    if seed_version and not pulled.seeded_from and (repo / f"{name}.json").exists():
        pulled.seeded_from = seed_version          # a copy installed before seeds were recorded
    plan = Plan(name, path, "new", None, learned_with, [], [])
    vt = version_tuple(learned_with)
    if vt is None:
        if not allow_unversioned:
            plan.refused.append("no learned_with version (learned before 0.1.0); --allow-unversioned to accept")
    elif vt < version_tuple(min_version):
        plan.refused.append(f"learned with {learned_with}, older than --min-version {min_version}")
    mine = confirmed_clusters(pulled)
    shipped = {p.stem: p for p in sorted(repo.glob("*.json"))}

    if plan.name in shipped:
        plan.action, plan.target = "update", plan.name
        base = GlyphDB.load(shipped[plan.name])
        if pulled.seeded_from and base_dir(repo, pulled.seeded_from) is not None:
            # a server copy that started from an older shipped version: only what it learned since
            # (its votes minus the seed's) is new, and that delta merges into the current set
            plan.action, plan.seed = "delta", pulled.seeded_from
            seed = GlyphDB.load(base_dir(repo, pulled.seeded_from) / f"{plan.name}.json")
            was = confirmed_clusters(base)
            for key, c in confirmed_clusters(pulled).items():
                if key in was:
                    continue
                sv = seed.lookup(key, c.top_rel) if key in seed.shapes else None
                if sv is not None and +sv.votes == c.votes:
                    continue                                  # unchanged since the seed
                check_cluster(c, base, None)
                plan.new.append(c)
            return plan
        if base.charset != pulled.charset:
            plan.refused.append(f"charset {pulled.charset} differs from the shipped {base.charset}")
        was = confirmed_clusters(base)
        for key, c in was.items():
            now = mine.get(key)
            s = pulled.shapes.get(key)
            if now is None and s is not None:
                # the cluster may have been joined into another one: follow the pulled cluster chain
                now = mine.get(s.cluster)
            if now is None:
                # an I/l cluster that collected enough reads of both is "conflicting" for the vote
                # rule but not lost: the converter lets the word decide for such clusters
                v = pulled.shapes[s.cluster].variant(c.top_rel, pulled.pos_tol) if s is not None else None
                voted = {k for k, n in (v.votes.items() if v else ()) if n >= MIN_VOTES}   # strays do not count
                if voted and c.label in voted and voted <= {"I", "l", "|"}:
                    plan.ambiguous += 1
                else:
                    plan.lost += 1
            elif now.label != c.label:
                plan.changed.append((key, c.label, now.label))
        for key, c in mine.items():
            if key not in was:
                check_cluster(c, pulled, key)
                plan.new.append(c)
        if plan.changed:
            plan.refused.append(f"{len(plan.changed)} shipped labels changed")
        if plan.lost:
            plan.refused.append(f"would drop {plan.lost} confirmed labels the shipped set has (merged from elsewhere?)")
        return plan

    # another name: the same font at the same size as a shipped set?
    total = sum(c.n for c in mine.values()) or 1
    best, best_share = None, 0.0
    for stem, p in shipped.items():
        base = GlyphDB.load(p)
        if base.charset != pulled.charset:
            continue
        share = sum(c.n for c in mine.values() if c.key in base.shapes) / total
        if share > best_share:
            best, best_share = (stem, base), share
    if best and best_share >= MERGE_SHARE:
        stem, base = best
        plan.action, plan.target = "merge", stem
        if plan.name in base.merged_from:
            plan.action = "skip"
            plan.refused.append(f"already merged into {stem}")
            return plan
        was = confirmed_clusters(base)
        for key, c in mine.items():
            if key in was:
                if was[key].label != c.label:
                    plan.changed.append((key, was[key].label, c.label))
                continue
            check_cluster(c, base, None)
            plan.new.append(c)
        if plan.changed:
            plan.refused.append(f"{len(plan.changed)} labels disagree with the shipped set")
        return plan
    for key, c in mine.items():
        check_cluster(c, pulled, key)
        plan.new.append(c)
    if not plan.new:
        plan.refused.append("no confirmed labels at all (nothing the converter would trust)")
    return plan


def describe(plan: Plan) -> str:
    head = f"{plan.name:24} {plan.action:6}" + (f" -> {plan.target}" if plan.target and plan.target != plan.name else "")
    if plan.action == "delta":
        head += f" (since {plan.seed})"
    head += f"  learned_with {plan.learned_with or '-'}  new confirmed clusters {len(plan.new)}"
    if plan.action == "update":
        head += f"  changed {len(plan.changed)}  lost {plan.lost}" + (f"  I/l now ambiguous {plan.ambiguous}" if plan.ambiguous else "")
    if plan.flagged:
        head += f"  flagged {len(plan.flagged)}"
    lines = [head]
    for key, old, new in plan.changed[:10]:
        lines.append(f"    label changed {old!r} -> {new!r} ({key[:12]})")
    for c in plan.flagged[:30]:
        lines.append(f"    {c.label!r:6} n={c.n:<4} {'; '.join(c.flags)}")
    for r in plan.refused:
        lines.append(f"    REFUSED: {r}")
    return "\n".join(lines)


def plans_for(d: Path, repo: Path, min_version: str, allow_unversioned: bool, seed_version: str | None = None) -> list[Plan]:
    return [make_plan(p, repo, min_version, allow_unversioned, seed_version) for p in sorted(d.glob("*.json"))]


# ---------- sheet ----------
def _font(size: int):
    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def sheet(plans: list[Plan], out: Path, scale: int = 3, cols: int = 8) -> int:
    """Every new confirmed cluster as a tile: the bitmap, the label, the vote count and the flags
    (red frame when flagged). One row of tiles per set, sorted by frequency."""
    tiles: list[tuple[Plan, Cluster]] = [(p, c) for p in plans for c in sorted(p.new, key=lambda c: -c.n)]
    if not tiles:
        return 0
    cell_w, cell_h = 150, 150
    font, small = _font(18), _font(12)
    rows = []
    for p in plans:
        if p.new:
            rows.append((p, [c for c in sorted(p.new, key=lambda c: -c.n)]))
    height = sum(28 + cell_h * ((len(cs) + cols - 1) // cols) for _, cs in rows) + 10
    im = Image.new("RGB", (cols * cell_w + 10, height), "white")
    dr = ImageDraw.Draw(im)
    y = 5
    for p, cs in rows:
        dr.text((8, y + 4), f"{p.name}: {p.action}" + (f" -> {p.target}" if p.target and p.target != p.name else "")
                + f", {len(cs)} new clusters, {len([c for c in cs if c.flags])} flagged", fill="black", font=font)
        y += 28
        for i, c in enumerate(cs):
            x0, y0 = 5 + (i % cols) * cell_w, y + (i // cols) * cell_h
            dr.rectangle([x0, y0, x0 + cell_w - 4, y0 + cell_h - 4], outline="red" if c.flags else "#cccccc", width=2)
            g = Image.fromarray((~c.bits * 255).astype(np.uint8))
            s = min(scale, max(1, (cell_w - 20) // max(1, g.width)), max(1, 80 // max(1, g.height)))
            g = g.resize((g.width * s, g.height * s), Image.NEAREST)
            im.paste(g, (x0 + (cell_w - g.width) // 2, y0 + 8))
            dr.text((x0 + 8, y0 + 94), c.label, fill="black", font=font)
            dr.text((x0 + 8, y0 + 116), f"{c.n} seen, {sum(c.votes.values())} reads", fill="#555555", font=small)
            if c.flags:
                dr.text((x0 + 8, y0 + 131), "; ".join(c.flags)[:24], fill="red", font=small)
        y += cell_h * ((len(cs) + cols - 1) // cols)
    im.save(out)
    return len(tiles)


# ---------- apply ----------
def apply_plan(plan: Plan, repo: Path, force: bool) -> str:
    if plan.action == "skip" or (plan.refused and not force):
        return f"{plan.name}: skipped ({'; '.join(plan.refused)})"
    if plan.flagged and not force:
        return f"{plan.name}: skipped ({len(plan.flagged)} flagged clusters; review the sheet, then --force)"
    if plan.action == "delta" and not plan.new and not force:
        return f"{plan.name}: nothing learned since {plan.seed}"
    if plan.action in ("update", "new"):
        db = GlyphDB.load(plan.path)              # a load/save round trip normalises the file
        db.path = repo / f"{plan.name}.json"
        if plan.action == "update" and not db.learned_with:
            db.learned_with = GlyphDB.load(db.path).learned_with     # a copy from before the stamp keeps its lineage's
        db.journal = None                         # write this content, do not replay onto the old file
        db.dirty = True
        db.save(final=True)
        db.path.with_suffix(".lock").unlink(missing_ok=True)
        return f"{plan.name}: {'updated' if plan.action == 'update' else 'added'} ({len(plan.new)} new clusters)"
    # merge: carry votes and gap statistics into the shipped set; sequences stay behind (their keys
    # depend on how the parts clustered in the pulled copy). A delta merge carries only the votes
    # a server copy added since the shipped version it was seeded from.
    target = GlyphDB.load(repo / f"{plan.target}.json")
    pulled = GlyphDB.load(plan.path)
    seed = GlyphDB.load(base_dir(repo, plan.seed) / f"{plan.name}.json") if plan.action == "delta" else None
    members: dict[str, list[str]] = {}
    for key, s in pulled.shapes.items():
        members.setdefault(s.cluster, []).append(key)
    votes = 0
    for key, s in pulled.shapes.items():
        if s.cluster != key:
            continue
        for v in s.variants:
            # the seed's votes for this cluster, whichever of its (pulled) members carried them
            # in the seed: clusters may have been joined since
            had: Counter = Counter()
            if seed is not None:
                for mk in members[key]:
                    if mk in seed.shapes:
                        sv = seed.lookup(mk, v.top_rel)
                        if sv is not None and seed.shapes[mk].cluster == mk:
                            had.update(+sv.votes)
            for label, n in (+v.votes).items():
                n -= had.get(label, 0)
                if n <= 0:
                    continue
                style = "".join(f for f in "bi" if v.styles.get(f, [0, 0])[1] > v.styles.get(f, [0, 0])[0])
                target.add_vote(key, s.bits, v.top_rel, label, style, weight=n)
                votes += n
    if seed is not None:
        return _finish_merge(target, plan, votes, pulled, delta=True)
    for it, gd in pulled.gaps.items():
        for g, (lc, sc) in gd.items():
            target.gaps[it][g][0] += lc
            target.gaps[it][g][1] += sc
    for k, (lc, sc) in pulled.pair_gaps.items():
        target.pair_gaps[k][0] += lc
        target.pair_gaps[k][1] += sc
    return _finish_merge(target, plan, votes, pulled)


def _finish_merge(target: GlyphDB, plan: Plan, votes: int, pulled: GlyphDB, delta: bool = False) -> str:
    if not delta:
        target.merged_from.append(plan.name)
    target.journal = None
    target.dirty = True
    target.save(final=True)
    target.path.with_suffix(".lock").unlink(missing_ok=True)
    what = f"delta since {plan.seed} merged" if delta else f"merged into {plan.target}"
    return f"{plan.name}: {what} ({votes} votes, {len(plan.new)} new clusters, {len(pulled.sequences)} sequences left behind)"


# ---------- audit ----------
def audit(repo: Path) -> int:
    """Every confirmed cluster of every set checked against its own set: a nearer cluster with a
    different confirmed label, a confusable look-alike close by, a dissenting read, and a
    multi-letter label on a glyph too narrow for its letters. Returns the number of findings."""
    total = 0
    for p in sorted(repo.glob("*.json")):
        db = GlyphDB.load(p)
        found: list[str] = []
        two = 0
        for c in confirmed_clusters(db).values():
            check_cluster(c, db, c.key)
            if len(c.label) > 1 and not db.fits_width(c.label, int(c.bits.shape[1])):
                c.flags.append(f"too narrow for {c.label!r} ({c.bits.shape[1]} px)")
            if sum(c.votes.values()) == 2:
                two += 1
            real = [f for f in c.flags if f.startswith(("conflict", "too narrow", "dissent"))]
            if real:
                found.append(f"    {c.label!r:6} n={c.n:<5} {'; '.join(real)}")
        total += len(found)
        print(f"{p.stem:24} confirmed {len(confirmed_clusters(db)):4}  on two reads {two:4}  findings {len(found)}")
        for line in found[:40]:
            print(line)
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pull"); p.add_argument("url"); p.add_argument("dir", type=Path)
    p = sub.add_parser("audit"); p.add_argument("--repo", type=Path, default=REPO)
    for name in ("plan", "sheet", "apply"):
        p = sub.add_parser(name)
        p.add_argument("dir", type=Path)
        if name == "sheet":
            p.add_argument("out", type=Path)
        p.add_argument("--repo", type=Path, default=REPO)
        p.add_argument("--min-version", default="0.1.0")
        p.add_argument("--allow-unversioned", action="store_true")
        p.add_argument("--seed-version", default=None, help="treat copies of shipped sets that record no seed as seeded from this release")
        p.add_argument("--only", default="", help="comma-separated set names")
        if name == "apply":
            p.add_argument("--force", action="store_true", help="apply refused or flagged sets too")
    a = ap.parse_args()
    if a.cmd == "pull":
        pull(a.url, a.dir)
        return
    if a.cmd == "audit":
        n = audit(a.repo)
        print(f"{n} findings")
        return
    plans = plans_for(a.dir, a.repo, a.min_version, a.allow_unversioned, a.seed_version)
    if a.only:
        only = set(a.only.split(","))
        plans = [p for p in plans if p.name in only]
    if a.cmd == "plan":
        for p in plans:
            print(describe(p))
        print(f"baseline would hold {len(list(a.repo.glob('*.json'))) + sum(1 for p in plans if p.action == 'new' and not p.refused)} sets")
    elif a.cmd == "sheet":
        n = sheet(plans, a.out)
        print(f"{n} new clusters on {a.out}" if n else "nothing new to show")
    elif a.cmd == "apply":
        for p in plans:
            print(apply_plan(p, a.repo, a.force))


if __name__ == "__main__":
    main()
