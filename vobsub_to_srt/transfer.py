"""Cross-resolution transfer: use a glyph DB learned at one raster size as a teacher for another.

Within one raster, glyphs are matched by exact bitmap. Across rasters (e.g. the same font at 1080p
and 720p) bitmaps differ, so the teacher's confirmed shapes are rescaled to the new raster and
compared by overlap. Only matches with a clear margin over every other label are transferred as
confirmed; close calls (typically I vs l) are left to the VLM.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .glyphdb import GlyphDB, trusted_label
from .segment import Glyph

log = logging.getLogger("vobsub_to_srt")

IOU_MIN = 0.80          # minimum overlap to transfer a label as prior
MARGIN = 0.12           # required IoU lead over the best candidate with a different label
# thin strokes whose distinguishing detail (1 px height, i-dot gap) does not survive resampling:
# never transferred, always learned from the VLM at the new raster
FRAGILE = {"I", "l", "|", "i", "1", "!", "j", "í", "ì", "ï", "î"}


@dataclass
class _Ref:
    key: str
    label: str
    bits: np.ndarray      # rescaled to the new raster
    top_rel: float        # rescaled
    n: int = 0            # ink pixels of `bits`


@dataclass
class Transfer:
    scale: float
    coverage: float                  # share of glyph occurrences with a confirmed transfer
    labels: dict[str, tuple[str, str]]   # new key -> (label, teacher key)


def resize_bits(bits: np.ndarray, scale: float) -> np.ndarray:
    h, w = bits.shape
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    im = Image.fromarray(bits.astype(np.uint8) * 255).resize((nw, nh), Image.BOX)
    return np.asarray(im) >= 128


def _iou(a: np.ndarray, b: np.ndarray, na: int | None = None, nb: int | None = None) -> float:
    """Best IoU over +-1 px shifts. `na`/`nb` are the ink counts when the caller has them."""
    na = int(np.count_nonzero(a)) if na is None else na
    nb = int(np.count_nonzero(b)) if nb is None else nb
    h, w = max(a.shape[0], b.shape[0]) + 2, max(a.shape[1], b.shape[1]) + 2
    frame = np.zeros((h, w), bool)
    frame[1:1 + a.shape[0], 1:1 + a.shape[1]] = a
    bh, bw = b.shape
    best = 0.0
    for dy in (0, 1, 2):
        if dy + bh > h:
            continue
        for dx in (0, 1, 2):
            if dx + bw > w:
                continue
            inter = int(np.count_nonzero(frame[dy:dy + bh, dx:dx + bw] & b))
            union = na + nb - inter                 # |A u B| = |A| + |B| - |A n B|
            if union and inter / union > best:
                best = inter / union
    return best


def _refs(teacher: GlyphDB, scale: float) -> list[_Ref]:
    out = []
    for s in teacher.shapes.values():
        for v in s.variants:
            lab = trusted_label(v.votes)[0]
            if lab:          # confirmed, non-fragment labels only
                bits = resize_bits(s.bits, scale)
                out.append(_Ref(s.key, lab, bits, v.top_rel * scale, int(np.count_nonzero(bits))))
    return out


class _Refs:
    """A teacher's rescaled shapes, stacked per bitmap size for batched overlap computation."""

    def __init__(self, refs: list[_Ref]):
        rows: dict[tuple[int, int], list[_Ref]] = {}
        self.by_h: dict[int, list[tuple[_Ref, int]]] = {}    # height -> (ref, row in its stack)
        for r in refs:
            hw = r.bits.shape
            self.by_h.setdefault(hw[0], []).append((r, len(rows.setdefault(hw, []))))
            rows[hw].append(r)
        self.stacks = {hw: (np.stack([r.bits for r in rs]), np.array([r.n for r in rs])) for hw, rs in rows.items()}

    def ious(self, a: np.ndarray, picked: list[tuple[_Ref, int]]) -> np.ndarray:
        """`_iou(a, r.bits)` for every picked candidate at once. Each differs from `a` by at most a
        pixel per side, so all of them sit on one canvas; the intersection only counts inside `a`'s
        own area, and whatever hangs over its edge never overlaps anyway."""
        ah, aw = a.shape
        canvas = np.zeros((len(picked), ah + 3, aw + 3), bool)
        ns = np.zeros(len(picked), int)
        groups: dict[tuple[int, int], list[int]] = {}
        for i, (r, _) in enumerate(picked):
            groups.setdefault(r.bits.shape, []).append(i)
        for (bh, bw), idx in groups.items():
            stack, n = self.stacks[bh, bw]
            rows = [picked[i][1] for i in idx]
            canvas[idx, 1:1 + bh, 1:1 + bw] = stack[rows]
            ns[idx] = n[rows]
        na = int(np.count_nonzero(a))
        best = np.zeros(len(picked))
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                inter = (a & canvas[:, 1 - oy:1 - oy + ah, 1 - ox:1 - ox + aw]).sum(axis=(1, 2))
                union = na + ns - inter             # |A u B| = |A| + |B| - |A n B|
                np.maximum(best, np.divide(inter, union, out=np.zeros(len(picked)), where=union > 0), out=best)
        return best


def match_all(teacher: GlyphDB, glyphs: dict[str, Glyph], keyfreq: Counter, scale: float) -> Transfer:
    refs = _Refs(_refs(teacher, scale))
    pos_tol = max(1.5, 0.08 * (teacher.unit or 30) * scale)
    labels: dict[str, tuple[str, str]] = {}
    covered = 0
    for key, g in glyphs.items():
        gh, gw, gt = g.h, g.w, g.top_rel
        picked = [(r, row) for dh in (-1, 0, 1) for r, row in refs.by_h.get(gh + dh, ())
                  if abs(r.bits.shape[1] - gw) <= 1 and abs(r.top_rel - gt) <= pos_tol]
        if not picked:
            continue
        cands = [(v, r) for v, (r, _) in zip(refs.ious(g.bits, picked).tolist(), picked)]
        cands.sort(key=lambda c: -c[0])
        best_iou, best = cands[0]
        other = next((c[0] for c in cands[1:] if c[1].label != best.label), 0.0)
        if best_iou < IOU_MIN or best_iou - other < MARGIN:
            continue
        if any(c in FRAGILE for c in best.label) or any(
                c[1].label != best.label and set(c[1].label) & FRAGILE for c in cands[1:4]):
            continue
        labels[key] = (best.label, best.key)
        covered += keyfreq[key]
    return Transfer(scale, covered / max(1, sum(keyfreq.values())), labels)


def _modal_heights(heights: Counter, n: int = 3) -> list[int]:
    return [h for h, _ in heights.most_common(n)]


def find_scale(teacher: GlyphDB, glyphs: dict[str, Glyph], keyfreq: Counter) -> Transfer | None:
    """Try scales suggested by the most common glyph heights, refine the best one."""
    t_heights = Counter(s.bits.shape[0] for s in teacher.shapes.values()
                        for v in s.variants if trusted_label(v.votes)[0])
    n_heights = Counter(g.h for g in glyphs.values())
    if not t_heights or not n_heights:
        return None
    cands = {round(nh / th, 3) for th in _modal_heights(t_heights) for nh in _modal_heights(n_heights)}
    cands = {s for s in cands if 0.3 <= s <= 3.5}
    top = sorted(glyphs, key=lambda k: -keyfreq[k])[:80]      # quick evaluation on frequent glyphs
    sub = {k: glyphs[k] for k in top}
    subfreq = Counter({k: keyfreq[k] for k in top})
    scored = sorted(((match_all(teacher, sub, subfreq, s).coverage, s) for s in cands), reverse=True)
    if not scored or scored[0][0] == 0:
        return None
    best_s = scored[0][1]
    for step in np.linspace(-0.03, 0.03, 7):                 # refine around the best candidate
        s = round(best_s * (1 + step), 4)
        cov = match_all(teacher, sub, subfreq, s).coverage
        if cov > scored[0][0]:
            scored[0] = (cov, s)
    return match_all(teacher, glyphs, keyfreq, scored[0][1])


def apply(teacher: GlyphDB, target: GlyphDB, tr: Transfer, glyphs: dict[str, Glyph]) -> None:
    """Write transferred labels into the target DB and carry over gaps, words and sequences."""
    t2n: dict[str, list[str]] = {}
    for nkey, (label, tkey) in tr.labels.items():
        g = glyphs[nkey]
        target.add_prior(nkey, g.bits, g.top_rel, label, f"teacher:{tkey}")
        t2n.setdefault(tkey, []).append(nkey)
    uniq = {t: n[0] for t, n in t2n.items() if len(n) == 1}

    def mapped(keys: list[str]) -> list[str] | None:
        out = [uniq.get(k) for k in keys]
        return None if None in out else out

    for k, (lc, sc) in teacher.pair_gaps.items():
        ka, kb, gap = k.split("|")
        m = mapped([ka, kb])
        if m:
            e = target.pair_gaps[f"{m[0]}|{m[1]}|{round(int(gap) * tr.scale)}"]
            e[0] += lc
            e[1] += sc
    for k, votes in teacher.words.items():
        m = mapped(k.split("|"))
        if m:
            target.words.setdefault("|".join(m), Counter()).update(votes)
    for k, votes in teacher.sequences.items():
        m = mapped(k.split("|"))
        if m:
            target.sequences.setdefault("|".join(m), Counter()).update(votes)
    target._bearings = None
    target.parent = teacher.name
    if teacher.unit:
        target.unit = teacher.unit * tr.scale
    target.dirty = True
