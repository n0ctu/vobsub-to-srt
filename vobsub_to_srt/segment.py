"""Binarization (fill-colour isolation) and line/glyph segmentation."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .vobsub import Cue

_EIGHT = np.ones((3, 3), dtype=bool)


MAX_DENSITY = 0.5     # a colour filling more of its bounding box than this is a backdrop, not text
MAX_EXPOSURE = 0.15   # text fill and anti-alias rings hardly ever touch transparency; outlines do
RING_THIN = 1.6       # foreign contacts per pixel: a one-pixel ring has >= 2, a text stroke far less
RING_CONTACT = 0.35   # share of a ring's foreign contacts that are the outline (it hugs the outline)
MIN_SHARE = 0.1       # colours smaller than this fraction of the largest text colour are leftovers


def fill_mask(cue: Cue) -> np.ndarray:
    """The text pixels of a cue (the fill palette values, see fill_values), cropped to the ink."""
    keep = fill_values(cue)
    if not keep:
        return np.zeros((0, 0), bool)
    return crop(np.isin(cue.image, keep))


def fill_values(cue: Cue) -> list[int]:
    """The palette values that are text fill.

    A VobSub cue has up to four palette colours: background, fill, and usually an outline and an
    anti-alias ring between them. Only the fill is text. The outline is the colour that faces
    transparency; the fill and the ring both sit inside it. The ring is thin (a one-pixel line has
    two foreign neighbours per pixel, a text stroke far fewer) and hugs the outline; the fill is
    thick. Every qualifying colour is kept (two speaker colours in one cue), backdrop boxes are
    excluded by their density."""
    img = cue.image
    if img.size == 0:
        return []
    h, w = img.shape
    padded = np.pad(img, 1, constant_values=4)                  # 4 = image border
    opaque = [u for u in range(4) if cue.alpha[u] >= 8]
    stats: dict[int, dict] = {}
    for v in opaque:
        m = img == v
        n = int(m.sum())
        if n == 0:
            continue
        hist = np.zeros(5, np.int64)                            # contacts: 0 transparent, 1-3, 4 border
        for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
            nb = padded[dy:dy + h, dx:dx + w]
            nb = nb[m & (nb != v)]
            nb = np.where(np.isin(nb, opaque) | (nb == 4), nb, 0)
            hist += np.bincount(nb, minlength=5)
        rows, cols = np.nonzero(m.any(axis=1))[0], np.nonzero(m.any(axis=0))[0]
        bbox = (rows[-1] - rows[0] + 1) * (cols[-1] - cols[0] + 1)
        stats[v] = {"n": n, "hist": hist, "density": n / bbox}
    if not stats:
        return []
    cand = {v: st for v, st in stats.items() if st["density"] < MAX_DENSITY} or stats
    backdrop = [v for v in stats if v not in cand]       # a box behind the text acts as background
    for st in cand.values():
        exposed = st["hist"][0] + st["hist"][4] + sum(st["hist"][v] for v in backdrop)
        st["exposure"] = exposed / max(int(st["hist"].sum()), 1)
    outline = max(cand, key=lambda v: cand[v]["exposure"])
    inner = {v: st for v, st in cand.items() if v != outline and st["exposure"] < MAX_EXPOSURE}
    if not inner:
        # no outline layout: the fill itself faces transparency; take the least exposed colour
        keep = [min(cand, key=lambda v: cand[v]["exposure"])]
    else:
        thin = {v: int(st["hist"].sum()) / st["n"] for v, st in inner.items()}
        contact = {v: st["hist"][outline] / max(int(st["hist"].sum()), 1) for v, st in inner.items()}
        fills = [v for v in inner if thin[v] <= RING_THIN] or [min(inner, key=thin.get)]
        if len(fills) >= 2:              # a thick ring: the one hugging the outline is the ring
            low = [v for v in fills if contact[v] < RING_CONTACT]
            if low and len(low) < len(fills):
                fills = low
        largest = max(inner[v]["n"] for v in fills)
        keep = [v for v in fills if inner[v]["n"] >= MIN_SHARE * largest]
    return list(keep)


def crop(mask: np.ndarray) -> np.ndarray:
    """Crop to the ink bounding box (SPU display areas can be much larger than the text)."""
    rows, cols = np.nonzero(mask.any(axis=1))[0], np.nonzero(mask.any(axis=0))[0]
    if not len(rows):
        return np.zeros((0, 0), bool)
    return mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


@dataclass
class Glyph:
    x: int                 # left, in cue coordinates
    y: int                 # top, in cue coordinates
    bits: np.ndarray       # (h, w) bool
    top_rel: int = 0       # top relative to line baseline (negative = above)
    key: str = ""          # hash of (w, h, bits)
    underlined: bool = False

    @property
    def w(self) -> int:
        return self.bits.shape[1]

    @property
    def h(self) -> int:
        return self.bits.shape[0]

    @property
    def right(self) -> int:
        return self.x + self.w


@dataclass
class Line:
    y0: int
    y1: int
    glyphs: list[Glyph] = field(default_factory=list)
    baseline: int = 0
    gaps: list[int] = field(default_factory=list)   # gap after glyph i (len = len(glyphs)-1)
    underlines: list[tuple[int, int]] = field(default_factory=list)  # x ranges of removed underline strokes


def glyph_key(bits: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=12)
    h.update(bits.shape[0].to_bytes(2, "big") + bits.shape[1].to_bytes(2, "big"))
    h.update(np.packbits(bits).tobytes())
    return h.hexdigest()


def _row_bands(mask: np.ndarray) -> list[tuple[int, int]]:
    rows = mask.any(axis=1)
    bands, start = [], None
    for y, r in enumerate(rows):
        if r and start is None:
            start = y
        elif not r and start is not None:
            bands.append((start, y))
            start = None
    if start is not None:
        bands.append((start, len(rows)))
    return bands


def split_lines(mask: np.ndarray) -> list[tuple[int, int]]:
    """Row bands, with small bands (i-dots, umlauts, accents) merged into the closest neighbour."""
    bands = _row_bands(mask)
    if len(bands) <= 1:
        return bands
    big = max(b - a for a, b in bands)
    changed = True
    while changed and len(bands) > 1:
        changed = False
        for i, (a, b) in enumerate(bands):
            if b - a >= 0.55 * big:
                continue
            gap_up = a - bands[i - 1][1] if i > 0 else 10 ** 9
            gap_dn = bands[i + 1][0] - b if i + 1 < len(bands) else 10 ** 9
            if min(gap_up, gap_dn) > 0.5 * big:
                continue  # genuinely separate small line (e.g. "...")
            j = i + 1 if gap_dn <= gap_up else i - 1
            lo, hi = min(i, j), max(i, j)
            bands[lo:hi + 1] = [(bands[lo][0], bands[hi][1])]
            changed = True
            break
    return bands


def _merge_components(boxes: list[list[int]]) -> list[list[int]]:
    """boxes: [x0, x1, y0, y1, label]. Merge vertically separate parts that overlap in x
    (i/j dots, umlauts, accents, ':' ';' '!' '?'). Separation is tested against each group's
    tallest part (its body), so a second umlaut dot still merges after the first one did."""
    groups = [{"box": b[:4], "body": b[:4], "labels": b[4:]} for b in sorted(boxes, key=lambda b: b[0])]
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = groups[i], groups[j]
                ov = min(a["box"][1], b["box"][1]) - max(a["box"][0], b["box"][0])
                if ov <= 0:
                    continue
                narrow = min(a["box"][1] - a["box"][0], b["box"][1] - b["box"][0])
                ab, bb = a["body"], b["body"]
                vsep = ab[3] <= bb[2] or bb[3] <= ab[2]
                if vsep and ov >= 0.3 * narrow:
                    body = ab if ab[3] - ab[2] >= bb[3] - bb[2] else bb
                    box = [min(a["box"][0], b["box"][0]), max(a["box"][1], b["box"][1]),
                           min(a["box"][2], b["box"][2]), max(a["box"][3], b["box"][3])]
                    groups[i] = {"box": box, "body": body, "labels": a["labels"] + b["labels"]}
                    del groups[j]
                    merged = True
                    break
            if merged:
                break
    return [g["box"] + g["labels"] for g in groups]


def _strip_underlines(band: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Remove long thin horizontal strokes in the lower part of a line band (underlines).
    They would otherwise fuse with descenders into one giant component."""
    h = band.shape[0]
    min_len = max(8, int(1.2 * h))
    max_thick = max(2, int(0.15 * h))
    hits: dict[int, list[tuple[int, int]]] = {}
    for y in range(h // 2, h):
        row = band[y]
        if row.sum() < min_len:
            continue
        d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
        starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        runs = [(a, b) for a, b in zip(starts, ends) if b - a >= min_len]
        if runs:
            hits[y] = runs
    if not hits:
        return band, []
    ys = sorted(hits)
    groups, cur = [], [ys[0]]
    for y in ys[1:]:
        if y == cur[-1] + 1:
            cur.append(y)
        else:
            groups.append(cur)
            cur = [y]
    groups.append(cur)
    out = band.copy()
    ranges: list[tuple[int, int]] = []
    for g in groups:
        if len(g) > max_thick:
            continue       # a thick bar is part of a glyph, not an underline
        for y in g:
            for a, b in hits[y]:
                out[y, a:b] = False
                ranges.append((int(a), int(b)))
    ranges.sort()
    merged: list[tuple[int, int]] = []
    for a, b in ranges:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return out, merged


def segment(mask: np.ndarray) -> list[Line]:
    lines: list[Line] = []
    for y0, y1 in split_lines(mask):
        band, underlines = _strip_underlines(mask[y0:y1])
        lab, n = ndimage.label(band, structure=_EIGHT)
        if n == 0:
            continue
        slices = ndimage.find_objects(lab)
        boxes = [[s[1].start, s[1].stop, s[0].start, s[0].stop, k + 1] for k, s in enumerate(slices)]
        boxes = _merge_components(boxes)
        line = Line(y0, y1, underlines=underlines)
        for x0, x1, gy0, gy1, *labels in boxes:
            sub = np.isin(lab[gy0:gy1, x0:x1], labels)
            g = Glyph(x0, y0 + gy0, sub)
            g.underlined = any(min(x1, b) - max(x0, a) >= 0.5 * (x1 - x0) for a, b in underlines)
            g.key = glyph_key(sub)
            line.glyphs.append(g)
        line.glyphs.sort(key=lambda g: g.x)
        bottoms = [g.y + g.h for g in line.glyphs]
        vals, counts = np.unique(bottoms, return_counts=True)
        # baseline: most common bottom among the lower half of glyph bottoms (ignores ' - ^ etc.)
        line.baseline = int(vals[np.argmax(counts)])
        for g in line.glyphs:
            g.top_rel = g.y - line.baseline
        line.gaps = [b.x - a.right for a, b in zip(line.glyphs, line.glyphs[1:])]
        lines.append(line)
    return lines


_SLANT_CACHE: dict[tuple, str] = {}
ITALIC_SHEAR = 0.25


def word_slant_class(glyphs: list[Glyph]) -> str:
    """'i' (clearly italic), 'u' (clearly upright) or '?' from vertical-stroke sharpness under shear."""
    sig = tuple((g.key, g.x - glyphs[0].x, g.y - glyphs[0].y) for g in glyphs)
    hit = _SLANT_CACHE.get(sig)
    if hit is None:
        hit = _SLANT_CACHE[sig] = _slant_class(glyphs)
    return hit


def _slant_class(glyphs: list[Glyph]) -> str:
    ys = np.concatenate([np.nonzero(g.bits)[0] + g.y for g in glyphs])
    xs = np.concatenate([np.nonzero(g.bits)[1] + g.x for g in glyphs])
    ys = ys - ys.max()
    scores = {}
    for s in np.linspace(0.0, 0.35, 15):
        x = np.round(xs + s * ys).astype(int)
        c = np.bincount(x - x.min()).astype(float)
        scores[round(float(s), 3)] = float((c * c).sum())
    best_s = max(scores, key=scores.get)
    s0 = scores[0.0]
    s_it = scores[min(scores, key=lambda k: abs(k - ITALIC_SHEAR))]
    if 0.15 <= best_s <= 0.3 and scores[best_s] >= 1.08 * s0:
        return "i"
    if best_s <= 0.05 and s0 >= 1.08 * s_it:
        return "u"
    return "?"


def italic_votes(all_lines: list[list[Line]], gap_threshold: float) -> dict[str, list[int]]:
    """Per glyph key: [upright, italic] word counts over a whole file.
    Pass 1: only words with an unambiguous slant vote. Pass 2: every word is classified by its
    confidently known glyphs, else by its neighbouring words, else by its own slant; all its glyphs vote."""
    words_by_line: list[list[list[Glyph]]] = []
    small: set[str] = set()      # punctuation-sized glyphs: they carry no slant and never decide
    for lines in all_lines:
        for l in lines:
            words: list[list[Glyph]] = [[]]
            for i, g in enumerate(l.glyphs):
                if i and l.gaps[i - 1] > gap_threshold:
                    words.append([])
                words[-1].append(g)
                if g.h < 0.4 * (l.y1 - l.y0):
                    small.add(g.key)
            words_by_line.append(words)
    first: dict[str, list[int]] = {}
    cls_cache: list[list[str]] = []
    for words in words_by_line:
        row = []
        for w in words:
            c = word_slant_class(w)
            row.append(c)
            if c != "?":
                for g in w:
                    if g.key not in small:
                        first.setdefault(g.key, [0, 0])[1 if c == "i" else 0] += 1
        cls_cache.append(row)

    def decided(key: str) -> str | None:
        u, i = first.get(key, (0, 0))
        if u + i >= 2 and max(u, i) >= 0.75 * (u + i):
            return "i" if i > u else "u"
        return None

    votes: dict[str, list[int]] = {}
    for words, row in zip(words_by_line, cls_cache):
        final: list[str] = []
        for w, c in zip(words, row):
            ks = [decided(g.key) for g in w]
            ks = [k for k in ks if k]
            if ks:
                final.append("i" if ks.count("i") * 2 > len(ks) else "u")
            else:
                final.append(c)
        for k, c in enumerate(final):   # unresolved words follow their neighbours
            if c == "?":
                nb = [final[j] for j in (k - 1, k + 1) if 0 <= j < len(final) and final[j] != "?"]
                final[k] = nb[0] if nb and all(x == nb[0] for x in nb) else "?"
        for w, c in zip(words, final):
            if c == "?":
                continue
            for g in w:
                votes.setdefault(g.key, [0, 0])[1 if c == "i" else 0] += 1
    return votes


_STROKE_CACHE: dict[str, np.ndarray] = {}


def _stroke_samples(g: Glyph) -> np.ndarray:
    """Local stroke widths: 2x distance-transform values on the medial ridge of the glyph."""
    hit = _STROKE_CACHE.get(g.key)
    if hit is None:
        dt = ndimage.distance_transform_edt(np.pad(g.bits, 1))
        ridge = (dt == ndimage.maximum_filter(dt, size=3)) & (dt > 0)
        hit = _STROKE_CACHE[g.key] = 2 * dt[ridge]
    return hit


def bold_votes(all_lines: list[list[Line]], gap_threshold: float,
               bold_ratio: float = 1.25, regular_ratio: float = 1.1) -> dict[str, list[int]]:
    """Per glyph key: [regular, bold] word counts. A word is bold if its stroke width is well above the
    file's median word stroke (subtitles are mostly regular weight), regular if close to it."""
    words: list[list[Glyph]] = []
    min_ink: list[float] = []
    for lines in all_lines:
        for l in lines:
            cur: list[Glyph] = []
            for i, g in enumerate(l.glyphs):
                if i and l.gaps[i - 1] > gap_threshold:
                    words.append(cur)
                    cur = []
                cur.append(g)
            if cur:
                words.append(cur)
            min_ink += [0.1 * (l.y1 - l.y0) ** 2] * (len(words) - len(min_ink))
    strokes = []
    for w, mi in zip(words, min_ink):
        if sum(int(g.bits.sum()) for g in w) < mi:
            strokes.append(None)       # too little ink (punctuation, tiny words)
        else:
            strokes.append(float(np.median(np.concatenate([_stroke_samples(g) for g in w]))))
    valid = [x for x in strokes if x is not None]
    if not valid:
        return {}
    med = float(np.median(valid))
    votes: dict[str, list[int]] = {}
    for w, sw in zip(words, strokes):
        if sw is None:
            continue
        if sw >= bold_ratio * med:
            idx = 1
        elif sw <= regular_ratio * med:
            idx = 0
        else:
            continue
        for g in w:
            votes.setdefault(g.key, [0, 0])[idx] += 1
    return votes
