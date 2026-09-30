"""nOCR-style recognition of segmented lines against a GlyphDB."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from collections import Counter

from .glyphdb import OVERRIDE_SHARE, OVERRIDE_VOTES, GlyphDB, Variant, trusted_label
from .styling import majority_style, render_styled
from .segment import Glyph, Line

T_ACCEPT = 0.08     # max pixel-diff ratio for a near match
T_MARGIN = 0.06     # required advantage over best candidate with a different label
AMBIG_IL = {"I", "l", "|"}      # glyphs that can be pixel-identical in sans-serif fonts


def confusable(labels: set[str]) -> bool:
    """True if all labels are the same string up to I/l/| substitutions ("ll" vs "II", "Il" vs "lI")."""
    labels = {l for l in labels if l}
    if len(labels) < 2 or len({len(l) for l in labels}) != 1:
        return False
    for chars in zip(*labels):
        if len(set(chars)) > 1 and not set(chars) <= AMBIG_IL:
            return False
    return True


@dataclass
class GlyphResult:
    text: str | None          # None = unknown/uncertain
    style: str                # subset of "biu"
    reason: str = ""          # why uncertain
    via: str = "exact"        # exact | near | seq | word | context


@dataclass
class LineResult:
    items: list[GlyphResult]
    spaces: list[bool | None]           # space before item i+1 (len = len(items)-1)
    glyph_spans: list[tuple[int, int]]  # glyph index range covered by each item

    @property
    def ok(self) -> bool:
        return all(i.text is not None for i in self.items) and all(s is not None for s in self.spaces)


@dataclass
class CueResult:
    lines: list[LineResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(l.ok for l in self.lines)

    def problems(self) -> list[tuple[int, int, str]]:
        out = []
        for li, l in enumerate(self.lines):
            for ci, it in enumerate(l.items):
                if it.text is None:
                    out.append((li, ci, it.reason or "unknown glyph"))
            for ci, s in enumerate(l.spaces):
                if s is None:
                    out.append((li, ci, "uncertain space"))
        return out

    def text(self, placeholder: str = "�") -> str:
        return "\n".join(render_line(l, placeholder) for l in self.lines)


def render_line(l: LineResult, placeholder: str = "\ufffd") -> str:
    """Join items into words; each word gets the majority style of its characters."""
    words: list[list[GlyphResult]] = [[]]
    for i, it in enumerate(l.items):
        if i > 0 and l.spaces[i - 1] is not False:
            words.append([])
        words[-1].append(it)
    chars: list[tuple[str, str]] = []
    for k, w in enumerate(words):
        if k:
            chars.append((" ", ""))
        styles = [it.style for it in w for _ in (it.text or "x")]
        st = majority_style(styles)
        chars += [(c, st) for it in w for c in (it.text if it.text is not None else placeholder)]
    return render_styled(chars)


def _decide(v: Variant) -> tuple[str | None, str]:
    """Label to read this glyph as. VLM votes decide (>= 2 votes, >= 2/3 majority); a teacher prior
    is used while no VLM read exists and counts as one vote once there are VLM reads."""
    votes = Counter(v.votes)
    if v.prior:
        if not +votes:
            return v.prior, ""
        votes[v.prior] += 1
    if confusable({k for k, n in votes.items() if n > 0}):
        return None, "ambiguous I/l"
    return trusted_label(votes)


def confirmed(v: Variant | None) -> str | None:
    """Label confirmed by VLM reads strongly enough to override a contradicting VLM read."""
    if v is None or confusable({k for k, n in v.votes.items() if n > 0}):
        return None
    total = sum(v.votes.values())
    if not total:
        return None
    top, n = v.votes.most_common(1)[0]
    if n < OVERRIDE_VOTES or n < OVERRIDE_SHARE * total:
        return None
    return top or None


def _diff_ratio(a: np.ndarray, b: np.ndarray) -> float:
    h, w = max(a.shape[0], b.shape[0]) + 2, max(a.shape[1], b.shape[1]) + 2
    A = np.zeros((h, w), bool)
    A[1:1 + a.shape[0], 1:1 + a.shape[1]] = a
    na, nb = int(a.sum()), int(b.sum())
    best = None
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            if dy + b.shape[0] > h or dx + b.shape[1] > w:
                continue
            B = np.zeros((h, w), bool)
            B[dy:dy + b.shape[0], dx:dx + b.shape[1]] = b
            d = int((A ^ B).sum())
            best = d if best is None else min(best, d)
    return best / max(1.0, (na + nb) / 2)


def near_match(db: GlyphDB, g: Glyph) -> tuple[str | None, Variant | None, float]:
    """Best near match among confirmed shapes of the SAME height (width +-1) at the same position.
    Within one raster a height difference means a different glyph (I vs l), so it is never tolerated.
    Returns (key, variant, ratio) or Nones. Callers treat the result as a suggestion only."""
    cands: list[tuple[float, str, Variant, str]] = []
    for dw in (-1, 0, 1):
        for key in db.by_size.get((g.h, g.w + dw), ()):
            shape = db.shapes[key]
            v = shape.variant(g.top_rel, db.pos_tol)
            if v is None:
                continue
            label, _ = _decide(v)
            if label is None or label == "":
                continue
            cands.append((_diff_ratio(g.bits, shape.bits), key, v, label))
    if not cands:
        return None, None, 1.0
    cands.sort(key=lambda c: c[0])
    best = cands[0]
    if best[0] > T_ACCEPT:
        return None, None, best[0]
    other = next((c for c in cands[1:] if c[3] != best[3]), None)
    if other is not None and other[0] - best[0] < T_MARGIN:
        return None, None, best[0]
    return best[1], best[2], best[0]


def _style(v: Variant | None, g: Glyph) -> str:
    st = v.style() if v else ""
    return st + "u" if g.underlined else st


def recognize_line(db: GlyphDB, line: Line, learn_near: bool = True, lexicon=None) -> LineResult:
    gl = line.glyphs
    items: list[GlyphResult] = []
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(gl):
        # multi-segment characters (e.g. '"', '%') - longest first
        seq_hit = None
        for k in (3, 2):
            if i + k <= len(gl):
                key = "|".join(g.key for g in gl[i:i + k])
                votes = db.sequences.get(key)
                label = trusted_label(votes)[0] if votes else None
                max_gap = 0.25 * (line.y1 - line.y0)
                if label is not None and all(line.gaps[i + t] <= max_gap for t in range(k - 1)):
                    seq_hit = (k, label)
                    break
        if seq_hit:
            k, label = seq_hit
            sts = [_style(db.lookup(g.key, g.top_rel), g) for g in gl[i:i + k]]
            items.append(GlyphResult(label, majority_style(sts), via="seq"))
            spans.append((i, i + k))
            i += k
            continue
        g = gl[i]
        v = db.lookup(g.key, g.top_rel)
        via = "exact"
        if v is None:
            key, nv, ratio = near_match(db, g)
            if nv is not None and learn_near:
                # one unconfirmed vote: the VLM still has to agree before this shape is trusted
                label, _ = _decide(nv)
                db.add_vote(g.key, g.bits, g.top_rel, label, nv.style(), derived_from=key)
                v = db.lookup(g.key, g.top_rel)
                via = "near"
        if v is None:
            items.append(GlyphResult(None, _style(None, g), "unknown glyph", via="none"))
        else:
            label, reason = _decide(v)
            if label == "":
                label, reason = None, "fragment of multi-part char"
            items.append(GlyphResult(label, _style(v, g), reason, via=via))
            if reason == "ambiguous I/l":
                items[-1].via = "ambig"
        spans.append((i, i + 1))
        i += 1

    spaces: list[bool | None] = []
    for a, b in zip(spans, spans[1:]):
        ga, gb = gl[a[1] - 1], gl[b[0]]
        gap = gb.x - ga.right
        ital = "i" in items[len(spaces)].style or "i" in items[len(spaces) + 1].style
        spaces.append(db.classify_gap(ga.key, gb.key, gap, ital))

    # two adjacent apostrophes without a space are a double quote
    k = 0
    while k + 1 < len(items):
        if items[k].text in ("'", "’") and items[k + 1].text in ("'", "’") and spaces[k] is False:
            items[k] = GlyphResult('"', items[k].style, via="seq")
            spans[k] = (spans[k][0], spans[k + 1][1])
            del items[k + 1], spans[k + 1], spaces[k]
        k += 1
    res = LineResult(items, spaces, spans)
    _resolve_words(db, line, res, lexicon)
    if lexicon is not None and hasattr(lexicon, "repair"):
        _repair_words(res, lexicon)
    return res


def _repair_words(res: LineResult, lexicon) -> None:
    for a, b in _word_ranges(res):
        texts = [res.items[k].text for k in range(a, b)]
        if None in texts:
            continue
        fixed = lexicon.repair(texts)
        if not fixed:
            continue
        pos = 0
        for k, t in zip(range(a, b), texts):
            if fixed[pos:pos + len(t)] != t:
                res.items[k].text, res.items[k].via = fixed[pos:pos + len(t)], "repair"
            pos += len(t)


def _word_ranges(res: LineResult) -> list[tuple[int, int]]:
    out, start = [], 0
    for i, s in enumerate(res.spaces):
        if s is not False:
            out.append((start, i + 1))
            start = i + 1
    out.append((start, len(res.items)))
    return out


def _resolve_words(db: GlyphDB, line: Line, res: LineResult, lexicon=None) -> None:
    """Resolve ambiguous I/l words: (1) word memory of this font DB, (2) optional lexicon gate.
    Anything still ambiguous stays uncertain (-> VLM). No case/position heuristics: texts may be
    all-caps, German, English, or any other language."""
    if not any(it.via == "ambig" for it in res.items):
        return
    for a, b in _word_ranges(res):
        amb = [k for k in range(a, b) if res.items[k].via == "ambig"]
        if not amb:
            continue
        gkeys = [line.glyphs[j].key for k in range(a, b) for j in range(*res.glyph_spans[k])]
        mem = db.words.get("|".join(gkeys))
        word = trusted_label(mem)[0] if mem else None
        via = "word"
        if word is None and lexicon is not None:
            word = lexicon.resolve([_candidates(db, line, res, k) for k in range(a, b)])
            via = "lexicon"
        if word is None:
            continue
        pos = 0
        parts = []
        for k in range(a, b):
            n = len(res.items[k].text or _candidates(db, line, res, k)[0])
            parts.append(word[pos:pos + n])
            pos += n
        if pos != len(word):
            continue   # memory/lexicon word does not fit the glyph segmentation
        for k, part in zip(range(a, b), parts):
            if res.items[k].via == "ambig":
                res.items[k].text, res.items[k].reason, res.items[k].via = part, "", via


def _candidates(db: GlyphDB, line: Line, res: LineResult, k: int) -> list[str]:
    """Possible texts for item k: its label, or all voted labels of an ambiguous glyph."""
    it = res.items[k]
    if it.via != "ambig":
        return [it.text or "\ufffd"]
    g = line.glyphs[res.glyph_spans[k][0]]
    v = db.lookup(g.key, g.top_rel)
    return sorted(l for l, n in v.votes.items() if n > 0 and l)


def recognize(db: GlyphDB, lines: list[Line], learn_near: bool = True, lexicon=None) -> CueResult:
    return CueResult([recognize_line(db, l, learn_near, lexicon) for l in lines])
