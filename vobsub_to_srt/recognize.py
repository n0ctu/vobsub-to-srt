"""nOCR-style recognition of segmented lines against a GlyphDB."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from collections import Counter

import re as _re

from .glyphdb import (CLUSTER_TOL, OVERRIDE_SHARE, OVERRIDE_VOTES, PROTO_MARGIN, PROTO_STRICT_TOL, PROTO_TOL, TOLERANT_MARGIN,
                      GlyphDB, Variant, diff_ratio, is_strict, tol_for, topology, trusted_label)
from .styling import inherit_punct_styles, majority_style, render_styled
from .segment import Glyph, Line

T_ACCEPT = CLUSTER_TOL   # max pixel-diff ratio for a near match
T_MARGIN = 0.06          # required advantage over best candidate with a different label
_TAG_RE = _re.compile(r"</?\s*([a-zA-Z]+)[^>]*>")


def strip_tags(text: str) -> str:
    return _TAG_RE.sub("", text)


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
    variant: Variant | None = None   # the memory variant this glyph was read with (exact or near)


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

    @property
    def letters_ok(self) -> bool:
        return all(i.text is not None for l in self.lines for i in l.items)

    def fill_spaces(self, reference: str) -> bool:
        """Every character is read but some word breaks are uncertain: take those from a reference
        reading (the vision model's text) whose characters agree line by line. The memory keeps
        the breaks it is sure of. True if the result is complete."""
        if not self.letters_ok:
            return False
        ref_lines = [l for l in strip_tags(reference).split("\n") if l.strip()]
        if len(ref_lines) != len(self.lines):
            return False
        for lr, ref in zip(self.lines, ref_lines):
            if ref.replace(" ", "") != "".join(it.text for it in lr.items):
                return False
            # char index at which each item starts, and the reference's breaks (space before index)
            breaks = set()
            n = 0
            for ch in ref:
                if ch == " ":
                    breaks.add(n)
                else:
                    n += 1
            pos = 0
            for k, it in enumerate(lr.items[:-1]):
                pos += len(it.text)
                if lr.spaces[k] is None:
                    lr.spaces[k] = pos in breaks
        return self.ok

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
    texts = ["".join(it.text if it.text is not None else placeholder for it in w) for w in words]
    wstyles = inherit_punct_styles([(txt, majority_style([it.style for it in w for _ in (it.text or "x")]))
                                    for txt, w in zip(texts, words)])
    chars: list[tuple[str, str]] = []
    for k, (txt, st) in enumerate(zip(texts, wstyles)):
        if k:
            chars.append((" ", ""))
        chars += [(c, st) for c in txt]
    return render_styled(chars)


def _decide(v: Variant, db: GlyphDB | None = None) -> tuple[str | None, str]:
    """Label to read this glyph as. VLM votes decide (>= 2 votes, >= 2/3 majority); a teacher prior
    is used while no VLM read exists and counts as one vote once there are VLM reads. In a font
    whose I and l are drawn alike (`db.il_identical`), a glyph voted only I or only l is still
    ambiguous: the word decides (lexicon), not the two cues that happened to teach it."""
    votes = Counter(v.votes)
    if v.prior:
        if not +votes:
            return v.prior, ""
        votes[v.prior] += 1
    labels = {k for k, n in votes.items() if n > 0}
    if confusable(labels):
        return None, "ambiguous I/l"
    label, reason = trusted_label(votes)
    if label in ("I", "l") and db is not None and db.il_identical:
        return None, "ambiguous I/l"
    return label, reason


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


_diff_ratio = diff_ratio


TOLERANT_MIN_H = 8        # glyph height (px) below which the edge-tolerant stage is not used
TOLERANT_MIN_INK = 25     # ... and minimum ink pixels


def _geo_disagree(file_votes: list | None, cluster_votes: list) -> bool:
    """Both the glyph's word geometry in this file and the cluster have an opinion, and they differ."""
    if not file_votes or file_votes[0] == file_votes[1] or cluster_votes[0] == cluster_votes[1]:
        return False
    return (file_votes[1] > file_votes[0]) != (cluster_votes[1] > cluster_votes[0])


def near_match(db: GlyphDB, g: Glyph) -> tuple[str | None, Variant | None, float]:
    """Best near match among confirmed shapes (any cluster member) of the same size +-1 px at the
    same position. A height difference is only bridged for labels whose identity does not hang on
    one pixel row (never for I/l, i, 1, punctuation). Returns (key, variant, ratio) or Nones."""
    cands: list[tuple[float, str, Variant, str]] = []
    for r, key, dh in db.near_candidates(g.bits):
        v = db.lookup(key, g.top_rel)
        if v is None:
            continue
        label, _ = _decide(v, db)
        if label is None or label == "" or (dh and is_strict(label)):
            continue
        if r <= tol_for(label):
            cands.append((r, key, v, label))
    if not cands and db.protos:
        # stage 1b: the cluster prototypes (median + stability mask, learned from every sample of
        # the letter in earlier files). Disagreements count on stable pixels only, so a jittered
        # variant scores ~0 and a different letter differs where the font is stable. Same decision
        # rules as stage 2 below; strict letters are allowed, the mask sees their one-pixel rows.
        topo = topology(g.key, g.bits)
        fi = db.file_geo[0].get(g.key)
        fb = db.file_geo[1].get(g.key)
        ordered = []
        for d, canon in db.proto_candidates(g.bits):
            v = db.lookup(canon, g.top_rel)
            if v is not None and topology(canon, db.shapes[canon].bits) == topo:
                ordered.append((d, canon, v, _decide(v, db)[0]))
        if ordered:
            d, canon, v, label = ordered[0]
            tol = PROTO_TOL if label and not is_strict(label) else PROTO_STRICT_TOL
            if d <= tol:
                if label is None:
                    if confusable({k for k, n in v.votes.items() if n > 0} | ({"I", "l"} if db.il_identical else set())):
                        return canon, v, d              # the font's I/l cluster: the word decides
                    return None, None, d                # unconfirmed cluster: wait for the VLM
                if _geo_disagree(fi, v.geo_italic) or _geo_disagree(fb, v.geo_bold):
                    return None, None, d
                other = next((c for c in ordered if c[3] != label and not (c[3] in AMBIG_IL and label in AMBIG_IL)), None)
                if other is not None and other[0] - d < PROTO_MARGIN:
                    return None, None, d
                return canon, v, d
    if not cands and db.tolerant and g.h >= TOLERANT_MIN_H and int(g.bits.sum()) >= TOLERANT_MIN_INK:
        # stage 2 (rescaled tracks only): the edge-tolerant difference. Jitter moves edge pixels
        # by one; the pixel difference of a small letter then exceeds the tolerance although the
        # glyph is the same. One pixel of tolerance also hides a slant or a weight difference,
        # so a cluster is only accepted if its italic/bold votes agree with this glyph's own word
        # geometry in the file; punctuation-sized glyphs are excluded (too few pixels).
        # Candidates nearest first; a candidate whose topology (components, holes) differs is a
        # different letter (l vs !, c vs e) and is skipped. The nearest compatible one decides:
        # a confirmed, non-strict label with agreeing slant/weight is read; an ambiguous I/l
        # cluster is joined (same height, pixel difference not far off) so the lexicon decides;
        # anything else (unconfirmed, strict letter, disagreeing geometry) makes the stage abstain
        # rather than fall through to a runner-up ("ti!!" is how a runner-up reads "till").
        fi = db.file_geo[0].get(g.key)
        fb = db.file_geo[1].get(g.key)
        topo = topology(g.key, g.bits)
        ordered = []
        for r, key, dh in db.tolerant_candidates(g.bits):
            v = db.lookup(key, g.top_rel)
            if v is not None and topology(key, db.shapes[key].bits) == topo:
                ordered.append((r, key, v, _decide(v, db)[0], dh))
        for r, key, v, label, dh in ordered:
            if label is None and confusable({k for k, n in v.votes.items() if n > 0} | ({"I", "l"} if db.il_identical else set())):
                # joining the font's I/l cluster: the word decides the letter anyway. One pixel of
                # height is jitter when the font draws I and l alike; otherwise it may be the
                # very difference between them and the join needs the same height.
                if (dh == 0 or db.il_identical) and diff_ratio(g.bits, db.shapes[key].bits) <= 0.35:
                    return key, v, r
                return None, None, r
            if not label or is_strict(label) or _geo_disagree(fi, v.geo_italic) or _geo_disagree(fb, v.geo_bold):
                return None, None, r
            other = next((c for c in ordered if c[3] != label), None)
            if other is not None and other[0] - r < TOLERANT_MARGIN:
                return None, None, r
            return key, v, r
        return None, None, 1.0
    if not cands:
        return None, None, 1.0
    cands.sort(key=lambda c: c[0])
    best = cands[0]
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
                key = db.seq_key(gl[i:i + k])
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
            if nv is not None:
                # a jittered variant of a known shape shares its cluster's votes. Learning runs
                # store it as a member; read-only runs (arbitration against the vision model,
                # the final re-read) use the match without storing anything.
                if learn_near:
                    db.join(g.key, g.bits, g.top_rel, key)
                    v = db.lookup(g.key, g.top_rel)
                else:
                    v = nv
                via = "near"
        if v is None:
            items.append(GlyphResult(None, _style(None, g), "unknown glyph", via="none"))
        else:
            label, reason = _decide(v, db)
            if label == "":
                label, reason = None, "fragment of multi-part char"
            items.append(GlyphResult(label, _style(v, g), reason, via=via, variant=v))
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
    v = it.variant or db.lookup(g.key, g.top_rel)     # a near match read-only is not stored
    if v is None:
        return ["\ufffd"]
    labels = sorted(l for l, n in v.votes.items() if n > 0 and l)
    if labels and set(labels) <= {"I", "l"} and db.il_identical:
        return ["I", "l"]
    return labels


def recognize(db: GlyphDB, lines: list[Line], learn_near: bool = True, lexicon=None) -> CueResult:
    return CueResult([recognize_line(db, l, learn_near, lexicon) for l in lines])
