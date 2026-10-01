"""Align VLM transcriptions to glyph segments and learn glyph labels from them."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .glyphdb import GlyphDB, trusted_label
from .recognize import AMBIG_IL, _decide, confirmed, confusable
from .segment import Line
from .styling import inherit_punct_styles, STYLE_ORDER, majority_style, parse_styled, render_styled  # noqa: F401

INF = float("inf")
_TAG = re.compile(r"</?\s*([a-zA-Z]+)[^>]*>")
MAX_COST_PER_CHAR = 0.6
CONFLICT_COST = 6.0          # VLM char contradicts a confirmed glyph label
PRIOR_CONFLICT_COST = 1.0    # VLM char contradicts a teacher prior (weak anchor)
QUOTES = {"'", '"', "`", "\u00b4"}
PUNCT = QUOTES | set(".,:;!?-_\u2026\u201c\u201d\u201e\u2018\u2019")   # characters drawn as small parts
COMBINE = {"''": '"', "’’": "”", "‘‘": "“", ",,": "„", "...": "…"}   # glyph runs that form one char
MAX_CONFLICT_SHARE = 0.2     # more conflicts than this per line = misalignment, not VLM misreads


def strip_tags(text: str) -> str:
    return _TAG.sub("", text)


def otsu_threshold(gaps: list[int]) -> float:
    g = np.array([x for x in gaps if x >= 0])
    if len(g) < 2:
        return 12.0
    vals = np.unique(g)
    best_t, best_var = float(vals[0]), -1.0
    for t in vals[:-1]:
        lo, hi = g[g <= t], g[g > t]
        w0, w1 = len(lo) / len(g), len(hi) / len(g)
        var = w0 * w1 * (lo.mean() - hi.mean()) ** 2
        if var > best_var:
            best_var, best_t = var, float(t) + 0.5
    return best_t


@dataclass
class Mapping:
    segs: tuple[int, int]     # glyph index range [a, b)
    chars: tuple[int, int]    # char index range [a, b) into the no-space char list
    text: str
    style: str
    db_label: str | None = None   # confirmed DB label of the (single) glyph, if any
    alts: list[str] | None = None  # readings of a pixel-identical glyph (I/l), for the lexicon gate


@dataclass
class Alignment:
    mappings: list[Mapping]
    cost: float
    spaces_before: set[int]   # glyph indices that start a new word
    conflicts: int = 0        # mappings where the VLM contradicts an override-strong glyph label
    soft_conflicts: int = 0   # ... a VLM-confirmed label that is not (yet) override-strong


def align_line(db: GlyphDB, line: Line, styled: list[tuple[str, bool]], gap_threshold: float) -> Alignment | None:
    gl = line.glyphs
    m = len(gl)
    chars = [c for c, _ in styled if c != " "]
    sty = [st for c, st in styled if c != " "]
    ital = ["i" in st for st in sty]
    n = len(chars)
    if m == 0 or n == 0:
        return None
    word_start = set()
    j = 0
    for c, _ in styled:
        if c == " ":
            word_start.add(j)
        else:
            j += 1
    widths = db.char_widths()
    avg_w = max(1.0, sum(g.w for g in gl) / n)
    known: list[str | None] = []      # label the DB would read (incl. teacher priors)
    strong: list[str | None] = []     # label confirmed by VLM reads (may override the VLM)
    unanimous: list[str | None] = []  # single label among all VLM reads so far (maybe only one read)
    trusted: list[str | None] = []    # label confirmed by VLM reads (>= 2, 2/3), maybe not override-strong
    alts: list[list[str] | None] = []
    for g in gl:
        v = db.lookup(g.key, g.top_rel)
        known.append(_decide(v)[0] if v else None)
        strong.append(confirmed(v))
        trusted.append((trusted_label(v.votes)[0] or None) if v and not confusable(
            {k for k, n in v.votes.items() if n > 0}) else None)
        labels = {k for k, n in v.votes.items() if n > 0} if v else set()
        unanimous.append(next(iter(labels)) if len(labels) == 1 else None)
        alts.append(sorted(labels) if confusable(labels) else None)

    def exp_w(text: str) -> float:
        return sum(widths.get(c, avg_w) for c in text)

    def boundary(i: int, j: int) -> float:
        if i == 0:
            return 0.0
        gap = gl[i].x - gl[i - 1].right
        want = j in word_start
        ital_ctx = ital[j - 1] if j > 0 else False
        pred = db.classify_gap(gl[i - 1].key, gl[i].key, gap, ital_ctx)
        if pred is not None:
            return 0.0 if pred == want else 4.0
        pred = gap > gap_threshold
        if pred == want:
            return 0.0
        return 0.5 if abs(gap - gap_threshold) <= 2 else 2.0

    def one_seg(i: int, j: int, k: int) -> float:
        if any((j + t) in word_start for t in range(1, k)):
            return INF
        text = "".join(chars[j:j + k])
        lab = known[i]
        if k > 1 and lab and len(lab) == 1:
            return INF    # a glyph known as one letter (even by prior) cannot be two ("e" -> "te")
        if lab:
            if lab == text:
                return 0.0
            if lab in AMBIG_IL and text in AMBIG_IL:
                return 0.5        # I vs l: often pixel-identical, not an alignment signal
            return CONFLICT_COST if trusted[i] else PRIOR_CONFLICT_COST
        return 0.5 * abs(gl[i].w - exp_w(text)) / avg_w + 1.5 * (k - 1)

    def multi_seg(i: int, s: int, j: int) -> float:
        for t in range(i + 1, i + s):
            gap = gl[t].x - gl[t - 1].right
            if gap > gap_threshold or db.classify_gap(gl[t - 1].key, gl[t].key, gap, ital[j]) is True:
                return INF
        # A glyph with a trusted letter label is never a part of a different character: merging it
        # means the VLM dropped a letter ("Verdopeln" over p p, "Lächer" over h l). Allowed only
        # if the glyphs together spell the char (sequence rule, '' -> ", ... -> …). Punctuation
        # parts (ticks, dots) are different: a double quote is drawn as two ticks and often read
        # as one, so parts that are unknown or punctuation may form any punctuation character.
        letters = any(trusted[t] and trusted[t] not in PUNCT for t in range(i, i + s))
        if letters or all(known[i:i + s]):
            seq = db.sequences.get(db.seq_key(gl[i:i + s]))
            together = "".join(known[t] or "\0" for t in range(i, i + s))
            combined = (trusted_label(seq)[0] if seq else None) or COMBINE.get(together, together)
            if combined != chars[j] and (letters or chars[j] not in PUNCT):
                return INF
        span = gl[i + s - 1].right - gl[i].x
        c = 0.5 * abs(span - exp_w(chars[j])) / avg_w + 1.5 * (s - 1)
        for t in range(i, i + s):
            if strong[t]:
                c += 4.0      # swallowing a confirmed glyph: the VLM most likely dropped a character
            elif known[t]:
                c += 1.0
        return c

    if m == n:
        # fast path: one glyph per char, spaces agree with gaps, no conflict with confirmed labels.
        # With conflicts the one-to-one reading may be a shifted one (a fused pair or a two-part
        # quote making the counts match by accident): the DP below decides then.
        cost = sum(boundary(i, i) for i in range(m))
        il = lambda a, b: a in AMBIG_IL and b in AMBIG_IL
        n_conf = sum(1 for i in range(m) if strong[i] and strong[i] != chars[i])
        soft = sum(1 for i in range(m) if not strong[i] and trusted[i] and trusted[i] != chars[i]
                   and not il(trusted[i], chars[i]))
        if cost == 0.0 and n_conf <= MAX_CONFLICT_SHARE * m:
            maps = [Mapping((i, i + 1), (i, i + 1), chars[i], sty[i], strong[i], alts[i]) for i in range(m)]
            n_conf -= _il_pairs(maps)
            fast = Alignment(maps, CONFLICT_COST * (n_conf + soft), {i for i in range(1, m) if i in word_start},
                             n_conf, soft)
            if not _shifted(fast):
                return fast

    dp = np.full((m + 1, n + 1), INF)
    back: dict[tuple[int, int], tuple[int, int]] = {}
    dp[0, 0] = 0.0
    for i in range(m):
        for j in range(n):
            base = dp[i, j]
            if base == INF:
                continue
            b = boundary(i, j)
            for k in (1, 2, 3):
                if j + k <= n:
                    c = base + b + one_seg(i, j, k)
                    if c < dp[i + 1, j + k]:
                        dp[i + 1, j + k] = c
                        back[(i + 1, j + k)] = (i, j)
            for s in (2, 3):
                if i + s <= m:
                    c = base + b + multi_seg(i, s, j)
                    if c < dp[i + s, j + 1]:
                        dp[i + s, j + 1] = c
                        back[(i + s, j + 1)] = (i, j)
    if dp[m, n] == INF:
        return None
    path = []
    cur = (m, n)
    while cur != (0, 0):
        prev = back[cur]
        path.append((prev, cur))
        cur = prev
    path.reverse()
    maps = []
    starts = set()
    for (i0, j0), (i1, j1) in path:
        if j0 in word_start and i0 > 0:
            starts.add(i0)
        text = "".join(chars[j0:j1])
        if i1 - i0 == 1:
            lab = strong[i0]
        else:   # several glyphs for one char: trusted sequence rule, else their confirmed labels
            seq = db.sequences.get(db.seq_key(gl[i0:i1]))
            lab = trusted_label(seq)[0] if seq else None
            parts = [strong[t] or unanimous[t] for t in range(i0, i1)]
            if lab is None and any(strong[i0:i1]) and all(parts):
                lab = COMBINE.get("".join(parts), "".join(parts))
        maps.append(Mapping((i0, i1), (j0, j1), text, majority_style(sty[j0:j1]), lab,
                            alts[i0] if i1 - i0 == 1 else None))
    _il_pairs(maps)
    n_conf = sum(1 for mp in maps if mp.db_label and mp.db_label != mp.text)
    soft = sum(1 for mp in maps if mp.segs[1] - mp.segs[0] == 1 and not mp.db_label
               and trusted[mp.segs[0]] and trusted[mp.segs[0]] != mp.text)
    return Alignment(maps, float(dp[m, n]), starts, n_conf, soft)


@dataclass
class LearnResult:
    learned: bool
    reason: str = ""
    conflicts: list[str] | None = None
    alignments: list[Alignment] | None = None


def _il_pairs(maps: list[Mapping]) -> int:
    """DB says I, VLM says l (or vice versa): not a correction the DB may force. The glyph may be
    pixel-identical for both letters, so both readings become alternatives for the lexicon gate."""
    n = 0
    for mp in maps:
        if mp.db_label and mp.db_label != mp.text and mp.db_label in AMBIG_IL and mp.text in AMBIG_IL:
            mp.alts = sorted(set(mp.alts or []) | {mp.db_label, mp.text})
            mp.db_label = None
            n += 1
    return n


def align_cue(db: GlyphDB, lines: list[Line], vlm_text: str,
              gap_threshold: float) -> tuple[list[Alignment] | None, str]:
    """Align every VLM line to its glyph line; (None, reason) if the text does not fit the glyphs."""
    vlm_lines = [l for l in vlm_text.split("\n") if l.strip()]
    if len(vlm_lines) != len(lines):
        return None, f"line count mismatch (vlm {len(vlm_lines)} vs image {len(lines)})"
    aligns = []
    for line, vl in zip(lines, vlm_lines):
        styled = parse_styled(vl)
        a = align_line(db, line, styled, gap_threshold)
        n_chars = sum(1 for c, _ in styled if c != " ")
        if a is None:
            return None, "no alignment"
        if a.conflicts > max(1, MAX_CONFLICT_SHARE * n_chars):
            return None, f"too many conflicts with confirmed glyphs ({a.conflicts})"
        if a.soft_conflicts >= 2:
            return None, f"misaligned: {a.soft_conflicts} confirmed glyphs contradicted"
        if _shifted(a):
            return None, "misaligned: shifted run of contradictions"   # never learn from a shifted reading
        single = sum(1 for mp in a.mappings if mp.segs[1] - mp.segs[0] == 1 and mp.db_label
                     and mp.db_label != mp.text)
        cost = a.cost - CONFLICT_COST * (single + a.soft_conflicts)   # VLM misreads, not alignment errors
        if cost > MAX_COST_PER_CHAR * n_chars + 1.0:
            return None, f"alignment cost too high ({cost:.1f} for {n_chars} chars)"
        aligns.append(a)
    return aligns, ""


ALREADY_LEARNED = "already learned from this image"


def learn_cue(db: GlyphDB, lines: list[Line], vlm_text: str, gap_threshold: float,
              source: str | None = None) -> LearnResult:
    """Align and learn. `source` identifies the cue image; each image contributes votes only once."""
    aligns, reason = align_cue(db, lines, vlm_text, gap_threshold)
    if aligns is None:
        return LearnResult(False, reason)
    if source is not None:
        if source in db.learned_sources:
            return LearnResult(False, ALREADY_LEARNED, alignments=aligns)
        db.learned_sources.add(source)
    conflicts = []
    seen: set[tuple] = set()      # one vote per (glyph, label) per cue: votes must be independent
    for line, a in zip(lines, aligns):
        gl = line.glyphs
        for mp in a.mappings:
            s0, s1 = mp.segs
            if s1 - s0 == 1:
                g = gl[s0]
                v = db.lookup(g.key, g.top_rel)
                prev = confirmed(v)
                if prev and prev != mp.text:
                    conflicts.append(f"{prev!r}->{mp.text!r}")
                # one vote per (cluster, label) per cue: jittered variants of one letter in one
                # cue are not independent evidence
                db.add_vote(g.key, g.bits, g.top_rel, mp.text, mp.style, once=seen)
            else:
                keys = [gl[t].key for t in range(s0, s1)]
                if (tuple(keys), mp.text) not in seen:
                    seen.add((tuple(keys), mp.text))
                    db.add_sequence(gl[s0:s1], mp.text)
        # gaps between mappings
        for mp_a, mp_b in zip(a.mappings, a.mappings[1:]):
            ga, gb = gl[mp_a.segs[1] - 1], gl[mp_b.segs[0]]
            db.add_gap(ga.key, gb.key, gb.x - ga.right, mp_b.segs[0] in a.spaces_before,
                       "i" in mp_a.style or "i" in mp_b.style)
        # word memory
        words: list[tuple[list[str], str]] = [([], "")]
        for mp in a.mappings:
            if mp.segs[0] in a.spaces_before:
                words.append(([], ""))
            keys, txt = words[-1]
            words[-1] = (keys + [gl[t].key for t in range(*mp.segs)], txt + mp.text)
        for keys, txt in words:
            if keys:
                db.add_word(keys, txt)
    return LearnResult(True, conflicts=conflicts or None, alignments=aligns)


def restyle(lines: list[Line], aligns: list[Alignment], style_of, lexicon=None) -> tuple[str, list[str]]:
    """Render the VLM's characters with styles taken from the glyphs they were aligned to.
    Where the VLM contradicts a confirmed glyph label, the confirmed label wins (character-level
    arbitration). For pixel-identical glyphs (I/l) the VLM only guesses from the same pixels, so a
    clear lexicon verdict wins over its reading. style_of(glyph, vlm_style) -> style string.
    Returns (text, corrections)."""
    out = []
    corrections: list[str] = []
    for line, a in zip(lines, aligns):
        shifted = _shifted(a)
        if shifted:
            corrections.append(f"misaligned: {shifted} contradictions in a shifted run, VLM text kept")
        words: list[list[tuple[Mapping, str, str]]] = [[]]
        for mp in a.mappings:
            if mp.segs[0] in a.spaces_before:
                words.append([])
            text = mp.text
            if mp.db_label and mp.db_label != mp.text and not shifted:
                corrections.append(f"{mp.text!r}->{mp.db_label!r}")
                text = mp.db_label
            sts = [style_of(line.glyphs[t], mp.style) for t in range(*mp.segs)]
            words[-1].append((mp, text, majority_style(sts)))
        if lexicon is not None:
            for w in words:
                _lexicon_word(w, lexicon, corrections)
        texts = ["".join(text for _, text, _ in w) for w in words]
        wstyles = inherit_punct_styles([(txt, majority_style([st_ for _, text, st_ in w for _ in text]))
                                        for txt, w in zip(texts, words)])
        chars: list[tuple[str, str]] = []
        for k, (txt, st) in enumerate(zip(texts, wstyles)):
            if k:
                chars.append((" ", ""))
            chars += [(c, st) for c in txt]
        out.append(render_styled(chars))
    return "\n".join(out), corrections


def _shifted(a: Alignment) -> int:
    """Number of contradictions when the alignment is off by one glyph: a run of neighbouring
    mappings where the memory reads glyph k as the character the VLM put on glyph k+1 (or k-1).
    Such a run means the VLM text and the glyphs do not line up (a character drawn as two glyphs,
    read as one), not that the VLM misread six letters in a row. Overriding would garble the word."""
    maps = a.mappings
    conflicts = [mp.db_label is not None and mp.db_label != mp.text for mp in maps]
    if sum(conflicts) < 3:
        return 0
    pairs = 0
    for k in range(len(maps) - 1):
        if conflicts[k] and conflicts[k + 1] and (maps[k].db_label == maps[k + 1].text
                                                   or maps[k].text == maps[k + 1].db_label):
            pairs += 1
    return sum(conflicts) if pairs >= 2 else 0


def _lexicon_word(word: list, lexicon, corrections: list[str]) -> None:
    options = []
    ambiguous = False
    for mp, text, _ in word:
        if mp.alts and mp.db_label is None and all(len(x) == len(text) for x in mp.alts):
            options.append(sorted(set(mp.alts) | {text}))
            ambiguous = True
        else:
            options.append([text])
    if not ambiguous:
        return
    chosen = lexicon.resolve(options)
    current = "".join(text for _, text, _ in word)
    if not chosen or chosen == current:
        return
    corrections.append(f"{current!r}->{chosen!r} (lexicon)")
    pos = 0
    for k, (mp, text, st) in enumerate(word):
        word[k] = (mp, chosen[pos:pos + len(text)], st)
        pos += len(text)
