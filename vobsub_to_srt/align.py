"""Align VLM transcriptions to glyph segments and learn glyph labels from them."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

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
    shifted: set = field(default_factory=set)   # mapping indices inside a shifted run (see _shift_run)


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
        # ... unless segmentation found the pieces physically joined through the anti-alias ring
        # (a k whose stem alone reads as a confirmed l): then they may well be one character.
        joined = bool(line.joined) and all(line.joined[t - 1] for t in range(i + 1, i + s))
        letters = not joined and any(trusted[t] and trusted[t] not in PUNCT for t in range(i, i + s))
        if letters or all(known[i:i + s]):
            together = "".join(known[t] or "\0" for t in range(i, i + s))
            combined = db.seq_label(gl[i:i + s]) or COMBINE.get(together, together)
            if combined != chars[j] and (letters or chars[j] not in PUNCT):
                return INF
        span = gl[i + s - 1].right - gl[i].x
        c = 0.5 * abs(span - exp_w(chars[j])) / avg_w + 1.5 * (s - 1)
        for t in range(i, i + s):
            if joined:
                continue      # pieces of one broken letter: no letter is being swallowed
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
        if cost == 0.0 and n_conf == 0 and soft == 0:
            maps = [Mapping((i, i + 1), (i, i + 1), chars[i], sty[i], strong[i], alts[i]) for i in range(m)]
            return Alignment(maps, 0.0, {i for i in range(1, m) if i in word_start}, 0, 0)

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
            lab = db.seq_label(gl[i0:i1])
            parts = [strong[t] or unanimous[t] for t in range(i0, i1)]
            if lab is None and any(strong[i0:i1]) and all(parts):
                lab = COMBINE.get("".join(parts), "".join(parts))
        maps.append(Mapping((i0, i1), (j0, j1), text, majority_style(sty[j0:j1]), lab,
                            alts[i0] if i1 - i0 == 1 else None))
    _il_pairs(maps)
    n_conf = sum(1 for mp in maps if mp.db_label and mp.db_label != mp.text)
    soft = sum(1 for mp in maps if mp.segs[1] - mp.segs[0] == 1 and not mp.db_label
               and trusted[mp.segs[0]] and trusted[mp.segs[0]] != mp.text)
    return Alignment(maps, float(dp[m, n]), starts, n_conf, soft, _shift_run(maps))


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
        single = sum(1 for mp in a.mappings if mp.segs[1] - mp.segs[0] == 1 and mp.db_label
                     and mp.db_label != mp.text)
        cost = a.cost - CONFLICT_COST * (single + a.soft_conflicts)   # VLM misreads, not alignment errors
        if cost > MAX_COST_PER_CHAR * n_chars + 1.0:
            return None, f"alignment cost too high ({cost:.1f} for {n_chars} chars)"
        aligns.append(a)
    return aligns, ""


_BREAK = re.compile(r"<\s*br\s*/?\s*>|\n", re.I)
RELINE_TOL = 3               # chars a line's text may differ from its glyph count at a candidate split
RELINE_WIDE = 0.2            # ... share of the glyph count a word boundary may miss by when nothing fits RELINE_TOL


def flatten(text: str) -> list[tuple[str, str]]:
    """The VLM's line breaks (newlines, <br>) become spaces: one styled character stream."""
    return parse_styled(_BREAK.sub(" ", text))


def reline(db: GlyphDB, lines: list[Line], vlm_text: str, gap_threshold: float) -> str:
    """Line breaks come from the image, not from the VLM. Flatten the VLM text and split it where
    the glyph lines end: among the word boundaries whose character count fits the glyph count of
    the line (+- RELINE_TOL), the split with the lowest alignment cost wins; with no fitting word
    boundary any character position is a candidate (the VLM merged two words across the break).
    If nothing aligns, the split closest to the glyph count is used."""
    _, parts = reline_parts(db, lines, flatten(vlm_text), gap_threshold)
    return "\n".join(render_styled(pt) for pt in parts if pt)


def reline_cues(db: GlyphDB, cues: list[list[Line]], vlm_text: str,
                gap_threshold: float) -> list[str] | None:
    """One VLM answer for several stacked cues: split it into one text per cue by the same rule
    (every line of every cue takes its share of the text). None if some line did not align."""
    lines = [l for c in cues for l in c]
    cost, parts = reline_parts(db, lines, flatten(vlm_text), gap_threshold)
    if cost == INF or len(parts) != len(lines):
        return None
    out, k = [], 0
    for c in cues:
        out.append("\n".join(render_styled(pt) for pt in parts[k:k + len(c)] if pt))
        k += len(c)
    return out


def reline_parts(db: GlyphDB, lines: list[Line], styled: list[tuple[str, str]],
                 gap_threshold: float) -> tuple[float, list[list[tuple[str, str]]]]:
    """Split one styled character stream over the glyph lines (see reline): total alignment cost
    and one part per line. Dynamic programme over (line, start position), memoised."""
    while styled and styled[0][0] == " ":
        styled = styled[1:]
    n_lines = len(lines)
    N = len(styled)
    if n_lines == 0 or N == 0:
        return INF, [[] for _ in lines]
    if n_lines == 1:
        a = align_line(db, lines[0], styled, gap_threshold)
        return (a.cost if a else INF), [styled]
    counts = [len(l.glyphs) for l in lines]
    before = [0] * (N + 1)          # non-space characters before position idx
    for idx, (c, _) in enumerate(styled):
        before[idx + 1] = before[idx] + (c != " ")
    # A line's share of the characters, estimated from its pixel width. Letters of heavy fonts
    # touch and become one glyph, so the glyph count may fall short by several characters; the
    # width does not care how the ink is split. Both estimates are tried.
    widths = [max(1, l.glyphs[-1].right - l.glyphs[0].x) if l.glyphs else 1 for l in lines]
    by_width = [round(before[N] * w / sum(widths)) for w in widths]
    memo: dict[tuple[int, int], tuple[float, list]] = {}

    def trimmed(a: int, b: int) -> tuple[int, int]:
        while a < b and styled[a][0] == " ":
            a += 1
        while b > a and styled[b - 1][0] == " ":
            b -= 1
        return a, b

    def line_cost(k: int, a: int, b: int) -> float:
        a, b = trimmed(a, b)
        if a >= b:
            return INF
        al = align_line(db, lines[k], styled[a:b], gap_threshold)
        return al.cost if al else INF

    def solve(start: int, k: int) -> tuple[float, list]:
        hit = memo.get((start, k))
        if hit is not None:
            return hit
        if k == n_lines - 1:
            a, b = trimmed(start, N)
            res = (line_cost(k, a, b), [styled[a:b]])
            memo[(start, k)] = res
            return res
        want = counts[k]
        wide = max(RELINE_TOL, int(RELINE_WIDE * want))

        def dev_of(n: int) -> int:
            return min(abs(n - want), abs(n - by_width[k]))

        cands: list[tuple[int, int, int, int]] = []  # (end of left part, start of right part, in-word, deviation)
        for idx in range(start + 1, N):
            n = before[idx] - before[start]
            if dev_of(n) <= RELINE_TOL and before[N] - before[idx] >= 1 and styled[idx][0] == " ":
                cands.append((idx, idx + 1, 0, dev_of(n)))
        if not cands:
            # No word boundary fits either estimate: the VLM merged two words across the break
            # (the split lies inside a word) or both estimates are off (a word boundary a little
            # further away). Both are candidates; the alignment cost decides, a word boundary wins ties.
            for idx in range(start + 1, N):
                n = before[idx] - before[start]
                if before[N] - before[idx] < 1:
                    continue
                if dev_of(n) <= RELINE_TOL and styled[idx][0] != " ":
                    cands.append((idx, idx, 1, dev_of(n)))
                elif dev_of(n) <= wide and styled[idx][0] == " ":
                    cands.append((idx, idx + 1, 0, dev_of(n)))
        best: tuple[tuple[float, int, int], list] | None = None
        for end, nxt, inword, dev in cands:
            a, b = trimmed(start, end)
            if a >= b:
                continue
            sub_cost, rest = solve(nxt, k + 1)
            total = line_cost(k, a, b) + sub_cost
            if best is None or (total, inword, dev) < best[0]:
                best = ((total, inword, dev), [styled[a:b]] + rest)
        if best is None:
            a, b = trimmed(start, N)
            res = (INF, [styled[a:b]] + [[] for _ in range(n_lines - k - 1)])
        else:
            res = (best[0][0], best[1])
        memo[(start, k)] = res
        return res

    return solve(0, 0)


ALREADY_LEARNED = "already learned from this image"


def learn_cue(db: GlyphDB, lines: list[Line], vlm_text: str, gap_threshold: float,
              source: str | None = None) -> LearnResult:
    """Align and learn. `source` identifies the cue image; each image contributes votes only once."""
    aligns, reason = align_cue(db, lines, vlm_text, gap_threshold)
    if aligns is None:
        return LearnResult(False, reason)
    if source is not None and not db.learn_source(source):
        return LearnResult(False, ALREADY_LEARNED, alignments=aligns)
    conflicts = []
    seen: set[tuple] = set()      # one vote per (glyph, label) per cue: votes must be independent
    for line, a in zip(lines, aligns):
        gl = line.glyphs
        for k, mp in enumerate(a.mappings):
            if k in a.shifted:
                continue          # the VLM text and the glyphs do not line up here: no evidence
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
                if (tuple(keys), mp.text) not in seen and not db.dropping_sequence(gl[s0:s1], mp.text):
                    seen.add((tuple(keys), mp.text))
                    db.add_sequence(gl[s0:s1], mp.text)
        # gaps between mappings
        for k, (mp_a, mp_b) in enumerate(zip(a.mappings, a.mappings[1:])):
            if k in a.shifted or k + 1 in a.shifted:
                continue
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


def _shift_run(maps: list[Mapping]) -> set[int]:
    """Mapping indices where the alignment is off by one glyph: runs of neighbouring contradictions
    in which the memory reads glyph k as the character the VLM put on glyph k+1 (or k-1). Such a
    run means the VLM text and the glyphs do not line up there (a dropped letter, a character drawn
    as two glyphs read as one), not that the VLM misread several letters in a row. The rest of the
    line is still evidence; the run itself is not, and overriding inside it would garble the word."""
    conflicts = [mp.db_label is not None and mp.db_label != mp.text for mp in maps]
    out: set[int] = set()
    run: list[int] = []
    pairs = 0
    for k in range(len(maps) + 1):
        if k < len(maps) and conflicts[k]:
            if run and (maps[run[-1]].db_label == maps[k].text or maps[run[-1]].text == maps[k].db_label):
                pairs += 1
            run.append(k)
            continue
        if len(run) >= 3 and pairs >= 2:
            out.update(run)
        run, pairs = [], 0
    return out


def _shifted(a: Alignment) -> int:
    return len(a.shifted)


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
