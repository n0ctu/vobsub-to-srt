"""Per-font glyph database: exact bitmaps with voted labels, sequence rules and gap statistics."""
from __future__ import annotations

import base64
import json
import os
from contextlib import contextmanager
try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from .version import app_version

POS_TOL_UNITS = 0.05   # tolerance on top_rel (baseline jitter), in font units (x-height)
RESCALED_POS_TOL = 2   # ... extra pixels for a rescaled track (vertical jitter)
SEQ_PART_TOL = 0.2     # a sequence part may snap to a sequence-bearing cluster this far away (key only)
XHEIGHT_CHARS = set("acemnorsuvwxz")   # lowercase letters without ascender/descender
MIN_VOTES = 2          # a label needs this many independent observations (cues) ...
MIN_SHARE = 2 / 3      # ... and at least this share of all votes before it is trusted
DISSENT_VOTES = 3      # a label that was ever read differently needs this many agreeing reads
# overriding a VLM reading needs more evidence than reading a glyph: the majority may just be the
# VLM's own habit (e.g. writing ' for the acute accent in O´Neil in 2 of 3 cues)
OVERRIDE_VOTES = 3
OVERRIDE_SHARE = 0.75
# Clusters: rescaled tracks never put a letter on the pixel grid the same way twice, so one letter
# comes as hundreds of bitmaps that differ by one-pixel edge jitter. Shapes within CLUSTER_TOL of a
# stored shape (same size +-1 px, same baseline position) join its cluster and share its votes: two
# reads of two jittered variants are the two independent reads the quarantine demands. For clean
# bitmap fonts nothing changes (their repeats are exact). Glyphs whose identity hangs on one pixel
# row or a position (I/l, i/l, 1, punctuation) never cluster across a height difference.
CLUSTER_TOL = 0.12
STRICT_TOL = 0.05       # for strict glyphs: an i differs from an l by its dot gap, ~0.1 of the ink
FUSED_WIDTH_SHARE = 0.7 # a multi-letter reading needs a glyph at least this wide relative to the letters
STRICT_MARGIN = 0.05    # ... unless no cluster reading differently lies within this of the match (rival_free)
LOOKALIKES = ("Il1|!iíìïîj", "035689gq")   # classes whose members a font may draw alike
CLUSTER_MARGIN = 0.06     # lead over the nearest shape with a different label
NEAR_CACHE_MAX = 200_000   # remembered near searches (glyph key, bound) before the memo is reset
RIVAL_MAX = 0.3            # widest margin rival_free looks at
TOLERANT_TOL = 0.05       # edge-tolerant difference accepted as "the same glyph, jittered" (recognize.near_match stage 2)
TOLERANT_MARGIN = 0.03    # ... unless another label's shape is nearly as close
PROTO_MIN = 6             # samples before a cluster's prototype is used for matching
PROTO_TOL = 0.04          # disagreements on stable pixels / stable ink accepted as the same letter
PROTO_STRICT_TOL = 0.02   # ... for strict letters (I l 1 ! i j) and unlabelled clusters
PROTO_MARGIN = 0.03       # lead over the nearest prototype with a different label
PROTO_MIN_PX = 3          # ... or at most this many disagreeing stable pixels (small glyphs: i , ')
PROTO_STABLE = 0.9        # a pixel is stable if ink in >= 90% or <= 10% of the samples
MIN_MEMBER_SEEN = 2       # a jitter member seen once is not kept (rescaled tracks: thousands per file)
MAX_MEMBERS = 24          # most frequent members kept per cluster; the rest is read via the near search
STRICT = set("Il|i1!jíìïî") | set("035689g")   # I/l & co; digits (and g/9) a jittered font blurs into each other.
                                               # Punctuation is told apart by size and baseline position instead
_ALIGN = [(dy, dx) for dy in (0, 1, 2) for dx in (0, 1, 2)]   # the 3x3 alignments of a near comparison


def _high(g) -> bool:
    """Does the part sit clearly above the baseline (its bottom higher than its own height)?"""
    top = getattr(g, "top_rel", None)
    if top is None:
        return False
    h = g.bits.shape[0]
    return top + h < -h


def _stacked(glyphs) -> bool:
    """True if any two consecutive parts are vertically separated (one above the other).
    Journal stand-ins carry the flag instead of a position."""
    if any(not hasattr(g, "y") for g in glyphs):
        return False
    return any(a.y + a.bits.shape[0] <= b.y or b.y + b.bits.shape[0] <= a.y for a, b in zip(glyphs, glyphs[1:]))


def diff_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """Pixel difference of two bitmaps (best of 3x3 alignments) relative to their mean ink."""
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


def local_diff_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """Pixel difference relative to the ink in the columns where the two bitmaps differ (best of
    3x3 alignments). Jitter touches every letter of a glyph, so this equals diff_ratio; a glyph of
    several touching letters that differs from another in ONE letter ('fte' vs 'ffe') has its
    difference concentrated in that letter's columns, and the ratio there is a letter's worth."""
    h, w = max(a.shape[0], b.shape[0]) + 2, max(a.shape[1], b.shape[1]) + 2
    A = np.zeros((h, w), bool)
    A[1:1 + a.shape[0], 1:1 + a.shape[1]] = a
    best = None
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            if dy + b.shape[0] > h or dx + b.shape[1] > w:
                continue
            B = np.zeros((h, w), bool)
            B[dy:dy + b.shape[0], dx:dx + b.shape[1]] = b
            D = A ^ B
            d = int(D.sum())
            if best is not None and d >= best[0]:
                continue
            cols = np.flatnonzero(D.any(axis=0))
            if cols.size == 0:
                return 0.0
            local = (int(A[:, cols[0]:cols[-1] + 1].sum()) + int(B[:, cols[0]:cols[-1] + 1].sum())) / 2
            best = (d, d / max(1.0, local))
    return best[1] if best else 0.0


def letters_in(*labels: str | None) -> int:
    return max((sum(c.isalnum() for c in l) for l in labels if l), default=0)


def is_strict(label: str | None) -> bool:
    return bool(label) and bool(set(label) & STRICT)


def tol_for(*labels: str | None) -> float:
    return STRICT_TOL if any(is_strict(l) for l in labels) else CLUSTER_TOL


def trusted_label(votes: Counter, dissent_votes: int = DISSENT_VOTES) -> tuple[str | None, str]:
    """Majority label if confirmed (>= MIN_VOTES, >= MIN_SHARE of votes, >= DISSENT_VOTES after any
    disagreement), else (None, reason)."""
    if not votes:
        return None, "no votes"
    total = sum(votes.values())
    top, n1 = votes.most_common(1)[0]
    if total < MIN_VOTES:
        return None, f"unconfirmed ({top!r} seen once)"
    # two unanimous reads suffice; once any read disagreed, the majority must be read a third time
    # (a 6/8 cluster read 6, 6, 8 stays open until a fourth read settles it)
    dissent = sum(1 for n in votes.values() if n > 0) > 1
    if n1 < MIN_SHARE * total or (dissent and n1 < dissent_votes):
        return None, f"conflicting votes {dict(votes)}"
    return top, ""




@dataclass
class Variant:
    top_rel: int
    votes: Counter = field(default_factory=Counter)     # label -> count
    styles: dict = field(default_factory=dict)          # VLM tags: "b"/"i" -> [no, yes]
    geo_italic: list = field(default_factory=lambda: [0, 0])  # word-slant votes: [upright, italic]
    geo_bold: list = field(default_factory=lambda: [0, 0])    # stroke-width votes: [regular, bold]
    # label transferred from a teacher DB at another raster size: usable while no VLM read
    # contradicts it, counts as one vote otherwise, never overrides the VLM
    prior: str | None = None
    # loaded from a main file written by 0.3.0 or later: it was settled when that file was saved
    # and stays in the main file (a later dissenting read must not push a confirmed glyph out)
    kept: bool = field(default=False, compare=False)

    def style(self) -> str:
        """Glyph style from geometry (word slant, stroke width); VLM tags only break ties.
        Underline is not a glyph property (see Glyph.underlined)."""
        return combine_style(self.geo_bold, self.geo_italic, self.styles)


def combine_style(geo_bold, geo_italic, vlm_styles: dict | None = None) -> str:
    """Geometry decides whenever it has an opinion; VLM tags (which miss and invent styles) only
    decide when the geometric votes are absent or exactly tied."""
    vlm_styles = vlm_styles or {}
    out = ""
    for flag, geo in (("b", geo_bold), ("i", geo_italic)):
        if geo[1] != geo[0]:
            on = geo[1] > geo[0]
        else:
            v = vlm_styles.get(flag, [0, 0])
            on = v[1] > v[0]
        if on:
            out += flag
    return out


@dataclass
class Shape:
    key: str
    bits: np.ndarray
    variants: list[Variant] = field(default_factory=list)   # only on a cluster's canonical shape
    derived_from: str | None = None   # key of the shape this was near-matched to
    cluster: str = ""                 # canonical shape whose variants hold the votes (own key if canonical)
    n: int = 1                        # occurrences seen (all files); jitter members seen once are pruned on save

    def variant(self, top_rel: int, tol: int = 2, create: bool = False) -> Variant | None:
        best = None
        for v in self.variants:
            d = abs(v.top_rel - top_rel)
            if d <= tol and (best is None or d < abs(best.top_rel - top_rel)):
                best = v
        if best is None and create:
            best = Variant(top_rel)
            self.variants.append(best)
        return best


_TOPO: dict[str, tuple[int, int]] = {}


def topology(key: str, bits: np.ndarray) -> tuple[int, int]:
    """(connected ink components, holes) of a glyph: a one-pixel jitter rarely changes them, a
    missing stroke or a gap does (l vs !, c vs e, o vs c). Cached per bitmap key."""
    hit = _TOPO.get(key)
    if hit is None:
        from scipy import ndimage
        comps = int(ndimage.label(bits, structure=np.ones((3, 3), int))[1])
        bg = ndimage.label(~np.pad(bits, 1))[1]          # 4-connected background regions
        hit = _TOPO[key] = (comps, max(0, int(bg) - 1))
        if len(_TOPO) > 200000:
            _TOPO.clear()
    return hit


def _unpack(b64: str, h: int, w: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(base64.b64decode(b64), np.uint8))[:h * w].reshape(h, w).astype(bool)


def _proto_json(p: "Proto | None") -> dict | None:
    if p is None or p.n <= 0:
        return None
    m, st = p.median, p.stable
    return {"n": p.n, "h": int(m.shape[0]), "w": int(m.shape[1]),
            "median": base64.b64encode(np.packbits(m).tobytes()).decode(),
            "stable": base64.b64encode(np.packbits(st).tobytes()).decode()}


def _dilate(stack: np.ndarray) -> np.ndarray:
    """3x3 dilation of a stack of bitmaps (n, h, w), with a one-pixel border added."""
    n, h, w = stack.shape
    out = np.zeros((n, h + 2, w + 2), bool)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out[:, dy:dy + h, dx:dx + w] |= stack
    return out


class Proto:
    """A cluster's prototype: every sample of the letter accumulated on one canvas (the canonical
    shape plus a one-pixel border, samples aligned by best shift). The median bitmap is the letter
    as the font draws it; pixels that are ink in nearly all or nearly no samples are stable, the
    rest flicker with the raster. Matching counts disagreements on stable pixels only, so a
    jittered variant scores ~0 while a different letter differs where it matters (a bar, a gap,
    an extra row). The counts (acc) are private training data; median + mask are publishable."""
    __slots__ = ("acc", "n", "_median", "_stable")

    def __init__(self, acc=None, n: int = 0, median=None, stable=None):
        self.acc, self.n, self._median, self._stable = acc, n, median, stable

    @property
    def median(self) -> np.ndarray:
        return self.acc * 2 > self.n if self.acc is not None else self._median

    @property
    def stable(self) -> np.ndarray:
        if self.acc is not None:
            return (self.acc >= PROTO_STABLE * self.n) | (self.acc <= (1 - PROTO_STABLE) * self.n)
        return self._stable


class _Part:
    """A glyph stand-in for journal replay (key, bits, top_rel)."""
    __slots__ = ("key", "bits", "top_rel")

    def __init__(self, key: str, bits: np.ndarray, top_rel: int):
        self.key, self.bits, self.top_rel = key, bits, top_rel


@contextmanager
def _locked(path: Path):
    """Exclusive file lock next to a DB file (advisory, POSIX); a no-op where unavailable."""
    lock = path.with_suffix(".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "w") as f:
        if fcntl is not None:
            fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_UN)


class GlyphDB:
    def __init__(self, name: str, path: Path | None = None):
        self.name = name
        self.path = path
        self.shapes: dict[str, Shape] = {}
        # multi-segment characters, e.g. '"' made of two ticks: "k1|k2" -> {label: votes}
        self.sequences: dict[str, Counter] = {}
        # gap statistics: italic flag -> gap px -> [letter_count, space_count]
        self.gaps: dict[str, dict[int, list[int]]] = {"0": defaultdict(lambda: [0, 0]), "1": defaultdict(lambda: [0, 0])}
        # per glyph pair gap stats: "ka|kb|gap" -> [letter, space]
        self.pair_gaps: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        # word memory for resolving ambiguous glyphs (I/l): "k1|k2|..." -> {text: votes}
        self.words: dict[str, Counter] = {}
        self.by_size: dict[tuple[int, int], list[str]] = defaultdict(list)
        # size bucket -> (keys, stacked bits, ink counts), grown in place as shapes join (buckets are
        # append-only except for pruning, which drops the bucket's stack)
        self._stacks: dict[tuple[int, int], tuple] = {}
        self._near_cache: dict[tuple[str, float], tuple[tuple, list]] = {}   # (glyph key, tol) -> (bucket stamp, result)
        self._lookalike_cache: dict[tuple[str, int], bool] = {}            # (label, height) -> lookalike_known
        self._width_cache: dict[str, float] | None = None                  # label -> median width of its clusters
        self._tstacks: dict[tuple[int, int], tuple[list[str], np.ndarray, np.ndarray]] = {}   # + dilated bits
        self._soft: dict[str, str] = {}           # soft_canonical cache, cleared when shapes are added
        self.dirty = False
        self._bearings: tuple | None = None   # fitted side-bearing model (cached)
        self._seq_n = -1                      # sequences count the _seq_sizes cache was built for
        # font unit: x-height in px of this DB's raster. Everything compared across rasters is
        # expressed in this unit; within the raster, exact bitmaps are used.
        self.unit: float | None = None
        self.parent: str | None = None     # teacher DB this one was bootstrapped from (other raster)
        self.learned_with: str | None = app_version()   # app version that created the set (kept on load)
        self.merged_from: list[str] = []                 # other sets whose votes were merged in (import tool)
        self.seeded_from: str | None = None              # image version that installed this copy (docker seed)
        # hashes of cue images already learned from: re-reading the same image (re-runs, duplicate
        # cues) is not independent evidence and must not confirm a label
        self.learned_sources: set[str] = set()
        # Private sidecar (never part of the publishable DB): source hashes always, the word
        # memory only if enabled. None = not attached (e.g. a read-only teacher DB).
        self.private_path: Path | None = None
        self.word_memory = False
        self.charset = "simplified"          # labels folded by simplify.py, or "literal"
        # Learning journal: every write since the last save, as (op, args). save() replays it onto
        # the file's current content (another worker may have learned meanwhile), see save().
        self.journal: list[tuple] | None = []
        # this file's geometry votes per glyph key ([upright, italic], [regular, bold]), added to
        # the file's accumulated votes on save (pipeline.apply_geo keeps the in-memory view)
        self.file_geo: tuple[dict, dict] = ({}, {})
        self._geo_credited: set[str] = set()   # clusters whose file geometry votes are already on disk
        # this file's glyph occurrence counts per key (pipeline sets it); credited to Shape.n once
        self.file_counts: dict[str, int] = {}
        # edge-tolerant matching (recognize.near_match stage 2) is only for rescaled tracks, whose
        # jitter it is made for; a crisp track's exact bitmaps never need it (pipeline sets it)
        self.tolerant = False
        self.rescaled = False      # set per file by the pipeline: the track's bitmaps jitter (see pos_tol)
        self._il_cache: tuple[int, bool] = (-1, False)
        self.protos: dict[str, Proto] = {}       # canonical key -> prototype (see Proto)
        self._pstacks: dict | None = None        # canvas size -> (keys, medians, masks), rebuilt lazily
        self._counts_credited: set[str] = set()
        self.saved_this_run = False           # a final save (with pruning) follows any mid-run save
        # Interim learning: glyph positions read but not settled yet (one read, or reads that
        # disagree) live in the private sidecar, not in the main file, so shipped and published sets
        # hold settled glyphs only. In memory both are one set; save() splits them (see _split).
        self._kept_seqs: set[str] = set()     # sequences loaded from a 0.3.0+ main file
        self._interim_from: Path | None = None   # sidecar whose interim part is merged in

    @property
    def pos_tol(self) -> int:
        base = max(1, round(POS_TOL_UNITS * self.unit)) if self.unit else 2
        # A rescaled track jitters vertically as well: the same tick lands at -29, -30 or -31. With
        # one pixel of tolerance each landing started its own cluster, so a quote pair was a new
        # sequence every time. Confusable positions (comma vs tick, hyphen vs underscore) lie
        # half an x-height or more apart, far outside the widened window.
        return base + RESCALED_POS_TOL if self.rescaled else base

    def update_unit(self) -> None:
        """x-height = median height of confirmed, upright x-height letters."""
        hs = []
        for s in self.shapes.values():
            for v in s.variants:
                lab = trusted_label(v.votes)[0]
                if lab and lab in XHEIGHT_CHARS and "i" not in v.style():
                    hs.append(s.bits.shape[0])
        if len(hs) >= 3:
            self.unit = float(np.median(hs))

    # ---------- clusters ----------
    def canonical(self, key: str) -> str:
        shape = self.shapes.get(key)
        return shape.cluster if shape else key

    def _label_of(self, shape: Shape, top_rel: int) -> str | None:
        """What the cluster has been read as so far (majority, confirmed or not), else its prior."""
        v = self.shapes[shape.cluster].variant(top_rel, self.pos_tol)
        if v is None:
            return None
        votes = +v.votes
        return votes.most_common(1)[0][0] if votes else v.prior

    def _bucket_stamp(self, h: int, w: int) -> tuple:
        return tuple(len(self.by_size.get((h + dh, w + dw), ())) for dh in (-1, 0, 1) for dw in (-1, 0, 1))

    def near_candidates(self, bits: np.ndarray, max_ratio: float = CLUSTER_TOL,
                        key: str | None = None) -> list[tuple[float, str, int]]:
        """All stored shapes of the same size +-1 px within `max_ratio` of `bits`, nearest first,
        as (ratio, key, dh). Vectorised per size bucket: the bucket's bitmaps are stacked once;
        shapes whose ink count alone puts them out of reach are dropped (the difference is at
        least |ink_a - ink_b|), the rest is compared for all 3x3 alignments in one operation.
        With `key` (the glyph's bitmap key) the result is remembered until a shape joins one of
        the nine size buckets: the probe, the reading pass and the clustering of a learned
        variant ask the same question about the same bitmap."""
        h, w = bits.shape
        stamp = self._bucket_stamp(h, w)
        hit = self._near_cache.get((key, max_ratio)) if key is not None else None
        if hit is not None and hit[0] == stamp:
            return [(r, k, dh) for r, b, i, k, dh in hit[1]]
        # Shapes only ever join a bucket (appended), so a remembered result stays valid for the
        # shapes it saw; only the newcomers of each grown bucket are compared. Candidates carry
        # (bucket order, index in bucket) so the final order equals that of a full search.
        found: list[tuple[float, int, int, str, int]] = list(hit[1]) if hit is not None else []
        seen = hit[0] if hit is not None else (0,) * 9
        na = int(bits.sum())
        for b, (dh, dw) in enumerate(_ALIGN):
            dh -= 1
            dw -= 1
            size = (h + dh, w + dw)
            keys = self.by_size.get(size)
            if not keys or len(keys) <= seen[b]:
                continue
            skeys, flat, nb = self._stack(size, keys)
            # ink-count bound: the pixel difference is at least |ink_a - ink_b|
            reach = np.abs(nb - na) <= max_ratio * np.maximum(1.0, (na + nb) / 2)
            if seen[b]:
                reach[:seen[b]] = False
            idx = np.nonzero(reach)[0]
            if not len(idx):
                continue
            # pixel difference = ink_a + ink_b - 2 * overlap; the overlap for all nine alignments
            # is one matrix product of the query's nine shifted windows with the bucket's bitmaps
            # (0/1 values: exact in float32)
            H, W = max(h, size[0]) + 2, max(w, size[1]) + 2
            A = np.zeros((H, W), np.float32)
            A[1:1 + h, 1:1 + w] = bits
            windows = np.lib.stride_tricks.sliding_window_view(A, size)[:3, :3].reshape(9, -1)
            overlap = windows @ flat[idx].T                               # (9, n)
            best = (na + nb[idx] - 2 * overlap.astype(np.int64)).min(axis=0)
            ratios = best / np.maximum(1.0, (na + nb[idx]) / 2)
            sel = np.nonzero(ratios <= max_ratio)[0]
            if len(sel):
                found.extend((r, b, i, skeys[i], dh) for r, i in zip(ratios[sel].tolist(), idx[sel].tolist()))
        found.sort(key=lambda c: c[:3])
        if key is not None:
            if len(self._near_cache) >= NEAR_CACHE_MAX:
                self._near_cache.clear()
            self._near_cache[(key, max_ratio)] = (stamp, found)
        return [(r, k, dh) for r, b, i, k, dh in found]

    def _stack(self, size: tuple[int, int], keys: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
        """The bucket's bitmaps as rows of one float32 matrix plus their ink counts. Shapes joining
        the bucket are appended to the existing matrix (capacity doubles), so a track that adds
        thousands of jittered variants to the same buckets does not restack them on every join."""
        st = self._stacks.get(size)
        n = len(keys)
        if st is None:
            cap = max(16, n)
            buf = np.zeros((cap, size[0] * size[1]), np.float32)
            ink = np.zeros(cap, np.int64)
            st = self._stacks[size] = [[], buf, ink]
        skeys, buf, ink = st
        m = len(skeys)
        if m < n:
            if n > len(buf):
                cap = max(n, 2 * len(buf))
                nbuf = np.zeros((cap, buf.shape[1]), np.float32)
                nbuf[:m] = buf[:m]
                nink = np.zeros(cap, np.int64)
                nink[:m] = ink[:m]
                st[1], st[2] = buf, ink = nbuf, nink
            new = np.stack([self.shapes[k].bits for k in keys[m:]]).reshape(n - m, -1)
            buf[m:n] = new
            ink[m:n] = new.sum(axis=1)
            skeys.extend(keys[m:])
        return skeys, buf[:n], ink[:n]

    def tolerant_candidates(self, bits: np.ndarray, max_ratio: float = TOLERANT_TOL) -> list[tuple[float, str, int]]:
        """Like near_candidates, with an edge-tolerant difference: ink of one bitmap that lies within
        one pixel of the other's ink does not count. A rescaled track moves edge pixels by one
        (thousands of unique bitmaps per episode); under this measure they are the same glyph,
        while letters that differ by a stroke (c/e, n/h) stay apart. Returns (ratio, key, dh)."""
        h, w = bits.shape
        out: list[tuple[float, str, int]] = []
        na = int(bits.sum())
        for dh in (-1, 0, 1):
            for dw in (-1, 0, 1):
                size = (h + dh, w + dw)
                keys = self.by_size.get(size)
                if not keys:
                    continue
                st = self._tstacks.get(size)
                if st is None or len(st[0]) != len(keys):
                    stack = np.stack([self.shapes[k].bits for k in keys])
                    st = (list(keys), stack, _dilate(stack))
                    self._tstacks[size] = st
                skeys, stack, dstack = st
                H, W = max(h, size[0]) + 4, max(w, size[1]) + 4
                A = np.zeros((H, W), bool)
                A[2:2 + h, 2:2 + w] = bits
                dA = _dilate(A[None])[0][1:-1, 1:-1]          # same canvas as A
                nb = stack.reshape(len(skeys), -1).sum(axis=1)
                best = None
                for dy in (1, 2, 3):
                    for dx in (1, 2, 3):
                        if dy + size[0] > H or dx + size[1] > W:
                            continue
                        B = np.zeros((len(skeys), H, W), bool)
                        B[:, dy:dy + size[0], dx:dx + size[1]] = stack
                        dB = np.zeros((len(skeys), H, W), bool)
                        dB[:, dy - 1:dy + size[0] + 1, dx - 1:dx + size[1] + 1] = dstack
                        d = (A[None] & ~dB).reshape(len(skeys), -1).sum(axis=1) + \
                            (B & ~dA[None]).reshape(len(skeys), -1).sum(axis=1)
                        best = d if best is None else np.minimum(best, d)
                ratios = best / np.maximum(1.0, (na + nb) / 2)
                for i in np.nonzero(ratios <= max_ratio)[0]:
                    out.append((float(ratios[i]), skeys[i], dh))
        out.sort(key=lambda c: c[0])
        return out

    def letter_widths(self) -> dict[str, float]:
        """Median bitmap width per confirmed single-letter label (recomputed after new votes)."""
        if self._width_cache is None:
            acc: dict[str, list[int]] = {}
            for s in self.shapes.values():
                if s.cluster != s.key:
                    continue
                for v in s.variants:
                    lab = trusted_label(v.votes)[0]
                    if lab and len(lab) == 1:
                        acc.setdefault(lab, []).append(int(s.bits.shape[1]))
            self._width_cache = {k: float(np.median(w)) for k, w in acc.items()}
        return self._width_cache

    def fits_width(self, text: str, width: int) -> bool:
        """Can a glyph `width` pixels wide hold `text`? Judged by the set's own letter widths; a
        set that knows nothing yet cannot judge (True). A glyph narrower than 0.7 of the letters'
        summed widths cannot be two letters: an aligner that pushed a surplus letter onto it was
        placing a model error, not reading a fused pair."""
        widths = self.letter_widths()
        if len(text) < 2 or not widths:
            return True
        typical = float(np.median(list(widths.values())))
        expected = sum(widths.get(c, typical) for c in text)
        return width >= FUSED_WIDTH_SHARE * expected

    def lookalike_known(self, label: str, h: int) -> bool:
        """Has the set confirmed a look-alike of `label` (another letter of its class) at about
        this height? Only then does the absence of a rival near a match mean anything: before
        the first I is learned, a one-row-shorter l must stay a question, not read as l."""
        cls = next((c for c in LOOKALIKES if label in c), None)
        if cls is None:
            return False
        hit = self._lookalike_cache.get((label, h))
        if hit is None:
            hit = False
            for s in self.shapes.values():
                if s.cluster != s.key or abs(s.bits.shape[0] - h) > 3:
                    continue
                for v in s.variants:
                    lab = trusted_label(v.votes)[0]
                    if lab and lab != label and lab in cls:
                        hit = True
                        break
                if hit:
                    break
            self._lookalike_cache[(label, h)] = hit
        return hit

    def rival_free(self, bits: np.ndarray, top_rel: int, label: str, r: float, exclude: str | None = None,
                   key: str | None = None) -> bool:
        """No cluster reading differently from `label` within STRICT_MARGIN of a match at `r`,
        in a set that knows a look-alike of the letter. Used to let a strict letter (I l 1,
        digits, g) bridge one pixel of height on a rescaled track; the strict tolerance itself
        stays, a look-alike the set has not learned yet may sit just beyond it."""
        if not self.lookalike_known(label, bits.shape[0]):
            return False
        # one search at the widest margin (remembered per glyph key), cut to this match's margin:
        # the same candidates in the same order as a search with the narrower bound
        reach = min(RIVAL_MAX, r + STRICT_MARGIN)
        for r2, k2, dh in self.near_candidates(bits, max_ratio=RIVAL_MAX, key=key):
            if r2 > reach:
                break
            shape = self.shapes[k2]
            if exclude is not None and shape.cluster == exclude:
                continue
            v = self.shapes[shape.cluster].variant(top_rel, self.pos_tol)
            votes = +v.votes if v is not None else None
            if votes and votes.most_common(1)[0][0] != label:
                return False
        return True

    def find_cluster(self, bits: np.ndarray, top_rel: int, label: str | None = None,
                     key: str | None = None) -> str | None:
        """Canonical key of the cluster a new shape belongs to, or None if it starts its own.
        Same height and the strict tolerance when the new label or the candidate's label is
        strict (I/l & co), unless no look-alike cluster exists nearby (rival_free)."""
        cands: list[tuple[float, str, str | None]] = []
        for r, ckey, dh in self.near_candidates(bits, key=key):
            shape = self.shapes[ckey]
            if self.shapes[shape.cluster].variant(top_rel, self.pos_tol) is None:
                continue
            lab = self._label_of(shape, top_rel)
            tol = tol_for(label, lab)
            if dh and (is_strict(label) or is_strict(lab)):
                # one pixel of height: jitter, when within the strict tolerance and no cluster
                # reading differently is nearby (never a wider tolerance, see near_match)
                want = lab or label
                if r > tol or want is None or (label and lab and label != lab) \
                        or not self.rival_free(bits, top_rel, want, r, exclude=shape.cluster, key=key):
                    continue
            if r <= tol and (letters_in(label, lab) < 2 or local_diff_ratio(bits, shape.bits) <= tol):
                cands.append((r, shape.cluster, lab))
        if not cands:
            return None
        cands.sort(key=lambda c: c[0])
        best = cands[0]
        if label is not None and best[2] is not None and best[2] != label:
            return None               # the cluster reads differently: keep the new shape apart
        other = next((c for c in cands[1:] if c[1] != best[1] and c[2] != best[2]), None)
        if other is not None and other[0] - best[0] < CLUSTER_MARGIN:
            return None
        return best[1]

    def place(self, key: str, bits: np.ndarray, top_rel: int, label: str | None = None,
              derived_from: str | None = None) -> Shape:
        """The shape for a glyph, created (and clustered) if new."""
        shape = self.shapes.get(key)
        if shape is None:
            canon = self.find_cluster(bits, top_rel, label, key=key)
            shape = Shape(key, bits.copy(), derived_from=derived_from or canon, cluster=canon or key)
            self.shapes[key] = shape
            self.by_size[bits.shape].append(key)
            self._tstacks.pop(bits.shape, None)
            self._soft.clear()
            self.dirty = True
        return shape

    def _log(self, *op) -> None:
        if self.journal is not None:
            self.journal.append(op)

    def learn_source(self, source: str) -> bool:
        """Claim a cue image as learned; False if it already was (its votes are not independent)."""
        if source in self.learned_sources:
            return False
        self.learned_sources.add(source)
        self._log("source", source)
        return True

    def join(self, key: str, bits: np.ndarray, top_rel: int, canonical_key: str) -> None:
        """Make a (new) shape a member of an existing shape's cluster, without voting."""
        if key in self.shapes:
            return
        if canonical_key not in self.shapes:
            return
        self._log("join", key, bits, top_rel, canonical_key)
        canon = self.canonical(canonical_key)
        self.shapes[key] = Shape(key, bits.copy(), derived_from=canonical_key, cluster=canon)
        self.by_size[bits.shape].append(key)
        self._tstacks.pop(bits.shape, None)
        self._soft.clear()
        self.dirty = True

    # ---------- learning ----------
    def add_vote(self, key: str, bits: np.ndarray, top_rel: int, label: str, style: str,
                 weight: int = 1, derived_from: str | None = None, once: set | None = None) -> str:
        """One observation of `label` for this glyph. Votes pool on the cluster's canonical shape.
        `once` (per cue) makes several jittered variants of one letter in one cue count once.
        Returns the canonical key."""
        shape = self.place(key, bits, top_rel, label, derived_from)
        canon = self.shapes[shape.cluster]
        v = canon.variant(top_rel, self.pos_tol, create=True)
        if once is not None:
            tag = (canon.key, v.top_rel, label)
            if tag in once:
                # no vote, but the shape (and its variant) now exist: the journal must create
                # them too, or a replayed copy clusters later votes differently
                self._log("touch", key, bits, top_rel, label, derived_from)
                return canon.key
            once.add(tag)
        self._log("vote", key, bits, top_rel, label, style, weight, derived_from)
        v.votes[label] += weight
        if v.votes[label] <= 3:
            self._width_cache = None               # a label may just have been confirmed
            if any(label in c for c in LOOKALIKES):
                self._lookalike_cache.clear()      # a look-alike may just have been confirmed
        if label in ("I", "l"):
            self._il_cache = (-1, False)
        for f in "bi":
            v.styles.setdefault(f, [0, 0])[1 if f in style else 0] += weight
        self.dirty = True
        return canon.key

    def add_prior(self, key: str, bits: np.ndarray, top_rel: int, label: str, source: str) -> None:
        self._log("prior", key, bits, top_rel, label, source)
        shape = self.place(key, bits, top_rel, label, source)
        self.shapes[shape.cluster].variant(top_rel, self.pos_tol, create=True).prior = label
        self.dirty = True

    def soft_canonical(self, key: str, bits: np.ndarray, top_rel: int) -> str:
        """Cluster of a glyph without storing it: its own cluster if known, else the cluster it
        would join (jittered variants of a tick must still form the same '"' sequence)."""
        shape = self.shapes.get(key)
        if shape is not None:
            return shape.cluster
        hit = self._soft.get(key)
        if hit is None:
            hit = self._soft[key] = self.find_cluster(bits, top_rel, key=key) or key
        return hit

    def _is_seq_part(self, bits: np.ndarray) -> bool:
        """Could this glyph be part of a stored sequence? (size within 1 px of a known part)"""
        h, w = bits.shape
        return any((h + dh, w + dw) in self._seq_sizes for dh in (-1, 0, 1) for dw in (-1, 0, 1))

    @property
    def _seq_sizes(self) -> set[tuple[int, int]]:
        sizes = getattr(self, "_seq_sizes_cache", None)
        if sizes is None or self._seq_n != len(self.sequences):
            sizes = set()
            for k in self.sequences:
                for part in k.split("|"):
                    sh = self.shapes.get(part)
                    if sh is not None:
                        sizes.add(sh.bits.shape)
            self._seq_sizes_cache, self._seq_n = sizes, len(self.sequences)
        return sizes

    def seq_key(self, glyphs, stacked: bool | None = None) -> str:
        """Key of a multi-glyph character (two ticks -> '"', three dots -> ...), by cluster.
        Glyphs that cannot be part of any stored sequence keep their own key (no search).
        Parts stacked on top of each other (a colon whose dots did not merge) are marked, so
        the same two dots side by side (an ellipsis) never match a stacked rule."""
        high = all(_high(g) for g in glyphs)
        parts = []
        for g in glyphs:
            if not self._is_seq_part(g.bits):
                parts.append(self.canonical(g.key))
                continue
            canon = self.soft_canonical(g.key, g.bits, g.top_rel)
            if canon not in self._seq_part_clusters(high):
                # Tiny parts (a tick: ~35 px of ink) jitter by a tenth of their ink per pixel, so a
                # rescaled track scatters them over many small clusters and the pair's votes over
                # as many keys. For the key only, a part snaps to the nearest cluster that already
                # forms a sequence at this height; glyph clustering itself stays strict (. vs ,).
                canon = self._nearest_seq_part(g, high) or canon
            parts.append(canon)
        key = "|".join(parts)
        if stacked is None:
            stacked = _stacked(glyphs)
        if stacked:
            key += "^"
        # Shapes carry no position: in a small font the tick of a " and a comma are the same
        # bitmap. A pair sitting well above the baseline (every part's bottom higher than its own
        # height) is a different character from the same pair on the baseline (" vs „), so high
        # sequences get their own key.
        if high:
            key += "'"
        return key

    def _seq_part_clusters(self, high: bool) -> set[str]:
        """Clusters that are parts of a stored sequence at this height (cached per table size)."""
        cache = getattr(self, "_seq_part_cache", None)
        if cache is None or cache[0] != len(self.sequences):
            low, hi = set(), set()
            for k in self.sequences:
                (hi if k.endswith("'") else low).update(k.rstrip("'^").split("|"))
            cache = self._seq_part_cache = (len(self.sequences), low, hi)
        return cache[2] if high else cache[1]

    def _nearest_seq_part(self, g, high: bool) -> str | None:
        wanted = self._seq_part_clusters(high)
        if not wanted:
            return None
        for r, key, dh in self.near_candidates(g.bits, max_ratio=SEQ_PART_TOL, key=g.key):
            cl = self.shapes[key].cluster
            if cl in wanted and self.shapes[cl].variant(g.top_rel, self.pos_tol) is not None:
                return cl
        return None

    def part_labels(self, glyphs) -> list[str | None]:
        """Each glyph's own trusted label (None if unknown or unconfirmed)."""
        out = []
        for g in glyphs:
            v = self.lookup(g.key, g.top_rel)
            out.append(trusted_label(v.votes)[0] if v is not None and v.votes else None)
        return out

    def dropping_sequence(self, glyphs, label: str) -> bool:
        """A rule that reads several glyphs, each confirmed as a character of its own, as fewer
        copies of one of them ('.' '.' -> '.') only records that the vision model miscounted a run
        (two dots of an ellipsis). Two ticks forming a quote or three dots forming an ellipsis
        character are different characters and stay allowed."""
        parts = self.part_labels(glyphs)
        return all(parts) and len(label) < len(glyphs) and label in parts

    def seq_label(self, glyphs) -> str | None:
        """Trusted label of the stored sequence these glyphs form, if any (see dropping_sequence)."""
        votes = self.sequences.get(self.seq_key(glyphs))
        label = trusted_label(votes)[0] if votes else None
        if label is not None and self.dropping_sequence(glyphs, label):
            return None
        return label

    def add_sequence(self, glyphs, label: str, stacked: bool | None = None) -> None:
        """The glyphs together spell `label`. Each part is stored (and clustered) without a label
        of its own, so jittered parts are recognised as the same sequence later."""
        if stacked is None:
            stacked = _stacked(glyphs)
        self._log("seq", [(g.key, g.bits, g.top_rel) for g in glyphs], label, stacked)
        for g in glyphs:
            shape = self.place(g.key, g.bits, g.top_rel)
            # A part carries no votes, but its cluster needs a variant at this position: joining a
            # cluster requires one, so without it every jittered tick of a rescaled track started
            # a new cluster and every jittered quote pair was a sequence never seen before.
            self.shapes[shape.cluster].variant(g.top_rel, self.pos_tol, create=True)
        self.sequences.setdefault(self.seq_key(glyphs, stacked), Counter())[label] += 1
        self.dirty = True

    def add_word(self, keys: list[str], text: str) -> None:
        if not self.word_memory:
            return
        self._log("word", list(keys), text)
        self.words.setdefault("|".join(keys), Counter())[text] += 1
        self.dirty = True

    def add_gap(self, ka: str, kb: str, gap: int, is_space: bool, italic: bool) -> None:
        self._log("gap", ka, kb, gap, is_space, italic)
        self.gaps["1" if italic else "0"][gap][1 if is_space else 0] += 1
        self.pair_gaps[f"{self.canonical(ka)}|{self.canonical(kb)}|{gap}"][1 if is_space else 0] += 1
        self.dirty = True
        self._bearings = None

    def fit_bearings(self):
        """Additive gap model: letter gap(a, b) ~ R[a] + L[b]; a space adds ~space_off on top.
        Kerning-heavy pairs (f w, r y, 1 :) then separate cleanly from real spaces."""
        if self._bearings is not None:
            return self._bearings
        letters, spaces = [], []
        for k, (lc, sc) in self.pair_gaps.items():
            a, b, g = k.split("|")
            if lc:
                letters.append((a, b, int(g), lc))
            if sc:
                spaces.append((a, b, int(g), sc))
        if len(letters) < 20 or len(spaces) < 5:
            self._bearings = (None, None, None, None, None)
            return self._bearings
        R: dict[str, float] = defaultdict(float)
        L: dict[str, float] = defaultdict(float)
        for _ in range(15):
            acc: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
            for a, b, g, c in letters:
                acc[a][0] += (g - L[b]) * c
                acc[a][1] += c
            for a, (sm, n) in acc.items():
                R[a] = sm / n
            acc = defaultdict(lambda: [0.0, 0.0])
            for a, b, g, c in letters:
                acc[b][0] += (g - R[a]) * c
                acc[b][1] += c
            for b, (sm, n) in acc.items():
                L[b] = sm / n
        r_def = float(np.median(list(R.values())))
        l_def = float(np.median(list(L.values())))
        res = []
        for a, b, g, c in spaces:
            res += [g - (R.get(a, r_def) + L.get(b, l_def))] * c
        space_off = float(np.median(res))
        # Letters that rarely follow (or precede) another letter inside a word have no bearing from
        # the letter pairs: in English a 'j' almost always starts a word. Its hook reaches left
        # under the previous letter, so the default bearing is off by pixels and every "is just"
        # lands in the uncertain band. With the space offset known, space pairs give the bearing.
        missing_l: dict[str, list[float]] = defaultdict(list)
        missing_r: dict[str, list[float]] = defaultdict(list)
        for a, b, g, c in spaces:
            if b not in L and a in R:
                missing_l[b] += [g - R[a] - space_off] * c
            if a not in R and b in L:
                missing_r[a] += [g - L[b] - space_off] * c
        for b, vals in missing_l.items():
            if len(vals) >= 2:
                L[b] = float(np.median(vals))
        for a, vals in missing_r.items():
            if len(vals) >= 2:
                R[a] = float(np.median(vals))
        self._bearings = (dict(R), dict(L), r_def, l_def, space_off)
        return self._bearings

    # ---------- queries ----------
    def lookup(self, key: str, top_rel: int) -> Variant | None:
        shape = self.shapes.get(key)
        return self.shapes[shape.cluster].variant(top_rel, self.pos_tol) if shape else None

    def classify_gap(self, ka: str, kb: str, gap: int, italic: bool) -> bool | None:
        """True = space, False = no space, None = uncertain."""
        pl, ps = self.pair_gaps.get(f"{self.canonical(ka)}|{self.canonical(kb)}|{gap}", (0, 0))
        if pl + ps >= 2:          # a single observation may stem from a misaligned cue
            if pl and not ps:
                return False
            if ps and not pl:
                return True
        R, L, r_def, l_def, space_off = self.fit_bearings()
        if R is not None and space_off > 4:
            # bearings are fitted per cluster: a jittered member shares its canonical's bearings
            ca, cb = self.canonical(ka), self.canonical(kb)
            known = (ca in R) + (cb in L)
            r = gap - (R.get(ca, r_def) + L.get(cb, l_def))
            margin = max(2.0, (0.2 if known == 2 else 0.3) * space_off)
            if r < space_off / 2 - margin:
                return False
            if r > space_off / 2 + margin:
                return True
            return None
        stats = self.gaps["1" if italic else "0"]
        if not stats:
            stats = self.gaps["0" if italic else "1"]
        if not stats:
            return None
        l, s = stats.get(gap, (0, 0))
        if l and not s:
            return False
        if s and not l:
            return True
        if l and s:
            if l >= 10 * s:
                return False
            if s >= 10 * l:
                return True
            return None
        letter_gaps = [g for g, (lc, _) in stats.items() if lc]
        space_gaps = [g for g, (_, sc) in stats.items() if sc]
        max_l = max(letter_gaps, default=None)
        min_s = min(space_gaps, default=None)
        if max_l is not None and gap <= max_l and (min_s is None or gap < min_s):
            return False
        if min_s is not None and gap >= min_s and (max_l is None or gap > max_l):
            return True
        return None

    def char_widths(self) -> dict[str, float]:
        """Median glyph width per single-character label (for alignment priors)."""
        acc: dict[str, list[int]] = defaultdict(list)
        for s in self.shapes.values():
            for v in s.variants:
                if v.votes:
                    lab = v.votes.most_common(1)[0][0]
                    acc[lab].append(s.bits.shape[1])
        return {k: float(np.median(v)) for k, v in acc.items()}

    # ---------- persistence ----------
    def _settled(self, v: Variant, key: str, main_parts: set[str]) -> bool:
        """Does this glyph position belong in the main file? Confirmed (trusted_label), a teacher's
        label, the font's I/l pair read at least twice (the word decides the letter), a part of a
        settled sequence (parts carry no votes), or kept from a 0.3.0+ main file."""
        if v.kept or v.prior:
            return True
        votes = +v.votes
        if not votes:
            return key in main_parts
        if trusted_label(votes)[0] is not None:
            return True
        from .recognize import confusable, voted_labels
        return sum(votes.values()) >= MIN_VOTES and confusable(voted_labels(votes))

    @staticmethod
    def _seq_parts(keys) -> set[str]:
        return {part for k in keys for part in k.rstrip("'^").split("|")}

    def _split(self) -> tuple[list, list, dict, dict]:
        """(main shapes, interim shapes, main sequences, interim sequences) as JSON records. A shape
        whose cluster has settled and unsettled positions appears in both, each with its own
        variants; jitter members go where their cluster's settled positions are (else interim)."""
        main_seqs, interim_seqs = {}, {}
        for k, v in self.sequences.items():
            settled = k in self._kept_seqs or trusted_label(v)[0] is not None
            (main_seqs if settled else interim_seqs)[k] = dict(v)
        main_parts = self._seq_parts(main_seqs)
        split: dict[str, tuple[list, list]] = {}
        for sh in self.shapes.values():
            if sh.variants:
                mv = [v for v in sh.variants if self._settled(v, sh.key, main_parts)]
                split[sh.key] = (mv, [v for v in sh.variants if not any(v is m for m in mv)])
        main_clusters = {k for k, (mv, _) in split.items() if mv} | (main_parts & self.shapes.keys())
        main, interim = [], []
        for sh in self.shapes.values():
            if sh.key in split:
                mv, iv = split[sh.key]
                if mv or sh.key in main_clusters:
                    main.append(self._shape_json(sh, mv, proto=True))
                    if iv:
                        interim.append(self._shape_json(sh, iv, proto=False))
                else:
                    interim.append(self._shape_json(sh, iv, proto=True))
            elif sh.cluster in main_clusters:
                main.append(self._shape_json(sh, [], proto=True))
            else:
                interim.append(self._shape_json(sh, [], proto=True))
        return main, interim, main_seqs, interim_seqs

    def _shape_json(self, s: "Shape", variants: list[Variant], proto: bool) -> dict:
        return {
            "key": s.key, "h": int(s.bits.shape[0]), "w": int(s.bits.shape[1]),
            "bits": base64.b64encode(np.packbits(s.bits).tobytes()).decode(),
            "derived_from": s.derived_from,
            "cluster": s.cluster if s.cluster != s.key else None,
            "n": s.n,
            "proto": _proto_json(self.protos.get(s.key)) if proto and s.key in self.protos else None,
            "variants": [{"top_rel": v.top_rel, "votes": dict(v.votes), "styles": v.styles,
                          "geo_italic": v.geo_italic, "geo_bold": v.geo_bold, "prior": v.prior}
                         for v in variants],
        }

    def to_json(self, parts: tuple | None = None) -> dict:
        """The main file: settled glyphs only (publishable). The interim part is interim_json()."""
        main, _, main_seqs, _ = parts or self._split()
        return {
            "name": self.name,
            "version": 2,
            "layout": 3,               # main file holds settled glyphs only; interim learning is private
            "learned_with": self.learned_with,
            "saved_with": app_version(),
            "merged_from": self.merged_from,
            "seeded_from": self.seeded_from,
            "charset": self.charset,
            "unit": self.unit,
            "parent": self.parent,
            "shapes": main,
            "sequences": main_seqs,
            "gaps": {it: {str(g): c for g, c in d.items()} for it, d in self.gaps.items()},
            "pair_gaps": dict(self.pair_gaps),
        }

    def interim_json(self, parts: tuple | None = None) -> dict:
        _, interim, _, interim_seqs = parts or self._split()
        return {"shapes": interim, "sequences": interim_seqs}

    def merge_interim(self, d: dict) -> None:
        """Add a sidecar's interim part to this set: shapes it does not hold are created, the votes
        of a position join the cluster the bitmap belongs to here. Exact positions only: the split
        never separates a position from itself, and nearby positions are distinct variants."""
        recs = sorted(d.get("shapes", []), key=lambda r: r.get("cluster") is not None)   # canonical first
        for r in recs:
            key = r["key"]
            sh = self.shapes.get(key)
            if sh is None:
                bits = _unpack(r["bits"], r["h"], r["w"])
                cl = r.get("cluster") or key
                sh = Shape(key, bits, derived_from=r.get("derived_from"),
                           cluster=cl if cl in self.shapes else key, n=int(r.get("n", 1)))
                self.shapes[key] = sh
                self.by_size[bits.shape].append(key)
                pj = r.get("proto")
                if pj and key not in self.protos:
                    self.protos[key] = Proto(n=int(pj["n"]), median=_unpack(pj["median"], pj["h"], pj["w"]),
                                             stable=_unpack(pj["stable"], pj["h"], pj["w"]))
            canon = self.shapes[sh.cluster]
            for v in r["variants"]:
                tv = canon.variant(v["top_rel"], 0)
                if tv is None:
                    canon.variants.append(Variant(v["top_rel"], Counter(v["votes"]), v.get("styles", {}),
                                                  list(v.get("geo_italic", [0, 0])), list(v.get("geo_bold", [0, 0])),
                                                  v.get("prior")))
                    continue
                tv.votes.update(v["votes"])
                for f, (no, yes) in (v.get("styles") or {}).items():
                    st = tv.styles.setdefault(f, [0, 0])
                    st[0] += no
                    st[1] += yes
                for mine, theirs in ((tv.geo_italic, v.get("geo_italic", [0, 0])), (tv.geo_bold, v.get("geo_bold", [0, 0]))):
                    mine[0] += theirs[0]
                    mine[1] += theirs[1]
                tv.prior = tv.prior or v.get("prior")
        for k, v in d.get("sequences", {}).items():
            self.sequences.setdefault(k, Counter()).update(v)
        self._stacks.clear(); self._tstacks.clear(); self._soft.clear(); self._near_cache.clear()
        self._pstacks = None
        self._width_cache = None
        self._lookalike_cache.clear()
        self._il_cache = (-1, False)
        self._seq_n = -1
        self._seq_part_cache = None

    def load_interim(self, private_dir: Path) -> None:
        """Merge the interim part of <private_dir>/<name>.json (once per set and sidecar)."""
        path = private_dir / f"{self.path.stem if self.path else self.name}.json"
        if self._interim_from == path or not path.exists():
            return
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("interim"):
            self.merge_interim(d["interim"])
        self._interim_from = path

    @classmethod
    def load(cls, path: Path) -> "GlyphDB":
        d = json.loads(path.read_text(encoding="utf-8"))
        db = cls(d["name"], path)
        db.unit = d.get("unit")
        db.parent = d.get("parent")
        db.learned_with = d.get("learned_with")
        db.merged_from = list(d.get("merged_from") or [])
        db.seeded_from = d.get("seeded_from")
        db.charset = d.get("charset", "literal")     # DBs from before simplification are literal
        layout = int(d.get("layout", 0))             # 3: settled glyphs only (0.3.0+); older files hold everything
        db.learned_sources = set(d.get("learned_sources", []))      # legacy inline
        for s in d["shapes"]:
            h, w = s["h"], s["w"]
            bits = np.unpackbits(np.frombuffer(base64.b64decode(s["bits"]), np.uint8))[:h * w].reshape(h, w).astype(bool)
            shape = Shape(s["key"], bits, derived_from=s.get("derived_from"), cluster=s.get("cluster") or s["key"],
                          n=int(s.get("n", 2)))      # legacy shapes without a count are kept
            for v in s["variants"]:
                shape.variants.append(Variant(v["top_rel"], Counter(v["votes"]), v.get("styles", {}),
                                              v.get("geo_italic", [0, 0]), v.get("geo_bold", [0, 0]),
                                              v.get("prior"), kept=layout >= 3))
            db.shapes[shape.key] = shape
            db.by_size[bits.shape].append(shape.key)
            pj = s.get("proto")
            if pj:
                db.protos[shape.key] = Proto(n=int(pj["n"]), median=_unpack(pj["median"], pj["h"], pj["w"]),
                                             stable=_unpack(pj["stable"], pj["h"], pj["w"]))
        for shape in db.shapes.values():          # a member whose canonical is gone stands alone
            if shape.cluster not in db.shapes:
                shape.cluster = shape.key
        db.sequences = {k: Counter(v) for k, v in d.get("sequences", {}).items()}
        if layout >= 3:
            db._kept_seqs = set(db.sequences)
        for it, gd in d.get("gaps", {}).items():
            for g, c in gd.items():
                db.gaps[it][int(g)] = list(c)
        for k, c in d.get("pair_gaps", {}).items():
            db.pair_gaps[k] = list(c)
        # legacy DBs kept words/sources inline: picked up here, moved to the sidecar on save
        db.words = {k: Counter(v) for k, v in d.get("words", {}).items()}
        db.dirty = bool(d.get("words") or d.get("learned_sources"))   # rewrite once to migrate them out
        return db

    def attach_private(self, private_dir: Path, word_memory: bool) -> None:
        """Load (or start) the private sidecar <word-memory dir>/<name>.json."""
        self.private_path = private_dir / f"{self.path.stem if self.path else self.name}.json"
        self.word_memory = word_memory
        if self.private_path.exists():
            d = json.loads(self.private_path.read_text(encoding="utf-8"))
            if d.get("interim") and self._interim_from != self.private_path:
                self.merge_interim(d["interim"])
            self._interim_from = self.private_path
            self.learned_sources |= set(d.get("learned_sources", []))
            for k, v in d.get("words", {}).items():
                self.words.setdefault(k, Counter()).update(v)
            for k, pj in d.get("protos", {}).items():        # the training corpus: per-pixel counts
                if k in self.shapes:
                    acc = np.frombuffer(base64.b64decode(pj["acc"]), np.uint16).astype(np.int32).reshape(pj["h"], pj["w"])
                    self.protos[k] = Proto(acc=acc.copy(), n=int(pj["n"]))
            self._pstacks = None
        if not word_memory:
            self.words = {}

    def replay(self, ops: list[tuple]) -> None:
        """Apply another copy's learning journal. Votes, sequences and gaps of a cue image this
        copy already learned from are skipped (evidence per image counts once)."""
        saved, self.journal = self.journal, None
        skip = False
        try:
            for op in ops:
                kind = op[0]
                if kind == "source":
                    skip = not self.learn_source(op[1])
                elif kind in ("vote", "seq", "gap") and skip:
                    continue
                elif kind == "vote":
                    _, key, bits, top_rel, label, style, weight, derived_from = op
                    self.add_vote(key, bits, top_rel, label, style, weight, derived_from)
                elif kind == "touch":
                    _, key, bits, top_rel, label, derived_from = op
                    shape = self.place(key, bits, top_rel, label, derived_from)
                    self.shapes[shape.cluster].variant(top_rel, self.pos_tol, create=True)
                elif kind == "seq":
                    self.add_sequence([_Part(k, b, t) for k, b, t in op[1]], op[2], op[3] if len(op) > 3 else False)
                elif kind == "gap":
                    self.add_gap(*op[1:])
                elif kind == "prior":
                    self.add_prior(*op[1:])
                elif kind == "join":
                    self.join(*op[1:])
                elif kind == "word":
                    self.add_word(*op[1:])
                elif kind == "obs":
                    self.observe(*op[1:])
        finally:
            self.journal = saved

    def apply_file_geo(self, file_geo: dict, file_geo_b: dict, credited: set[str]) -> None:
        """Add one file's geometry votes to the accumulated votes (pooled per cluster). Clusters in
        `credited` got them at an earlier save; the ones credited now are added to the set, so a
        shape learned later in the run still receives the file's votes exactly once."""
        members: dict[str, list[str]] = {}
        for k in set(file_geo) | set(file_geo_b):
            if k in self.shapes:
                members.setdefault(self.canonical(k), []).append(k)
        for canon, keys in members.items():
            if canon in credited:
                continue
            credited.add(canon)
            gi, gb = [0, 0], [0, 0]
            for k in keys:
                fi, fb = file_geo.get(k, [0, 0]), file_geo_b.get(k, [0, 0])
                gi[0] += fi[0]; gi[1] += fi[1]; gb[0] += fb[0]; gb[1] += fb[1]
            for v in self.shapes[canon].variants:
                v.geo_italic = [v.geo_italic[0] + gi[0], v.geo_italic[1] + gi[1]]
                v.geo_bold = [v.geo_bold[0] + gb[0], v.geo_bold[1] + gb[1]]
        self.dirty = True

    @property
    def il_identical(self) -> bool:
        """The font draws I and l alike: some cluster carries confirmed votes for both. Then a
        cluster voted 'l' alone cannot be trusted to be an l either (recognize._decide)."""
        n = len(self.shapes)
        if self._il_cache[0] != n:
            hit = any(v.votes.get("I", 0) >= 2 and v.votes.get("l", 0) >= 2
                      for sh in self.shapes.values() for v in sh.variants)
            if not hit:
                # separate clusters (a one-pixel width jitter apart) voted I and l respectively,
                # at the same height: the font draws them alike
                sizes = {"I": set(), "l": set()}
                for sh in self.shapes.values():
                    for v in sh.variants:
                        lab = trusted_label(v.votes)[0]
                        if lab in sizes:
                            sizes[lab].add(sh.bits.shape)
                hit = any(abs(hi - hl) <= 1 and abs(wi - wl) <= 1
                          for hi, wi in sizes["I"] for hl, wl in sizes["l"])
            self._il_cache = (n, hit)
        return self._il_cache[1]

    # ---------- prototypes ----------
    def observe(self, canon: str, bits: np.ndarray, weight: int = 1) -> None:
        """Add `weight` samples of a glyph to its cluster's prototype (the training step)."""
        sh = self.shapes.get(canon)
        if sh is None or weight <= 0:
            return
        H, W = sh.bits.shape[0] + 2, sh.bits.shape[1] + 2
        p = self.protos.get(canon)
        if p is None or p.acc is None or p.acc.shape != (H, W):
            acc = np.zeros((H, W), np.int32)
            n = 0
            if p is not None and p._median is not None and p._median.shape == (H, W):
                acc += p._median.astype(np.int32) * p.n     # public median only: seed the counts
                n = p.n
            p = self.protos[canon] = Proto(acc=acc, n=n)
        h, w = bits.shape
        if h > H or w > W:
            return
        target = p.median if p.n else np.pad(sh.bits, 1)
        best, best_a = None, None
        for dy in range(H - h + 1):
            for dx in range(W - w + 1):
                a = np.zeros((H, W), bool)
                a[dy:dy + h, dx:dx + w] = bits
                d = int((a ^ target).sum())
                if best is None or d < best:
                    best, best_a = d, a
        p.acc += best_a.astype(np.int32) * weight
        p.n += weight
        self._pstacks = None
        self._log("obs", canon, bits, weight)
        self.dirty = True

    def proto_candidates(self, bits: np.ndarray, max_ratio: float = 0.2) -> list[tuple[float, str, int]]:
        """Clusters whose prototype the glyph fits: (disagreements on stable pixels / stable ink,
        canonical key, disagreeing pixels), nearest first. Only prototypes with >= PROTO_MIN
        samples take part."""
        if self._pstacks is None:
            groups: dict[tuple[int, int], list] = {}
            for k, p in self.protos.items():
                if p.n >= PROTO_MIN and k in self.shapes:
                    groups.setdefault(p.median.shape, []).append(k)
            self._pstacks = {size: (keys, np.stack([self.protos[k].median for k in keys]),
                                    np.stack([self.protos[k].stable for k in keys]))
                             for size, keys in groups.items()}
        h, w = bits.shape
        out: list[tuple[float, str]] = []
        for (H, W), (keys, M, S) in self._pstacks.items():
            if not (h <= H <= h + 3 and w <= W <= w + 3):     # canvas = shape + 2, sample within +-1
                continue
            ink = np.maximum(1, (M & S).reshape(len(keys), -1).sum(axis=1))
            best = None
            for dy in range(H - h + 1):
                for dx in range(W - w + 1):
                    a = np.zeros((H, W), bool)
                    a[dy:dy + h, dx:dx + w] = bits
                    d = ((a[None] ^ M) & S).reshape(len(keys), -1).sum(axis=1)
                    best = d if best is None else np.minimum(best, d)
            ratios = best / ink
            out += [(float(r), k, int(d)) for r, k, d in zip(ratios, keys, best) if r <= max_ratio]
        out.sort()
        return out

    def credit_counts(self, counts: dict[str, int], credited: set[str]) -> None:
        """Add one file's glyph occurrences to Shape.n, each key once per run (a shape starts at 1)."""
        for k, n in counts.items():
            sh = self.shapes.get(k)
            if sh is not None and k not in credited:
                credited.add(k)
                sh.n += max(0, n - 1)

    def prune_members(self) -> int:
        """Drop cluster members that carry no information of their own: jittered variants seen only
        once (a rescaled track produces thousands per episode, most never recur) and, per cluster,
        all but the MAX_MEMBERS most frequent members. Votes, sequences and gaps live on canonical
        shapes and are untouched; a dropped bitmap is still read via the near search."""
        groups: dict[str, list[Shape]] = {}
        for sh in self.shapes.values():
            if sh.cluster != sh.key and not sh.variants:
                groups.setdefault(sh.cluster, []).append(sh)
        drop: list[str] = []
        for members in groups.values():
            keep = sorted((m for m in members if m.n >= MIN_MEMBER_SEEN), key=lambda m: -m.n)
            drop += [m.key for m in members if m.n < MIN_MEMBER_SEEN] + [m.key for m in keep[MAX_MEMBERS:]]
        for k in drop:
            sh = self.shapes.pop(k)
            bucket = self.by_size.get(sh.bits.shape)
            if bucket and k in bucket:
                bucket.remove(k)
            self._stacks.pop(sh.bits.shape, None)
            self._tstacks.pop(sh.bits.shape, None)
        if drop:
            self._soft.clear()
            self._near_cache.clear()        # results may name dropped shapes; stamps are bucket lengths
            self.dirty = True
        return len(drop)

    def _adopt(self, other: "GlyphDB") -> None:
        """Continue with another copy's state (after merging into it)."""
        keep = {"journal", "path", "private_path", "word_memory", "file_geo", "_geo_credited",
                "file_counts", "_counts_credited", "saved_this_run", "tolerant", "rescaled"}
        for k, v in other.__dict__.items():
            if k not in keep:
                setattr(self, k, v)
        self._bearings = None
        self._seq_n = -1

    def save(self, path: Path | None = None, final: bool = False) -> None:
        """Write the DB. If the file exists, this copy's learning is merged into the file's current
        content under a file lock: several workers may learn into the same glyph set at once, each
        replaying its journal onto what the others wrote (all writes are additive counters).
        `final` (the run's last save) also prunes jitter members; mid-run pruning would remove
        bitmaps the same file still needs."""
        path = path or self.path
        assert path is not None
        with _locked(path):
            if path.exists() and self.journal is not None:
                other = GlyphDB.load(path)
                if self.private_path is not None:
                    other.attach_private(self.private_path.parent, self.word_memory)
                other.replay(self.journal)
                other.apply_file_geo(*self.file_geo, self._geo_credited)
                other.credit_counts(self.file_counts, self._counts_credited)
                if final:
                    other.prune_members()
                other._write(path)
                self._adopt(other)
            else:
                # first write of a new DB: the in-memory variants already carry this file's votes
                # (pipeline.apply_geo), so every cluster present now counts as credited
                self._geo_credited |= {self.canonical(k) for k in set(self.file_geo[0]) | set(self.file_geo[1])
                                       if k in self.shapes}
                self.credit_counts(self.file_counts, self._counts_credited)
                if final:
                    self.prune_members()
                self._write(path)
        self.journal = [] if self.journal is not None else None
        self.saved_this_run = True

    def _write(self, path: Path) -> None:
        self.update_unit()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        parts = self._split()
        tmp.write_text(json.dumps(self.to_json(parts), ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        self.path = path
        if self.private_path is not None:
            self.private_path.parent.mkdir(parents=True, exist_ok=True)
            priv = {"db": self.name, "learned_sources": sorted(self.learned_sources),
                    "interim": self.interim_json(parts),
                    "protos": {k: {"h": p.acc.shape[0], "w": p.acc.shape[1], "n": p.n,
                                   "acc": base64.b64encode(np.minimum(p.acc, 65535).astype(np.uint16).tobytes()).decode()}
                               for k, p in self.protos.items() if p.acc is not None and k in self.shapes}}
            if self.word_memory:
                priv["words"] = {k: dict(v) for k, v in self.words.items()}
            tmp = self.private_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(priv, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.private_path)
        self.dirty = False
