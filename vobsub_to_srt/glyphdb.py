"""Per-font glyph database: exact bitmaps with voted labels, sequence rules and gap statistics."""
from __future__ import annotations

import base64
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

POS_TOL_UNITS = 0.05   # tolerance on top_rel (baseline jitter), in font units (x-height)
XHEIGHT_CHARS = set("acemnorsuvwxz")   # lowercase letters without ascender/descender
MIN_VOTES = 2          # a label needs this many independent observations (cues) ...
MIN_SHARE = 2 / 3      # ... and at least this share of all votes before it is trusted
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
CLUSTER_MARGIN = 0.06   # lead over the nearest shape with a different label
STRICT = set("Il|i1!jíìïî")   # punctuation is told apart by size and baseline position instead


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


def is_strict(label: str | None) -> bool:
    return bool(label) and bool(set(label) & STRICT)


def tol_for(*labels: str | None) -> float:
    return STRICT_TOL if any(is_strict(l) for l in labels) else CLUSTER_TOL


def trusted_label(votes: Counter) -> tuple[str | None, str]:
    """Majority label if confirmed (>= MIN_VOTES and >= MIN_SHARE of votes), else (None, reason)."""
    if not votes:
        return None, "no votes"
    total = sum(votes.values())
    top, n1 = votes.most_common(1)[0]
    if total < MIN_VOTES:
        return None, f"unconfirmed ({top!r} seen once)"
    if n1 < MIN_SHARE * total:
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
        self._stacks: dict[tuple[int, int], tuple[list[str], np.ndarray]] = {}   # bucket -> (keys, bits)
        self._soft: dict[str, str] = {}           # soft_canonical cache, cleared when shapes are added
        self.dirty = False
        self._bearings: tuple | None = None   # fitted side-bearing model (cached)
        self._seq_n = -1                      # sequences count the _seq_sizes cache was built for
        # font unit: x-height in px of this DB's raster. Everything compared across rasters is
        # expressed in this unit; within the raster, exact bitmaps are used.
        self.unit: float | None = None
        self.parent: str | None = None     # teacher DB this one was bootstrapped from (other raster)
        # hashes of cue images already learned from: re-reading the same image (re-runs, duplicate
        # cues) is not independent evidence and must not confirm a label
        self.learned_sources: set[str] = set()
        # Private sidecar (never part of the publishable DB): source hashes always, the word
        # memory only if enabled. None = not attached (e.g. a read-only teacher DB).
        self.private_path: Path | None = None
        self.word_memory = False
        self.charset = "simplified"          # labels folded by simplify.py, or "literal"

    @property
    def pos_tol(self) -> int:
        return max(1, round(POS_TOL_UNITS * self.unit)) if self.unit else 2

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

    def near_candidates(self, bits: np.ndarray, max_ratio: float = CLUSTER_TOL) -> list[tuple[float, str, int]]:
        """All stored shapes of the same size +-1 px within `max_ratio` of `bits`, nearest first,
        as (ratio, key, dh). Vectorised per size bucket: the bucket's bitmaps are stacked once and
        compared for all 3x3 alignments at a time."""
        h, w = bits.shape
        out: list[tuple[float, str, int]] = []
        na = int(bits.sum())
        for dh in (-1, 0, 1):
            for dw in (-1, 0, 1):
                size = (h + dh, w + dw)
                keys = self.by_size.get(size)
                if not keys:
                    continue
                st = self._stacks.get(size)
                if st is None or len(st[0]) != len(keys):
                    st = (list(keys), np.stack([self.shapes[k].bits for k in keys]))
                    self._stacks[size] = st
                skeys, stack = st
                H, W = max(h, size[0]) + 2, max(w, size[1]) + 2
                A = np.zeros((H, W), bool)
                A[1:1 + h, 1:1 + w] = bits
                nb = stack.reshape(len(skeys), -1).sum(axis=1)
                best = None
                for dy in (0, 1, 2):
                    for dx in (0, 1, 2):
                        if dy + size[0] > H or dx + size[1] > W:
                            continue
                        B = np.zeros((len(skeys), H, W), bool)
                        B[:, dy:dy + size[0], dx:dx + size[1]] = stack
                        d = (A[None] ^ B).reshape(len(skeys), -1).sum(axis=1)
                        best = d if best is None else np.minimum(best, d)
                ratios = best / np.maximum(1.0, (na + nb) / 2)
                for i in np.nonzero(ratios <= max_ratio)[0]:
                    out.append((float(ratios[i]), skeys[i], dh))
        out.sort(key=lambda c: c[0])
        return out

    def find_cluster(self, bits: np.ndarray, top_rel: int, label: str | None = None) -> str | None:
        """Canonical key of the cluster a new shape belongs to, or None if it starts its own.
        Same height only when the new label or the candidate's label is strict (I/l & co)."""
        cands: list[tuple[float, str, str | None]] = []
        for r, key, dh in self.near_candidates(bits):
            shape = self.shapes[key]
            if self.shapes[shape.cluster].variant(top_rel, self.pos_tol) is None:
                continue
            lab = self._label_of(shape, top_rel)
            if dh and (is_strict(label) or is_strict(lab)):
                continue
            if r <= tol_for(label, lab):
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
            canon = self.find_cluster(bits, top_rel, label)
            shape = Shape(key, bits.copy(), derived_from=derived_from or canon, cluster=canon or key)
            self.shapes[key] = shape
            self.by_size[bits.shape].append(key)
            self._stacks.pop(bits.shape, None)
            self._soft.clear()
            self.dirty = True
        return shape

    def join(self, key: str, bits: np.ndarray, top_rel: int, canonical_key: str) -> None:
        """Make a (new) shape a member of an existing shape's cluster, without voting."""
        if key in self.shapes:
            return
        canon = self.canonical(canonical_key)
        self.shapes[key] = Shape(key, bits.copy(), derived_from=canonical_key, cluster=canon)
        self.by_size[bits.shape].append(key)
        self._stacks.pop(bits.shape, None)
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
                return canon.key
            once.add(tag)
        v.votes[label] += weight
        for f in "bi":
            v.styles.setdefault(f, [0, 0])[1 if f in style else 0] += weight
        self.dirty = True
        return canon.key

    def add_prior(self, key: str, bits: np.ndarray, top_rel: int, label: str, source: str) -> None:
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
            hit = self._soft[key] = self.find_cluster(bits, top_rel) or key
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

    def seq_key(self, glyphs) -> str:
        """Key of a multi-glyph character (two ticks -> '"', three dots -> ...), by cluster.
        Glyphs that cannot be part of any stored sequence keep their own key (no search)."""
        return "|".join(self.soft_canonical(g.key, g.bits, g.top_rel) if self._is_seq_part(g.bits) else
                        self.canonical(g.key) for g in glyphs)

    def add_sequence(self, glyphs, label: str) -> None:
        """The glyphs together spell `label`. Each part is stored (and clustered) without a label
        of its own, so jittered parts are recognised as the same sequence later."""
        for g in glyphs:
            self.place(g.key, g.bits, g.top_rel)
        self.sequences.setdefault(self.seq_key(glyphs), Counter())[label] += 1
        self.dirty = True

    def add_word(self, keys: list[str], text: str) -> None:
        if not self.word_memory:
            return
        self.words.setdefault("|".join(keys), Counter())[text] += 1
        self.dirty = True

    def add_gap(self, ka: str, kb: str, gap: int, is_space: bool, italic: bool) -> None:
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
        self._bearings = (dict(R), dict(L), r_def, l_def, space_off)
        return self._bearings

    # ---------- queries ----------
    def lookup(self, key: str, top_rel: int) -> Variant | None:
        shape = self.shapes.get(key)
        return self.shapes[shape.cluster].variant(top_rel, self.pos_tol) if shape else None

    def classify_gap(self, ka: str, kb: str, gap: int, italic: bool) -> bool | None:
        """True = space, False = no space, None = uncertain."""
        pl, ps = self.pair_gaps.get(f"{self.canonical(ka)}|{self.canonical(kb)}|{gap}", (0, 0))
        if pl or ps:
            if pl and not ps:
                return False
            if ps and not pl:
                return True
        R, L, r_def, l_def, space_off = self.fit_bearings()
        if R is not None and space_off > 4:
            known = (ka in R) + (kb in L)
            r = gap - (R.get(ka, r_def) + L.get(kb, l_def))
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
    def to_json(self) -> dict:
        return {
            "name": self.name,
            "version": 2,
            "charset": self.charset,
            "unit": self.unit,
            "parent": self.parent,
            "shapes": [{
                "key": s.key, "h": int(s.bits.shape[0]), "w": int(s.bits.shape[1]),
                "bits": base64.b64encode(np.packbits(s.bits).tobytes()).decode(),
                "derived_from": s.derived_from,
                "cluster": s.cluster if s.cluster != s.key else None,
                "variants": [{"top_rel": v.top_rel, "votes": dict(v.votes), "styles": v.styles,
                              "geo_italic": v.geo_italic, "geo_bold": v.geo_bold, "prior": v.prior}
                             for v in s.variants],
            } for s in self.shapes.values()],
            "sequences": {k: dict(v) for k, v in self.sequences.items()},
            "gaps": {it: {str(g): c for g, c in d.items()} for it, d in self.gaps.items()},
            "pair_gaps": dict(self.pair_gaps),
        }

    @classmethod
    def load(cls, path: Path) -> "GlyphDB":
        d = json.loads(path.read_text(encoding="utf-8"))
        db = cls(d["name"], path)
        db.unit = d.get("unit")
        db.parent = d.get("parent")
        db.charset = d.get("charset", "literal")     # DBs from before simplification are literal
        db.learned_sources = set(d.get("learned_sources", []))      # legacy inline
        for s in d["shapes"]:
            h, w = s["h"], s["w"]
            bits = np.unpackbits(np.frombuffer(base64.b64decode(s["bits"]), np.uint8))[:h * w].reshape(h, w).astype(bool)
            shape = Shape(s["key"], bits, derived_from=s.get("derived_from"), cluster=s.get("cluster") or s["key"])
            for v in s["variants"]:
                shape.variants.append(Variant(v["top_rel"], Counter(v["votes"]), v.get("styles", {}),
                                              v.get("geo_italic", [0, 0]), v.get("geo_bold", [0, 0]),
                                              v.get("prior")))
            db.shapes[shape.key] = shape
            db.by_size[bits.shape].append(shape.key)
        for shape in db.shapes.values():          # a member whose canonical is gone stands alone
            if shape.cluster not in db.shapes:
                shape.cluster = shape.key
        db.sequences = {k: Counter(v) for k, v in d.get("sequences", {}).items()}
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
            self.learned_sources |= set(d.get("learned_sources", []))
            for k, v in d.get("words", {}).items():
                self.words.setdefault(k, Counter()).update(v)
        if not word_memory:
            self.words = {}

    def save(self, path: Path | None = None) -> None:
        path = path or self.path
        self.update_unit()
        assert path is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_json(), ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        self.path = path
        if self.private_path is not None:
            self.private_path.parent.mkdir(parents=True, exist_ok=True)
            priv = {"db": self.name, "learned_sources": sorted(self.learned_sources)}
            if self.word_memory:
                priv["words"] = {k: dict(v) for k, v in self.words.items()}
            tmp = self.private_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(priv, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.private_path)
        self.dirty = False
