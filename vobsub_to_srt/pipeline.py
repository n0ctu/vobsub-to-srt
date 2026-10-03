"""Probe -> train -> infer -> retry -> finalize."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import time
from collections import Counter
from typing import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .align import ALREADY_LEARNED, align_cue, geometry_spaces, learn_cue, otsu_threshold, quotes_by_position, reline, reline_cues, restyle, strip_tags
from . import transfer
from .glyphdb import GlyphDB, combine_style

RESCALED_ONCE_SHARE = 0.5   # share of glyph bitmaps occurring once above which a track counts as rescaled
from .lexicon import make_lexicon
from .names import random_db_name
from .simplify import CHARSET_LITERAL, CHARSET_SIMPLIFIED, simplify
from .recognize import CueResult, near_match, recognize, _decide
from .segment import Line, bold_votes, fill_mask, fill_mask_ex, italic_votes, segment, fill_values
from .srt import fmt_ts, normalize_text, render_srt
from .vlm import VLMClient, mask_to_png, render_cue_png, sheet_png, split_sheet
from .vobsub import Cue, load_vobsub, load_vobsub_bytes

log = logging.getLogger("vobsub_to_srt")


@dataclass
class VobSubData:
    """An .idx/.sub pair held in memory."""
    name: str          # display name / output stem
    idx: bytes
    sub: bytes


@dataclass
class ProcessResult:
    srt: str
    report: dict


@dataclass
class Options:
    db_dir: Path = Path("glyph-memory")
    out_dir: Path | None = None          # write <stem>.srt and <stem>.report.json here; None = memory only
    debug_dir: Path | None = None        # PNGs of cues that could not be read (diagnostics)
    diagnostics: bool = False            # keep raw VLM answers in the report
    batch_size: int = 16
    sheet: int = 1                       # cues per VLM request, stacked into one image (1 = off)
    render: str = "mask2"                # image sent to the VLM (vlm.RENDERS); strict re-asks use the 3x mask
    mode: str = "hybrid"                 # hybrid | vlm-only | nocr-only
    train_until: float = 0.01            # stop "training" when unknown glyph occurrences < this share
    min_probe_coverage: float = 0.5
    placeholder: str = "�"
    low_confidence: str = "accept"       # unresolved cues: "accept" glyphs seen once (flagged) | "placeholder"
    max_vlm_cues: int | None = None      # API requests per file; when used up, finish teacher-less
    progress: Callable[[dict], None] | None = None   # called with {"event": ..., ...} at each stage
    context: int = 12                    # previous transcribed cues passed to the VLM as reference (0 = off)
    track: int = 0
    rescale: float | None = None         # testing only: resample masks to simulate another resolution
    private_dir: Path = Path("word-memory")   # per-DB sidecar: learned image hashes (+ word memory)
    word_memory: bool = False            # remember resolved words per glyph sequence (not publishable)
    keep_special_chars: bool = False     # False: fold quote/dash/ellipsis variants (see simplify.py)
    download_dicts: bool = True          # fetch missing Hunspell dictionaries on first use
    lexicon: str = "auto"                # I/l tie-break: auto (wordfreq, then Hunspell) | wordfreq | hunspell | off


@dataclass
class CueState:
    cue: Cue
    mask: np.ndarray
    lines: list[Line]
    text: str | None = None
    source: str = ""                     # nocr | vlm | vlm-retry | fallback
    result: CueResult | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)
    requeried: bool = False
    provisional: str | None = None       # first VLM answer if it fitted (with contradictions)


def _unknown_keys(db: GlyphDB, st: CueState) -> set[str]:
    out = set()
    for l in st.lines:
        for g in l.glyphs:
            v = db.lookup(g.key, g.top_rel)
            if v is None or _decide(v)[0] in (None, ""):
                out.add(g.key)
    return out


def glyph_items(db: GlyphDB, glyphs: dict, keyfreq: Counter) -> dict[str, dict]:
    """Trusted glyph shapes of a file, for display, one entry per cluster: canonical key ->
    {label, w, h, bits, n, style}. Jittered variants of one letter (a rescaled track has
    thousands) are one entry, drawn with the file's most frequent variant and counted together.
    Shapes without a trusted label (unknown, quarantined, fragments) are left out."""
    out: dict[str, dict] = {}
    for key, g in sorted(glyphs.items(), key=lambda kv: -keyfreq[kv[0]]):
        v = db.lookup(key, g.top_rel)
        label = _decide(v)[0] if v else None
        if not label:
            continue
        canon = db.canonical(key) if key in db.shapes else key
        it = out.get(canon)
        if it is None:
            out[canon] = {"key": canon, "label": label, "w": g.w, "h": g.h,
                          "bits": base64.b64encode(np.packbits(g.bits).tobytes()).decode(),
                          "n": keyfreq[key], "style": v.style()}
        else:
            it["n"] += keyfreq[key]
    return out


def glyph_inventory(db: GlyphDB) -> dict:
    """Everything a glyph set has learned, for review: one entry per cluster with a trusted label
    {key, label, style, w, h, bits, n (occurrences over all files), members, proto}, sorted by
    frequency, plus counts of clusters still unconfirmed. Private data (word memory, sources,
    per-pixel training counts) is not part of it."""
    items: list[dict] = []
    unconfirmed = 0
    members: Counter = Counter()
    occ: Counter = Counter()
    for s in db.shapes.values():
        members[s.cluster] += 1
        occ[s.cluster] += s.n
    for key, s in db.shapes.items():
        if s.cluster != key:
            continue
        best = max(s.variants, key=lambda v: sum(v.votes.values()), default=None)
        label = _decide(best, db)[0] if best else None
        if not label:
            unconfirmed += 1
            continue
        items.append({"key": key, "label": label, "style": best.style(), "w": int(s.bits.shape[1]), "h": int(s.bits.shape[0]),
                      "bits": base64.b64encode(np.packbits(s.bits).tobytes()).decode(),
                      "n": int(occ[key]), "members": int(members[key]), "proto": key in db.protos})
    items.sort(key=lambda it: -it["n"])
    return {"name": db.name, "unit": db.unit, "charset": db.charset, "parent": db.parent,
            "items": items, "unconfirmed": unconfirmed, "shapes": len(db.shapes)}


def image_id(mask: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=8)
    h.update(np.array(mask.shape, np.int32).tobytes())
    h.update(np.packbits(mask).tobytes())
    return h.hexdigest()


def probe(db_dir: Path, keyfreq: Counter, glyphs: dict, min_cov: float,
          charset: str = CHARSET_SIMPLIFIED) -> tuple[GlyphDB, float, str]:
    """1) exact bitmap coverage of existing DBs (same font + raster);
    2) otherwise a DB of the same font at another raster size as teacher (scaled transfer);
    3) otherwise a fresh DB."""
    total = sum(keyfreq.values()) or 1
    dbs: list[GlyphDB] = []
    best, best_cov = None, 0.0
    for p in sorted(db_dir.glob("*.json")) if db_dir.is_dir() else []:
        try:
            db = GlyphDB.load(p)
        except Exception as e:  # corrupt DB should not stop the run
            log.warning("probe: cannot load %s: %s", p, e)
            continue
        if db.charset != charset:
            continue          # simplified and literal labels must never mix
        dbs.append(db)
        exact = sum(n for key, n in keyfreq.items() if key in db.shapes) / total
        cov = exact
        if exact < min_cov:
            near = sum(n for key, n in keyfreq.items()
                       if key not in db.shapes and near_match(db, glyphs[key])[1] is not None) / total
            cov = exact + near
        log.info("probe: %s covers %.1f%% of glyph occurrences (%.1f%% exact)", p.name, 100 * cov, 100 * exact)
        if cov > best_cov:
            best, best_cov = db, cov
    if best is not None and best_cov >= min_cov:
        return best, best_cov, "exact"
    name = random_db_name(db_dir)          # neutral name: DBs can be shared without revealing sources
    suffix = "" if charset == CHARSET_SIMPLIFIED else f".{charset}"
    new = GlyphDB(name, db_dir / f"{name}{suffix}.json")
    new.charset = charset
    best_tr, teacher = None, None
    for db in dbs:
        tr = transfer.find_scale(db, glyphs, keyfreq)
        if tr is not None:
            log.info("probe: %s as teacher at scale %.3f covers %.1f%% (confirmed transfer)",
                     db.path.name, tr.scale, 100 * tr.coverage)
            if best_tr is None or tr.coverage > best_tr.coverage:
                best_tr, teacher = tr, db
    if best_tr is not None and best_tr.coverage >= min_cov:
        transfer.apply(teacher, new, best_tr, glyphs)
        return new, best_tr.coverage, f"teacher {teacher.name} @ scale {best_tr.scale:.3f}"
    return new, 0.0, "new"


def _no_spaces(text: str) -> str:
    return strip_tags(text).replace(" ", "")


def _save_debug(opts: Options, st: CueState, tag: str) -> None:
    if not opts.debug_dir:
        return
    opts.debug_dir.mkdir(parents=True, exist_ok=True)
    img = np.where(st.mask, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(opts.debug_dir / f"{tag}_cue{st.cue.index:04d}.png")


async def process_file(source: Path | VobSubData, client: VLMClient | None, opts: Options) -> ProcessResult:
    """Convert one track. `source` is a path to the .idx (the .sub next to it) or the pair in memory.
    Nothing is written to disk unless opts.out_dir / opts.debug_dir are set; the glyph memory
    (and its private sidecar) is the only persistent state."""
    t0 = time.time()
    calls0 = (client.calls, client.cache_hits, client.throttled) if client else (0, 0, 0)
    if isinstance(source, VobSubData):
        idx, cues = load_vobsub_bytes(source.idx, source.sub, opts.track)
        stem, display = source.name, source.name
    else:
        idx, cues = load_vobsub(source, opts.track)
        stem, display = source.stem, str(source)
    idx_path = Path(display)     # for log lines / events only
    lang = idx.tracks[opts.track].lang or "en"
    lexicon = make_lexicon(opts.lexicon, lang, download=opts.download_dicts)
    charset = CHARSET_LITERAL if opts.keep_special_chars else CHARSET_SIMPLIFIED
    fold = (lambda t: t) if opts.keep_special_chars else simplify
    states: list[CueState] = []
    keyfreq: Counter = Counter()
    report_notes: list[str] = []
    sample_glyph: dict = {}
    all_gaps: list[int] = []
    for c in cues:
        m, bridge = fill_mask_ex(c)
        if opts.rescale:
            m = transfer.resize_bits(m, opts.rescale)     # testing: simulate another resolution
            bridge = None
        ls = segment(m, bridge)
        states.append(CueState(c, m, ls))
        for l in ls:
            keyfreq.update(g.key for g in l.glyphs)
            for g in l.glyphs:
                sample_glyph.setdefault(g.key, g)
            all_gaps += l.gaps
    gap_t = otsu_threshold(all_gaps)
    geo = italic_votes([st.lines for st in states], gap_t)
    geo_b = bold_votes([st.lines for st in states], gap_t)
    log.info("%s: %d cues, %d glyphs, %d unique shapes, gap threshold %.1f px (%.1fs)",
             idx_path.name, len(states), sum(keyfreq.values()), len(keyfreq), gap_t, time.time() - t0)

    budget = {"used": 0, "exhausted": False}

    def budget_left() -> int:
        if opts.max_vlm_cues is None:
            return 10 ** 9
        return max(0, opts.max_vlm_cues - budget["used"])

    emitted: dict[int, tuple[str, str]] = {}

    def flush_cues() -> None:
        """Send cues whose text is new or changed since the last flush (live transcript)."""
        if not opts.progress:
            return
        items = []
        for st in states:
            if st.text is None:
                continue
            key = (st.text, st.source)
            if emitted.get(st.cue.index) != key:
                emitted[st.cue.index] = key
                items.append({"i": st.cue.index, "start": st.cue.start_ms, "end": st.cue.end_ms,
                              "text": st.text, "src": st.source})
        if items:
            emit("cues", items=items, resolved=len(emitted))

    shown: dict[str, tuple[str, str]] = {}      # cluster -> (label, style) last sent
    start_shapes: set[str] = set()               # clusters the memory held before this file

    def flush_glyphs() -> None:
        """Send glyph shapes whose trusted label is new or changed (live glyph tables)."""
        if not opts.progress:
            return
        if not shown and not start_shapes:
            start_shapes.update(db.shapes)       # first flush: right after the probe, before learning
        current = glyph_items(db, sample_glyph, keyfreq)
        items = []
        for key, it in current.items():
            sig = (it["label"], it["style"])
            if shown.get(key) != sig:
                shown[key] = sig
                # "new" = a cluster this file created, not a jittered variant of a known letter
                items.append({**it, "new": key not in start_shapes})
        pending = len(keyfreq) - len(current)
        if items:
            emit("glyphs", items=items, known=len(current), pending=pending)

    def emit(event: str, **data) -> None:
        if opts.progress:
            opts.progress({"event": event, "file": idx_path.name, "cues": len(states),
                           "vlm_used": budget["used"], "vlm_budget": opts.max_vlm_cues, **data})

    def hooks(phase: str) -> dict:
        """Progress callbacks for a VLM batch: one small event per answered request, and the
        newly resolved cues / learned glyphs after every applied answer."""
        return {"on_progress": lambda k, n: emit("vlm", phase=phase, answered=k, batch=n),
                "on_apply": lambda: (flush_cues(), flush_glyphs())}

    db, cov, probe_mode = probe(opts.db_dir, keyfreq, sample_glyph, opts.min_probe_coverage, charset)
    log.info("using DB %s (%s, coverage %.1f%%)", db.path.name if db.path else db.name, probe_mode, 100 * cov)
    emit("probe", db=db.name, mode=probe_mode, coverage=round(cov, 4))
    flush_glyphs()
    db.attach_private(opts.private_dir, opts.word_memory)
    db.file_counts = dict(keyfreq)
    once_share = sum(1 for n in keyfreq.values() if n == 1) / max(1, len(keyfreq))
    if len(keyfreq) >= 200 and once_share >= RESCALED_ONCE_SHARE:
        # a track rescaled from another resolution: one-pixel edge jitter makes most letter
        # bitmaps unique (crisp tracks: ~10-20% unique, rescaled: ~70%). Reading relies on shape
        # clustering, which costs more vision requests than usual.
        log.warning("rescaled track suspected: %.0f%% of %d glyph bitmaps occur once", 100 * once_share, len(keyfreq))
        emit("notice", kind="rescaled", unique_share=round(once_share, 3), bitmaps=len(keyfreq))
        # Edge-tolerant near matching for this track's jitter, but only against a glyph set that
        # already knows the font: a new set learns its first file through exact bitmaps (the proven
        # trajectory); from the next file on, jittered variants of known letters are read instead
        # of asked.
        db.tolerant = probe_mode != "new"
        db.rescaled = True
        report_notes.append(f"rescaled track suspected: {100 * once_share:.0f}% of {len(keyfreq)} glyph bitmaps occur once")
    # Geometry votes accumulate across files: what the DB already holds plus this file's votes.
    # (Decisions compare the two counts, so re-running a file cannot flip them.) The votes are
    # only persisted together with real learning; a pure recognition run leaves the DB file untouched.
    prev_geo = {k: [list(v.geo_italic), list(v.geo_bold)] for k, sh in db.shapes.items() for v in sh.variants[:1]}
    file_geo, file_geo_b = dict(geo), dict(geo_b)      # this file's votes per glyph key
    db.file_geo = (file_geo, file_geo_b)               # persisted with the next save (merged)

    def pooled_geo(key: str) -> tuple[list[int], list[int]]:
        """Geometry votes of a glyph's whole cluster: the DB's accumulated votes plus this file's
        votes of every member (jittered variants of one letter are one letter)."""
        canon = db.canonical(key)
        pi, pb = prev_geo.get(canon, ([0, 0], [0, 0]))
        gi, gb = list(pi), list(pb)
        for k in members.get(canon, (key,)):
            fi, fb = file_geo.get(k, [0, 0]), file_geo_b.get(k, [0, 0])
            gi[0] += fi[0]; gi[1] += fi[1]; gb[0] += fb[0]; gb[1] += fb[1]
        return gi, gb

    members: dict[str, list[str]] = {}

    def apply_geo() -> None:
        """Italic/bold come from glyph geometry (word slant, stroke width); VLM tags only break ties."""
        members.clear()
        for k in set(file_geo) | set(file_geo_b):
            members.setdefault(db.canonical(k), []).append(k)
        for canon, keys in members.items():
            shape = db.shapes.get(canon)
            if shape:
                gi, gb = pooled_geo(keys[0])
                for v in shape.variants:
                    v.geo_italic, v.geo_bold = gi, gb

    def style_of(g, vlm_style: str) -> str:
        vlm = {f: [0, 1] if f in vlm_style else [1, 0] for f in "bi"}
        gi, gb = pooled_geo(g.key)
        st = combine_style(gb, gi, vlm)
        return st + "u" if g.underlined else st

    apply_geo()
    stats = Counter()
    vlm_raw: dict[int, str] = {}
    requery: list[CueState] = []
    failures: dict[int, str] = {}
    flagged: dict[int, list[str]] = {}
    total_occ = sum(keyfreq.values()) or 1

    def history(st: CueState) -> list[str]:
        """Up to opts.context preceding cues (time order) that already have text, tags stripped."""
        out: list[str] = []
        k = states.index(st) - 1
        while k >= 0 and len(out) < opts.context:
            if states[k].text:
                out.append(strip_tags(states[k].text).replace("\n", " "))
            k -= 1
        return out[::-1]

    def accepted_key(st: CueState) -> str:
        return f"accepted-{image_id(st.mask)}"

    async def vlm_cue(st: CueState, strict: bool = False) -> str:
        # the answer finally accepted for this image in an earlier run (it fit the glyphs):
        # re-runs reproduce the same text instead of re-asking a VLM that may answer differently
        acc = None if strict else client.cache_get(accepted_key(st))
        if acc is not None:
            client.cache_hits += 1
            vlm_raw[st.cue.index] = acc
            return relined(st, acc)
        png = mask_to_png(st.mask, scale=3) if strict else \
            render_cue_png(st.cue, st.mask, fill_values(st.cue), opts.render)
        text, cached = await client.transcribe_ex(png, max(1, len(st.lines)), lang, strict=strict,
                                                  context=history(st) if opts.context else None)
        if not cached:
            budget["used"] += 1
        vlm_raw[st.cue.index] = text
        return relined(st, text)

    async def vlm_sheet(group: list[CueState]) -> list[str]:
        """One request for several cues stacked into one image. Cues with an accepted answer are
        served from the cache; the sheet answer is split by empty lines, else by the glyphs; if
        neither works the cues are asked one by one."""
        results: dict[int, str] = {}
        todo: list[CueState] = []
        for st in group:
            acc = client.cache_get(accepted_key(st))
            if acc is not None:
                client.cache_hits += 1
                vlm_raw[st.cue.index] = acc
                results[st.cue.index] = relined(st, acc)
            else:
                todo.append(st)
        if len(todo) == 1:
            results[todo[0].cue.index] = await vlm_cue(todo[0])
        elif todo:
            png = sheet_png([st.mask for st in todo], scale=2)
            text, cached = await client.transcribe_ex(png, 0, lang, sheet=len(todo),
                                                      context=history(todo[0]) if opts.context else None)
            if not cached:
                budget["used"] += 1
            parts = split_sheet(fold(text), len(todo))
            if parts is None:
                parts = reline_cues(db, [st.lines for st in todo], fold(text), gap_t)
                stats["sheet_split_by_glyphs" if parts else "sheet_unsplittable"] += 1
            if parts is None:
                answers = await asyncio.gather(*(vlm_cue(st) for st in todo))
                for st, a in zip(todo, answers):
                    results[st.cue.index] = a
            else:
                for st, pt in zip(todo, parts):
                    vlm_raw[st.cue.index] = pt
                    results[st.cue.index] = relined(st, pt)
        return [results[st.cue.index] for st in group]

    def relined(st: CueState, text: str) -> str:
        """The VLM's line breaks are not trusted: the image's lines decide (align.reline)."""
        text = fold(text)
        return text if opts.mode == "vlm-only" else reline(db, st.lines, text, gap_t)


    def apply_vlm(st: CueState, text: str, source: str) -> None:
        if opts.keep_special_chars and opts.mode != "vlm-only":
            text = quotes_by_position(db, st.lines, text, gap_t)   # „ or " by where the glyph sits
        st.text = normalize_text(text)
        st.source = source
        stats[source] += 1
        if opts.mode == "vlm-only":
            return                      # reference mode: raw VLM output, no learning/arbitration
        if not st.requeried:
            # VLM text that misfits the glyphs or contradicts confirmed glyphs is wrong somewhere:
            # learn nothing from it yet, ask once more strictly (requery_misfits) and learn from that
            aligns, _ = align_cue(db, st.lines, text, gap_t)
            if aligns is None or any(a.conflicts or a.soft_conflicts for a in aligns):
                if aligns:
                    st.text, _ = restyle(st.lines, aligns, style_of, lexicon)   # provisional
                    st.provisional = text
                requery.append(st)
                return
        elif st.provisional and align_cue(db, st.lines, text, gap_t)[0] is None:
            # the strict answer does not fit at all, the first one did (up to contradictions that
            # arbitration resolves): keep the better-fitting answer
            text = st.provisional
            st.text = normalize_text(text)
        lr = learn_cue(db, st.lines, text, gap_t, source=image_id(st.mask))
        apply_geo()
        if lr.alignments and client is not None and client.cache_get(accepted_key(st)) is None:
            client.cache_put(accepted_key(st), text)
        if lr.alignments:
            # VLM characters (except where they contradict confirmed glyphs), geometric styles,
            # word breaks from the memory's gap statistics where they are confident
            gap_notes = geometry_spaces(db, st.lines, lr.alignments)
            st.text, corrections = restyle(st.lines, lr.alignments, style_of, lexicon)
            if gap_notes:
                stats["spaces_from_gap_statistics"] += 1
                flagged.setdefault(st.cue.index, []).append("; ".join(gap_notes))
            if corrections:
                stats["char_arbitrated"] += 1
                flagged.setdefault(st.cue.index, []).append("DB overrides VLM: " + ", ".join(corrections))
                log.warning("cue %d: arbitration: %s", st.cue.index, ", ".join(corrections))
        if not lr.learned and lr.reason != ALREADY_LEARNED:
            st.notes.append(f"not learned: {lr.reason}")
            log.debug("cue %d not learnable: %s", st.cue.index, lr.reason)
            stats["not_learned"] += 1
        elif not lr.learned:
            stats["already_learned"] += 1
        if lr.conflicts:
            flagged[st.cue.index] = list(lr.conflicts)
        # Arbitration: can the (updated) DB read this cue on its own, and does it agree?
        res = recognize(db, st.lines, learn_near=False, lexicon=lexicon)
        if not res.ok and res.letters_ok and res.fill_spaces(st.text):
            stats["spaces_filled_from_vlm"] += 1     # only the uncertain breaks come from the VLM
        if res.ok:
            ocr_text = res.text()
            if strip_tags(ocr_text) == strip_tags(st.text):
                if ocr_text != st.text:
                    stats["italics_from_geometry"] += 1
                st.text = ocr_text          # same characters: keep deterministic italics
            elif _no_spaces(ocr_text) == _no_spaces(st.text):
                # same letters, different word breaks: spaces come from the measured gaps and
                # the memory's gap statistics, not from the vision model ("Liebst e", "w ollt")
                stats["spaces_from_geometry"] += 1
                flagged.setdefault(st.cue.index, []).append(f"spaces from geometry: {st.text!r} -> {ocr_text!r}")
                st.text = ocr_text
            elif (not lr.alignments or any(c.startswith("misaligned") for c in corrections)
                  or any(a.conflicts >= 2 for a in lr.alignments)):
                # the VLM text does not even fit the glyphs (dropped/added letters), or only with
                # several contradictions of confirmed glyphs, while the DB reads the whole cue:
                # trust the DB. A single contradiction is left to per-character arbitration.
                note = f"VLM {st.text!r} vs nOCR {ocr_text!r}"
                flagged.setdefault(st.cue.index, []).append(note)
                st.text, st.source = ocr_text, "nocr-arbitrated"
                stats["arbitrated"] += 1
                log.warning("cue %d: VLM text does not fit the glyphs, using nOCR: %s", st.cue.index, note)

    async def requery_misfits() -> None:
        """VLM text that does not fit the glyphs (dropped/added letters, wrong line count) is
        certainly wrong somewhere: ask again with the strict prompt at 3x scale, once per cue."""
        while requery:
            todo = [st for st in requery if not st.requeried][:budget_left()]
            requery.clear()
            for st in todo:
                st.requeried = True
            if todo:
                log.info("re-asking %d cues whose VLM text misfits or contradicts the glyphs", len(todo))
                await _vlm_batch(todo, lambda s_: vlm_cue(s_, strict=True), apply_vlm, failures, "vlm-strict",
                                 **hooks("requery"))

    sheet_kw = {"group": opts.sheet, "call_group": vlm_sheet} if opts.sheet > 1 and client is not None and client.profile == "chat" else {}

    if opts.mode == "vlm-only":
        await _vlm_batch(states, vlm_cue, apply_vlm, failures, "vlm", **hooks("inference"), **sheet_kw)
    else:
        rounds = 0
        last_flush = [time.time()]

        def flush_live(n: int) -> None:
            """The matching pass over a long file is pure CPU work; the live transcript grows
            every ten cues or every second instead of once at the end of the pass."""
            if n % 10 == 0 or time.time() - last_flush[0] >= 1.0:
                last_flush[0] = time.time()
                flush_cues()

        for recheck in range(4):
            while True:
                unresolved = []
                for n, st in enumerate(states, 1):
                    if st.text is not None or st.cue.index in failures:
                        continue
                    st.result = recognize(db, st.lines, lexicon=lexicon)
                    if st.result.ok:
                        st.text = st.result.text()
                        st.source = "nocr"
                        stats["nocr"] += 1
                        flush_live(n)
                    else:
                        unresolved.append(st)
                flush_cues()
                flush_glyphs()
                if unresolved and client is not None and opts.low_confidence == "accept":
                    # An image the memory already learned from cannot teach anything a second
                    # time (its votes would not count), so asking the model again only costs a
                    # request: a re-upload reads those cues from what the first pass taught, with
                    # glyphs confirmed only once marked low-confidence as in the teacher-less path.
                    still = []
                    for st in unresolved:
                        tent = None
                        if image_id(st.mask) in db.learned_sources:
                            tent = recognize(db, st.lines, learn_near=False, lexicon=lexicon, tentative=True)
                        if tent is None or not tent.ok:
                            still.append(st)
                            continue
                        st.text = tent.text()
                        if tent.low_confidence():
                            st.source = "tentative"
                            stats["tentative"] += 1
                            flagged.setdefault(st.cue.index, []).append(
                                "low confidence: " + ", ".join(sorted(set(repr(t) for t in tent.low_confidence())))
                                + " seen once before (image learned earlier, model not asked again)")
                        else:
                            st.source = "nocr"
                            stats["nocr"] += 1
                        stats["learned_image_skipped"] += 1
                    if len(still) != len(unresolved):
                        log.info("%d cues whose image was learned before are read from memory instead of asked again",
                                 len(unresolved) - len(still))
                        unresolved = still
                        flush_cues()
                if not unresolved or opts.mode == "nocr-only":
                    break
                if budget_left() == 0:
                    if not budget["exhausted"]:
                        budget["exhausted"] = True
                        log.warning("VLM budget of %d requests used up: %d cues stay unresolved and are "
                                    "written teacher-less", opts.max_vlm_cues, len(unresolved))
                        emit("budget_exhausted", unresolved=len(unresolved))
                    break
                # unknown glyph occurrences (file-wide) decide training vs inference phase
                unknown = set()
                for st in unresolved:
                    unknown |= _unknown_keys(db, st)
                unk_share = sum(keyfreq[k] for k in unknown) / total_occ
                phase = "training" if unk_share > opts.train_until else "inference"
                batch = _select_batch(db, unresolved, keyfreq, min(opts.batch_size, budget_left()))
                rounds += 1
                emit("round", round=rounds, phase=phase, unresolved=len(unresolved), batch=len(batch))
                log.info("round %d [%s]: %d unresolved cues, unknown glyphs %.2f%% -> VLM on %d cues",
                         rounds, phase, len(unresolved), 100 * unk_share, len(batch))
                await _vlm_batch(batch, vlm_cue, apply_vlm, failures, "vlm", **hooks(phase), **sheet_kw)
                await requery_misfits()
                flush_cues()
                flush_glyphs()
                if db.dirty:
                    db.save()
            if recheck == 3 or opts.mode == "nocr-only":
                break
            # Re-read nOCR cues with the final DB: evidence learned later (e.g. a glyph that turned
            # out to be pixel-identical for I and l) may change or invalidate an earlier reading.
            reopened = changed = 0
            for st in states:
                if st.source != "nocr":
                    continue
                res = recognize(db, st.lines, learn_near=False, lexicon=lexicon)
                if not res.ok:
                    st.text, st.source = None, ""
                    stats["nocr"] -= 1
                    reopened += 1
                elif res.text() != st.text:
                    flagged.setdefault(st.cue.index, []).append(f"re-read: {st.text!r} -> {res.text()!r}")
                    st.text = res.text()
                    stats["nocr_reread"] += 1
                    changed += 1
            if reopened or changed:
                log.info("re-read with final DB: %d cues changed, %d reopened", changed, reopened)
            if not reopened:
                break

    # ---- retries ----
    if failures and client is not None and budget_left() > 0:
        retry = [st for st in states if st.cue.index in failures][:budget_left()]
        emit("retry", cues=len(retry))
        log.info("retrying %d failed cues with strict prompt / 3x scale", len(retry))
        still: dict[int, str] = {}
        await _vlm_batch(retry, lambda s: vlm_cue(s, strict=True), apply_vlm, still, "vlm-retry", **hooks("retry"))
        for st in retry:
            if st.cue.index not in still:
                failures.pop(st.cue.index, None)
        failures = {k: still[k] for k in still}
        flush_cues()

    # ---- re-arbitration: VLM cues against the final DB (glyphs confirmed later now count) ----
    for st in states:
        raw = vlm_raw.get(st.cue.index)
        if raw is None or st.source not in ("vlm", "vlm-retry", "vlm-strict"):
            continue
        aligns, _ = align_cue(db, st.lines, relined(st, raw), gap_t)
        res_final = recognize(db, st.lines, learn_near=False, lexicon=lexicon)
        if not res_final.ok and res_final.letters_ok:
            res_final.fill_spaces(st.text)
        if res_final.ok and _no_spaces(res_final.text()) == _no_spaces(st.text) and \
                strip_tags(res_final.text()) != strip_tags(st.text):
            flagged.setdefault(st.cue.index, []).append(f"spaces from geometry (final): {st.text!r} -> {res_final.text()!r}")
            st.text = res_final.text()
            stats["spaces_from_geometry_final"] += 1
            continue
        if not aligns:
            res = recognize(db, st.lines, learn_near=False, lexicon=lexicon)
            if res.ok and strip_tags(res.text()) != strip_tags(st.text):
                flagged.setdefault(st.cue.index, []).append(
                    f"final DB reading {res.text()!r} replaces VLM {st.text!r} (VLM text does not fit the glyphs)")
                log.warning("cue %d: final DB reading replaces VLM: %r -> %r", st.cue.index, st.text, res.text())
                st.text, st.source = res.text(), "nocr-arbitrated"
                stats["arbitrated_final"] += 1
            continue
        if aligns:
            gap_notes = geometry_spaces(db, st.lines, aligns)
            text, corrections = restyle(st.lines, aligns, style_of, lexicon)
            corrections = corrections + gap_notes
            if any(c.startswith("misaligned") for c in corrections) or any(a.conflicts >= 2 for a in aligns):
                res = recognize(db, st.lines, learn_near=False, lexicon=lexicon)
                if res.ok:
                    text = res.text()        # the text only fits with contradictions: the memory's reading wins
            if corrections and text != st.text:
                st.text = text
                stats["char_arbitrated_final"] += 1
                flagged.setdefault(st.cue.index, []).append("DB overrides VLM (final): " + ", ".join(corrections))
                log.warning("cue %d: arbitration (final pass): %s", st.cue.index, ", ".join(corrections))

    flush_cues()          # re-read / re-arbitration may have changed texts

    # ---- final fallback for anything unresolved ----
    for st in states:
        if st.text is not None:
            continue
        res = st.result or recognize(db, st.lines, lexicon=lexicon)
        if opts.low_confidence == "accept":
            # No model to ask (none configured, budget used up, or it failed): glyphs the model
            # read once before are taken as a low-confidence reading instead of a placeholder.
            # The cue is flagged in the report and the page; nothing is learned from it.
            tent = recognize(db, st.lines, learn_near=False, lexicon=lexicon, tentative=True)
            if tent.low_confidence():
                res = tent
                note = "low confidence: " + ", ".join(sorted(set(repr(t) for t in tent.low_confidence()))) + " seen once before"
                flagged.setdefault(st.cue.index, []).append(note)
                if tent.ok:
                    st.text = tent.text()
                    st.source = "tentative"
                    stats["tentative"] += 1
                    continue
        st.text = res.text(opts.placeholder)
        st.source = "fallback"
        stats["fallback"] += 1
        problems = res.problems()
        kinds = Counter(reason for _, _, reason in problems)
        li, ci, reason = problems[0] if problems else (0, 0, "unresolved")
        items = res.lines[li] if res.lines else None
        g0 = st.lines[li].glyphs[items.glyph_spans[ci][0]] if items and ci < len(items.glyph_spans) else None
        log.error("cue %d @%s: %s; first at line %d glyph %d (x=%s) [%s]", st.cue.index, fmt_ts(st.cue.start_ms),
                  ", ".join(f"{n} x {k}" for k, n in kinds.most_common()), li + 1, ci + 1,
                  g0.x if g0 else "?", failures.get(st.cue.index, "not sent to VLM"))
        for li, ci, reason in problems:
            log.debug("  cue %d line %d glyph %d: %s", st.cue.index, li + 1, ci + 1, reason)
        _save_debug(opts, st, "failed")

    flush_cues()
    flush_glyphs()
    # Training step: every glyph of this file whose cluster is known feeds the cluster's prototype
    # (median + stability mask), so the next file reads its jittered variants from memory.
    for key, g in sample_glyph.items():
        if key in db.shapes:
            db.observe(db.canonical(key), g.bits, keyfreq[key])
    if db.dirty or db.saved_this_run:
        db.save(final=True)          # the run's last save prunes jitter members

    srt_text = render_srt([(st.cue.start_ms, st.cue.end_ms, st.text or "") for st in states])
    report = {
        "file": display, "language": lang, "charset": charset, "db": str(db.path), "probe": probe_mode, "probe_coverage": cov,
        "cues": len(states), "by_source": dict(stats),
        "vlm_budget": {"max": opts.max_vlm_cues, "used": budget["used"], "exhausted": budget["exhausted"]},
        "notes": report_notes,
        "vlm_requests": (client.calls - calls0[0]) if client else 0,
        "vlm_throttled_429": (client.throttled - calls0[2]) if client else 0,
        "vlm_cache_hits": (client.cache_hits - calls0[1]) if client else 0,
        "failures": {str(k): v for k, v in failures.items()},
        "flagged": {str(k): v for k, v in flagged.items()},
        "not_learned": {str(st.cue.index): st.notes for st in states if st.notes},
        "lexicon": lexicon.stats if lexicon else None,
        "seconds": round(time.time() - t0, 1),
    }
    if opts.diagnostics:
        report["vlm_raw"] = {str(k): v for k, v in sorted(vlm_raw.items())}
    out = None
    if opts.out_dir is not None:
        opts.out_dir.mkdir(parents=True, exist_ok=True)
        out = opts.out_dir / (stem + ".srt")
        out.write_text(srt_text, encoding="utf-8")
        (opts.out_dir / (stem + ".report.json")).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log.info("%s -> %s | %s | %.1fs", idx_path.name, out or "(memory)", dict(stats), report["seconds"])
    emit("done", by_source=dict(stats), seconds=report["seconds"], unresolved=stats.get("fallback", 0),
         low_confidence=stats.get("tentative", 0))
    if stats.get("tentative"):
        log.warning("%s: %d cues read with low confidence (glyphs seen once before; see the report's flagged cues)",
                    idx_path.name, stats["tentative"])
    if stats.get("fallback"):
        log.warning("%s: %d of %d cues contain glyphs this glyph memory does not know (marked %s); "
                    "configure a VLM endpoint to learn them", idx_path.name, stats["fallback"], len(states),
                    opts.placeholder)
    return ProcessResult(srt_text, report)




def _select_batch(db: GlyphDB, unresolved: list[CueState], keyfreq: Counter, size: int) -> list[CueState]:
    """Greedy set cover: pick cues that teach the most frequent unknown glyphs, prefer short cues."""
    cand = [(st, _unknown_keys(db, st)) for st in unresolved]
    covered: set[str] = set()
    batch: list[CueState] = []
    remaining = list(cand)
    while remaining and len(batch) < size:
        def score(item):
            st, keys = item
            gain = sum(keyfreq[k] for k in keys - covered)
            n = sum(len(l.glyphs) for l in st.lines)
            return gain / (1.0 + 0.02 * n)
        remaining.sort(key=score, reverse=True)
        st, keys = remaining.pop(0)
        batch.append(st)
        covered |= keys
    return batch


async def _vlm_batch(batch, call, apply, failures: dict[int, str], source: str,
                     on_progress=None, on_apply=None, group: int = 1, call_group=None) -> None:
    """Ask the VLM about every cue of the batch concurrently and apply the answers in batch order.

    The order matters: applying learns into the glyph memory, and applying in a fixed order is what
    keeps re-runs deterministic. Answers are still applied as early as possible (each one as soon as
    it and all its predecessors are in), so callers can stream progress: `on_progress(k, n)` runs
    when k of n requests have been answered, `on_apply()` after each applied answer."""
    if group > 1 and call_group is not None:
        # sheets: one request per `group` consecutive cues, one answer list per request
        groups = [batch[k:k + group] for k in range(0, len(batch), group)]
        gtasks = [asyncio.ensure_future(call_group(g)) for g in groups]

        async def _nth(gt, i: int):
            return (await gt)[i]
        tasks = [asyncio.ensure_future(_nth(gt, i)) for g, gt in zip(groups, gtasks) for i in range(len(g))]
    else:
        tasks = [asyncio.ensure_future(call(st)) for st in batch]
    answered = 0

    def _done(_task) -> None:
        nonlocal answered
        answered += 1
        if on_progress:
            on_progress(answered, len(tasks))

    for t in tasks:
        t.add_done_callback(_done)
    for st, t in zip(batch, tasks):
        try:
            r = await t
        except Exception as e:  # noqa: BLE001
            failures[st.cue.index] = f"{type(e).__name__}: {e}"
            log.warning("cue %d @%s: VLM failed: %s", st.cue.index, fmt_ts(st.cue.start_ms), e)
            continue
        failures.pop(st.cue.index, None)
        apply(st, r, source)
        if on_apply:
            on_apply()
