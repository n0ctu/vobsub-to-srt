"""Probe -> train -> infer -> retry -> finalize."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .align import ALREADY_LEARNED, align_cue, learn_cue, otsu_threshold, restyle, strip_tags
from . import transfer
from .glyphdb import GlyphDB, combine_style
from .lexicon import make_lexicon
from .names import random_db_name
from .simplify import CHARSET_LITERAL, CHARSET_SIMPLIFIED, simplify
from .recognize import CueResult, near_match, recognize, _decide
from .segment import Line, bold_votes, fill_mask, italic_votes, segment
from .srt import fmt_ts, normalize_text, write_srt
from .vlm import VLMClient, mask_to_png
from .vobsub import Cue, load_vobsub

log = logging.getLogger("vobsub_to_srt")


@dataclass
class Options:
    db_dir: Path = Path("glyph-memory")
    out_dir: Path = Path("out")
    debug_dir: Path | None = Path("debug")
    batch_size: int = 16
    mode: str = "hybrid"                 # hybrid | vlm-only | nocr-only
    train_until: float = 0.01            # stop "training" when unknown glyph occurrences < this share
    min_probe_coverage: float = 0.5
    placeholder: str = "�"
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
        cov = sum(n for key, n in keyfreq.items() if key in db.shapes) / total
        log.info("probe: %s covers %.1f%% of glyph occurrences (exact)", p.name, 100 * cov)
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


def _save_debug(opts: Options, st: CueState, tag: str) -> None:
    if not opts.debug_dir:
        return
    opts.debug_dir.mkdir(parents=True, exist_ok=True)
    img = np.where(st.mask, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(opts.debug_dir / f"{tag}_cue{st.cue.index:04d}.png")


async def process_file(idx_path: Path, client: VLMClient | None, opts: Options) -> dict:
    t0 = time.time()
    calls0 = (client.calls, client.cache_hits, client.throttled) if client else (0, 0, 0)
    idx, cues = load_vobsub(idx_path, opts.track)
    lang = idx.tracks[opts.track].lang or "en"
    lexicon = make_lexicon(opts.lexicon, lang, download=opts.download_dicts)
    charset = CHARSET_LITERAL if opts.keep_special_chars else CHARSET_SIMPLIFIED
    fold = (lambda t: t) if opts.keep_special_chars else simplify
    states: list[CueState] = []
    keyfreq: Counter = Counter()
    sample_glyph: dict = {}
    all_gaps: list[int] = []
    for c in cues:
        m = fill_mask(c)
        if opts.rescale:
            m = transfer.resize_bits(m, opts.rescale)     # testing: simulate another resolution
        ls = segment(m)
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

    db, cov, probe_mode = probe(opts.db_dir, keyfreq, sample_glyph, opts.min_probe_coverage, charset)
    log.info("using DB %s (%s, coverage %.1f%%)", db.path.name if db.path else db.name, probe_mode, 100 * cov)
    db.attach_private(opts.private_dir, opts.word_memory)
    # Geometry votes accumulate across files: what the DB already holds plus this file's votes.
    # (Decisions compare the two counts, so re-running a file cannot flip them.) The votes are
    # only persisted together with real learning; a pure recognition run leaves the DB file untouched.
    prev_geo = {k: [list(v.geo_italic), list(v.geo_bold)] for k, sh in db.shapes.items() for v in sh.variants[:1]}
    for key in set(geo) | set(geo_b):
        pi, pb = prev_geo.get(key, ([0, 0], [0, 0]))
        gi, gb = geo.get(key, [0, 0]), geo_b.get(key, [0, 0])
        geo[key] = [pi[0] + gi[0], pi[1] + gi[1]]
        geo_b[key] = [pb[0] + gb[0], pb[1] + gb[1]]

    def apply_geo() -> None:
        """Italic/bold come from glyph geometry (word slant, stroke width); VLM tags only break ties."""
        for key in set(geo) | set(geo_b):
            shape = db.shapes.get(key)
            if shape:
                for v in shape.variants:
                    v.geo_italic = list(geo.get(key, [0, 0]))
                    v.geo_bold = list(geo_b.get(key, [0, 0]))

    def style_of(g, vlm_style: str) -> str:
        vlm = {f: [0, 1] if f in vlm_style else [1, 0] for f in "bi"}
        st = combine_style(geo_b.get(g.key, [0, 0]), geo.get(g.key, [0, 0]), vlm)
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

    def accepted_file(st: CueState) -> Path:
        return client.cache_dir / f"accepted-{image_id(st.mask)}.json"

    async def vlm_cue(st: CueState, strict: bool = False) -> str:
        # the answer finally accepted for this image in an earlier run (it fit the glyphs):
        # re-runs reproduce the same text instead of re-asking a VLM that may answer differently
        acc = accepted_file(st)
        if not strict and acc.exists():
            client.cache_hits += 1
            text = json.loads(acc.read_text())["text"]
            vlm_raw[st.cue.index] = text
            return fold(text)
        png = mask_to_png(st.mask, scale=3 if strict else 2)
        text = await client.transcribe(png, max(1, len(st.lines)), lang, strict=strict,
                                       context=history(st) if opts.context else None)
        vlm_raw[st.cue.index] = text
        return fold(text)

    def apply_vlm(st: CueState, text: str, source: str) -> None:
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
        if lr.alignments and client is not None:
            acc = accepted_file(st)
            if not acc.exists():
                acc.write_text(json.dumps({"text": text}, ensure_ascii=False))
        if lr.alignments:
            # VLM characters (except where they contradict confirmed glyphs), geometric styles
            st.text, corrections = restyle(st.lines, lr.alignments, style_of, lexicon)
            if corrections:
                stats["char_arbitrated"] += 1
                flagged.setdefault(st.cue.index, []).append("DB overrides VLM: " + ", ".join(corrections))
                log.warning("cue %d: confirmed glyphs override VLM: %s", st.cue.index, ", ".join(corrections))
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
        if res.ok:
            ocr_text = res.text()
            if strip_tags(ocr_text) == strip_tags(st.text):
                if ocr_text != st.text:
                    stats["italics_from_geometry"] += 1
                st.text = ocr_text          # same characters: keep deterministic italics
            elif not lr.alignments:
                # the VLM text does not even fit the glyphs (dropped/added letters) while the DB
                # reads the whole cue: trust the DB. Where the text does fit, per-character
                # arbitration (with its stricter override threshold) has already decided.
                note = f"VLM {st.text!r} vs nOCR {ocr_text!r}"
                flagged.setdefault(st.cue.index, []).append(note)
                st.text, st.source = ocr_text, "nocr-arbitrated"
                stats["arbitrated"] += 1
                log.warning("cue %d: VLM text does not fit the glyphs, using nOCR: %s", st.cue.index, note)

    async def requery_misfits() -> None:
        """VLM text that does not fit the glyphs (dropped/added letters, wrong line count) is
        certainly wrong somewhere: ask again with the strict prompt at 3x scale, once per cue."""
        while requery:
            todo = [st for st in requery if not st.requeried]
            requery.clear()
            for st in todo:
                st.requeried = True
            if todo:
                log.info("re-asking %d cues whose VLM text misfits or contradicts the glyphs", len(todo))
                await _vlm_batch(todo, lambda s_: vlm_cue(s_, strict=True), apply_vlm, failures, "vlm-strict")

    async def requery_misfits() -> None:
        """VLM text that does not fit the glyphs (dropped/added letters, wrong line count) is
        certainly wrong somewhere: ask again with the strict prompt at 3x scale, once per cue."""
        while requery:
            todo = [st for st in requery if not st.requeried]
            requery.clear()
            for st in todo:
                st.requeried = True
            if todo:
                log.info("re-asking %d cues whose VLM text misfits or contradicts the glyphs", len(todo))
                await _vlm_batch(todo, lambda s_: vlm_cue(s_, strict=True), apply_vlm, failures, "vlm-strict")

    if opts.mode == "vlm-only":
        await _vlm_batch(states, vlm_cue, apply_vlm, failures, "vlm")
    else:
        rounds = 0
        for recheck in range(4):
            while True:
                unresolved = []
                for st in states:
                    if st.text is not None or st.cue.index in failures:
                        continue
                    st.result = recognize(db, st.lines, lexicon=lexicon)
                    if st.result.ok:
                        st.text = st.result.text()
                        st.source = "nocr"
                        stats["nocr"] += 1
                    else:
                        unresolved.append(st)
                if not unresolved or opts.mode == "nocr-only":
                    break
                # unknown glyph occurrences (file-wide) decide training vs inference phase
                unknown = set()
                for st in unresolved:
                    unknown |= _unknown_keys(db, st)
                unk_share = sum(keyfreq[k] for k in unknown) / total_occ
                phase = "training" if unk_share > opts.train_until else "inference"
                batch = _select_batch(db, unresolved, keyfreq, opts.batch_size)
                rounds += 1
                log.info("round %d [%s]: %d unresolved cues, unknown glyphs %.2f%% -> VLM on %d cues",
                         rounds, phase, len(unresolved), 100 * unk_share, len(batch))
                await _vlm_batch(batch, vlm_cue, apply_vlm, failures, "vlm")
                await requery_misfits()
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
    if failures and client is not None:
        retry = [st for st in states if st.cue.index in failures]
        log.info("retrying %d failed cues with strict prompt / 3x scale", len(retry))
        still: dict[int, str] = {}
        await _vlm_batch(retry, lambda s: vlm_cue(s, strict=True), apply_vlm, still, "vlm-retry")
        for st in retry:
            if st.cue.index not in still:
                failures.pop(st.cue.index, None)
        failures = {k: still[k] for k in still}

    # ---- re-arbitration: VLM cues against the final DB (glyphs confirmed later now count) ----
    for st in states:
        raw = vlm_raw.get(st.cue.index)
        if raw is None or st.source not in ("vlm", "vlm-retry", "vlm-strict"):
            continue
        aligns, _ = align_cue(db, st.lines, fold(raw), gap_t)
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
            text, corrections = restyle(st.lines, aligns, style_of, lexicon)
            if corrections and text != st.text:
                st.text = text
                stats["char_arbitrated_final"] += 1
                flagged.setdefault(st.cue.index, []).append("DB overrides VLM (final): " + ", ".join(corrections))
                log.warning("cue %d: confirmed glyphs override VLM (final pass): %s",
                            st.cue.index, ", ".join(corrections))

    # ---- final fallback for anything unresolved ----
    for st in states:
        if st.text is not None:
            continue
        res = st.result or recognize(db, st.lines, lexicon=lexicon)
        st.text = res.text(opts.placeholder)
        st.source = "fallback"
        stats["fallback"] += 1
        for li, ci, reason in res.problems():
            items = res.lines[li]
            g0 = st.lines[li].glyphs[items.glyph_spans[ci][0]] if ci < len(items.glyph_spans) else None
            log.error("cue %d @%s line %d glyph %d (x=%s): %s [%s]", st.cue.index, fmt_ts(st.cue.start_ms),
                      li + 1, ci + 1, g0.x if g0 else "?", reason, failures.get(st.cue.index, "not sent to VLM"))
        _save_debug(opts, st, "failed")

    if db.dirty:
        db.save()

    out = opts.out_dir / (idx_path.stem + ".srt")
    write_srt(out, [(st.cue.start_ms, st.cue.end_ms, st.text or "") for st in states])
    report = {
        "file": str(idx_path), "language": lang, "charset": charset, "db": str(db.path), "probe": probe_mode, "probe_coverage": cov,
        "cues": len(states), "by_source": dict(stats),
        "vlm_requests": (client.calls - calls0[0]) if client else 0,
        "vlm_throttled_429": (client.throttled - calls0[2]) if client else 0,
        "vlm_cache_hits": (client.cache_hits - calls0[1]) if client else 0,
        "failures": {str(k): v for k, v in failures.items()},
        "flagged": {str(k): v for k, v in flagged.items()},
        "not_learned": {str(st.cue.index): st.notes for st in states if st.notes},
        "lexicon": lexicon.stats if lexicon else None,
        "vlm_raw": {str(k): v for k, v in sorted(vlm_raw.items())},
        "seconds": round(time.time() - t0, 1),
    }
    (opts.out_dir / (idx_path.stem + ".report.json")).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log.info("%s -> %s | %s | %.1fs", idx_path.name, out, dict(stats), report["seconds"])
    return report


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


async def _vlm_batch(batch, call, apply, failures: dict[int, str], source: str) -> None:
    results = await asyncio.gather(*(call(st) for st in batch), return_exceptions=True)
    for st, r in zip(batch, results):
        if isinstance(r, BaseException):
            failures[st.cue.index] = f"{type(r).__name__}: {r}"
            log.warning("cue %d @%s: VLM failed: %s", st.cue.index, fmt_ts(st.cue.start_ms), r)
        else:
            failures.pop(st.cue.index, None)
            apply(st, r, source)
