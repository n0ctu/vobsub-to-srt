"""0.3.0: settled glyphs in the main file, interim learning (single reads) in the private sidecar."""
import json
import shutil
from collections import Counter

import numpy as np

from vobsub_to_srt.glyphdb import GlyphDB, trusted_label
from vobsub_to_srt.segment import Glyph, glyph_key, segment

from test_core import render
from test_clusters import _jitter


def _main_labels(path) -> dict:
    d = json.load(open(path))
    return {(s["key"], v["top_rel"]): dict(v["votes"]) for s in d["shapes"] for v in s["variants"]}


def _all_votes(db: GlyphDB) -> dict:
    return {(k, v.top_rel): dict(+v.votes) for k, s in db.shapes.items() for v in s.variants if +v.votes}


def _glyphs(text):
    return {ch: g for ch, g in zip(text, segment(render(text))[0].glyphs)}


def test_split_keeps_settled_glyphs_in_the_main_file(tmp_path):
    g = _glyphs("Hxq")
    db = GlyphDB("s", tmp_path / "s.json")
    db.attach_private(tmp_path / "wm", False)
    H, x, q = g["H"], g["x"], g["q"]
    for _ in range(2):
        db.add_vote(H.key, H.bits, H.top_rel, "H", "")             # confirmed
    db.add_vote(x.key, x.bits, x.top_rel, "x", "")                 # read once
    db.add_vote(q.key, q.bits, q.top_rel, "I", "")                 # the I/l pair: settled
    db.add_vote(q.key, q.bits, q.top_rel, "l", "")
    pr = np.ones((7, 2), bool)
    db.add_prior("k-prior", pr, -7, "!", "teacher")                # a teacher's label: settled
    db.sequences["a|b"] = Counter({'"': 2})                         # confirmed sequence
    db.sequences["c|d"] = Counter({"%": 1})                         # seen once
    before = _all_votes(db)
    db.save(final=True)
    main = _main_labels(tmp_path / "s.json")
    assert (H.key, H.top_rel) in main and (q.key, q.top_rel) in main and ("k-prior", -7) in main
    assert (x.key, x.top_rel) not in main
    d = json.load(open(tmp_path / "s.json"))
    assert d["layout"] == 3 and set(d["sequences"]) == {"a|b"}
    side = json.load(open(tmp_path / "wm" / "s.json"))["interim"]
    assert [s["key"] for s in side["shapes"]] == [x.key] and set(side["sequences"]) == {"c|d"}
    again = GlyphDB.load(tmp_path / "s.json")
    assert (x.key, x.top_rel) not in _all_votes(again)              # the published part alone
    again.attach_private(tmp_path / "wm", False)
    assert _all_votes(again) == before and set(again.sequences) == {"a|b", "c|d"}
    again.attach_private(tmp_path / "wm", False)                    # attaching twice adds nothing
    assert _all_votes(again) == before


def test_a_glyph_read_once_is_confirmed_by_the_next_film_across_a_reseed(tmp_path):
    """Film A reads a rare glyph once; a release replaces the main file with the shipped set (the
    sidecar stays, as on the server); film B reads a jittered copy of it, which confirms it."""
    g = _glyphs("Hq")
    H, q = g["H"], g["q"]
    gm, wm = tmp_path / "gm", tmp_path / "wm"
    shipped = GlyphDB("f", tmp_path / "shipped" / "f.json")
    for _ in range(2):
        shipped.add_vote(H.key, H.bits, H.top_rel, "H", "")
    shipped.journal = None
    shipped.save(final=True)
    gm.mkdir()
    shutil.copy(tmp_path / "shipped" / "f.json", gm / "f.json")

    a = GlyphDB.load(gm / "f.json"); a.attach_private(wm, False)    # film A
    assert a.learn_source("film-a-cue-7")
    a.add_vote(q.key, q.bits, q.top_rel, "q", "")
    a.save(final=True)
    assert trusted_label(a.lookup(q.key, q.top_rel).votes)[0] is None
    assert (q.key, q.top_rel) not in _main_labels(gm / "f.json")

    shutil.copy(tmp_path / "shipped" / "f.json", gm / "f.json")     # the release reseeds the set

    b = GlyphDB.load(gm / "f.json"); b.attach_private(wm, False)    # film B
    assert b.lookup(q.key, q.top_rel).votes == Counter({"q": 1})    # the single read survived
    jq = _jitter(q.bits, 3)
    assert b.learn_source("film-b-cue-2")
    canon = b.add_vote(glyph_key(jq), jq, q.top_rel, "q", "")
    assert canon == q.key and trusted_label(b.lookup(q.key, q.top_rel).votes)[0] == "q"
    b.save(final=True)
    assert _main_labels(gm / "f.json")[(q.key, q.top_rel)] == {"q": 2}
    assert not json.load(open(wm / "f.json"))["interim"]["shapes"]


def test_a_confirmed_glyph_stays_in_the_main_file_after_a_dissenting_read(tmp_path):
    g = _glyphs("H")["H"]
    db = GlyphDB("d", tmp_path / "d.json"); db.attach_private(tmp_path / "wm", False)
    for _ in range(2):
        db.add_vote(g.key, g.bits, g.top_rel, "H", "")
    db.save(final=True)
    db = GlyphDB.load(tmp_path / "d.json"); db.attach_private(tmp_path / "wm", False)
    db.add_vote(g.key, g.bits, g.top_rel, "N", "")                 # a misread: now 2:1, open again
    assert trusted_label(db.lookup(g.key, g.top_rel).votes)[0] is None
    db.save(final=True)
    assert _main_labels(tmp_path / "d.json")[(g.key, g.top_rel)] == {"H": 2, "N": 1}


def test_without_a_sidecar_only_settled_glyphs_are_written(tmp_path):
    g = _glyphs("Hx")
    db = GlyphDB("n", tmp_path / "n.json")
    for _ in range(2):
        db.add_vote(g["H"].key, g["H"].bits, g["H"].top_rel, "H", "")
    db.add_vote(g["x"].key, g["x"].bits, g["x"].top_rel, "x", "")
    db.save(final=True)
    assert set(_main_labels(tmp_path / "n.json")) == {(g["H"].key, g["H"].top_rel)}


def test_probe_sees_a_font_known_only_from_single_reads(tmp_path):
    from vobsub_to_srt.pipeline import probe
    text = "aeoscn"
    glyphs = segment(render(text))[0].glyphs
    gm, wm = tmp_path / "gm", tmp_path / "wm"
    db = GlyphDB("once", gm / "once.json"); db.attach_private(wm, False)
    for g, ch in zip(glyphs, text):
        db.add_vote(g.key, g.bits, g.top_rel, ch, "")
    db.save(final=True)
    keyfreq = Counter({g.key: 3 for g in glyphs})
    sample = {g.key: g for g in glyphs}
    assert probe(gm, keyfreq, sample, 0.5)[2] == "new"               # the main file alone: unknown
    found, cov, mode = probe(gm, keyfreq, sample, 0.5, private_dir=wm)
    assert mode == "exact" and found.name == "once" and cov == 1.0
