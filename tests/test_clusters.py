"""Cluster-pooled votes: jittered variants of one letter share their evidence."""
import numpy as np

from vobsub_to_srt.glyphdb import MIN_VOTES, GlyphDB, trusted_label
from vobsub_to_srt.recognize import _decide, near_match
from vobsub_to_srt.segment import Glyph, glyph_key

from test_core import render
from vobsub_to_srt.segment import segment


def _jitter(bits: np.ndarray, seed: int) -> np.ndarray:
    """Flip a few edge pixels: the kind of difference a rescaled track produces."""
    rng = np.random.default_rng(seed)
    out = bits.copy()
    edge = np.argwhere(bits & ~np.roll(bits, 1, axis=1))
    for y, x in edge[rng.choice(len(edge), size=max(1, len(edge) // 12), replace=False)]:
        out[y, x] = False
    return out


def test_two_jittered_reads_confirm_the_cluster():
    db = GlyphDB("t")
    g = segment(render("e"))[0].glyphs[0]
    a, b = _jitter(g.bits, 1), _jitter(g.bits, 2)
    ka, kb = glyph_key(a), glyph_key(b)
    assert ka != kb
    db.add_vote(ka, a, g.top_rel, "e", "", once=set())
    assert trusted_label(db.lookup(ka, g.top_rel).votes)[0] is None            # one read: quarantined
    db.add_vote(kb, b, g.top_rel, "e", "", once=set())
    assert db.canonical(kb) == db.canonical(ka)                                 # same cluster ...
    assert trusted_label(db.lookup(ka, g.top_rel).votes)[0] == "e"             # ... two reads: trusted
    assert trusted_label(db.lookup(kb, g.top_rel).votes)[0] == "e"
    # a third variant is read from memory through a near match, without any vote
    c = _jitter(g.bits, 3)
    kc = glyph_key(c)
    gc = Glyph(g.x, g.y, c, g.top_rel, key=kc)
    key, v, ratio = near_match(db, gc)
    assert v is not None and _decide(v)[0] == "e" and ratio <= 0.12


def test_variants_in_one_cue_vote_once():
    db = GlyphDB("t")
    g = segment(render("e"))[0].glyphs[0]
    a, b = _jitter(g.bits, 1), _jitter(g.bits, 2)
    once = set()
    db.add_vote(glyph_key(a), a, g.top_rel, "e", "", once=once)
    db.add_vote(glyph_key(b), b, g.top_rel, "e", "", once=once)                 # same cue
    assert sum(db.lookup(glyph_key(a), g.top_rel).votes.values()) == 1 < MIN_VOTES


def test_conflicting_label_does_not_join():
    db = GlyphDB("t")
    g = segment(render("e"))[0].glyphs[0]
    a, b = _jitter(g.bits, 1), _jitter(g.bits, 2)
    for _ in range(2):
        db.add_vote(glyph_key(a), a, g.top_rel, "e", "", once=set())
    db.add_vote(glyph_key(b), b, g.top_rel, "c", "", once=set())               # the VLM misread it
    assert db.canonical(glyph_key(b)) == glyph_key(b)                           # kept apart
    assert trusted_label(db.lookup(glyph_key(a), g.top_rel).votes)[0] == "e"


def test_strict_glyphs_never_cluster_across_heights():
    db = GlyphDB("t")
    lines = segment(render("l"))
    g = lines[0].glyphs[0]
    for _ in range(2):
        db.add_vote(g.key, g.bits, g.top_rel, "l", "", once=set())
    shorter = g.bits[1:]                                                         # one pixel row less: an I
    assert db.find_cluster(shorter, g.top_rel + 1, "I") is None
    assert db.find_cluster(shorter, g.top_rel + 1, None) is None


def test_cluster_survives_save_and_load(tmp_path):
    db = GlyphDB("t", tmp_path / "t.json")
    g = segment(render("e"))[0].glyphs[0]
    a, b = _jitter(g.bits, 1), _jitter(g.bits, 2)
    db.add_vote(glyph_key(a), a, g.top_rel, "e", "", once=set())
    db.add_vote(glyph_key(b), b, g.top_rel, "e", "", once=set())
    db.save(final=True)
    db2 = GlyphDB.load(tmp_path / "t.json")
    # the canonical (a) is stored with both votes; the once-seen jitter member b is pruned on save
    # (rescaled tracks produce thousands) and is found again through the near search
    assert glyph_key(a) in db2.shapes and glyph_key(b) not in db2.shapes
    assert db2.find_cluster(b, g.top_rel, "e") == db2.canonical(glyph_key(a))
    assert sum(db2.lookup(glyph_key(a), g.top_rel).votes.values()) == 2
    v = db2.lookup(db2.find_cluster(b, g.top_rel, "e"), g.top_rel)       # read through the cluster
    assert trusted_label(v.votes)[0] == "e"
