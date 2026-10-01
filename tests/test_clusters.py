"""Cluster-pooled votes: jittered variants of one letter share their evidence."""
import numpy as np

from vobsub_to_srt.glyphdb import MIN_VOTES, GlyphDB, diff_ratio, trusted_label
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


def test_edge_jitter_is_read_through_the_tolerant_stage():
    """A rescaled track moves edge pixels by one: the pixel difference of a small letter exceeds
    the tolerance, the edge-tolerant stage still reads it; a different letter is not matched."""
    import numpy as np
    from vobsub_to_srt.recognize import near_match
    db = GlyphDB("t")
    db.tolerant = True                       # as the pipeline sets it for a rescaled track
    gc = segment(render("c", size=22))[0].glyphs[0]
    ge = segment(render("e", size=22))[0].glyphs[0]
    db.add_vote(gc.key, gc.bits, gc.top_rel, "c", "", once=set())
    db.add_vote(gc.key + "2", gc.bits, gc.top_rel, "c", "", once=set())
    # jitter: grow every edge on the left half by one pixel
    j = gc.bits.copy()
    half = j.shape[1] // 2
    j[:, :half] |= np.roll(gc.bits, 1, axis=0)[:, :half] | np.roll(gc.bits, 1, axis=1)[:, :half]
    jit = type(gc)(gc.x, gc.y, j, gc.top_rel, glyph_key(j))
    assert diff_ratio(j, gc.bits) > 0.12                      # the pixel measure rejects it
    key, v, ratio = near_match(db, jit)
    assert v is not None and trusted_label(v.votes)[0] == "c" and ratio <= 0.05
    assert near_match(db, ge)[1] is None                      # 'e' is not a jittered 'c'


def test_tolerant_stage_respects_slant_and_is_off_for_crisp_tracks():
    import numpy as np
    from vobsub_to_srt.recognize import near_match
    db = GlyphDB("t")
    gc = segment(render("c", size=22))[0].glyphs[0]
    db.add_vote(gc.key, gc.bits, gc.top_rel, "c", "", once=set())
    db.add_vote(gc.key + "2", gc.bits, gc.top_rel, "c", "", once=set())
    j = gc.bits.copy(); half = j.shape[1] // 2
    j[:, :half] |= np.roll(gc.bits, 1, axis=0)[:, :half] | np.roll(gc.bits, 1, axis=1)[:, :half]
    jit = type(gc)(gc.x, gc.y, j, gc.top_rel, glyph_key(j))
    assert near_match(db, jit)[1] is None                     # crisp track: stage 2 is off
    db.tolerant = True
    assert near_match(db, jit)[1] is not None
    v = db.lookup(gc.key, gc.top_rel)
    v.geo_italic = [0, 5]                                      # the cluster is italic ...
    db.file_geo = ({jit.key: [4, 0]}, {})                      # ... this glyph's words are upright
    assert near_match(db, jit)[1] is None


def _jittered(bits, k):
    """Shift the k-th edge column/row of a bitmap by one pixel (raster jitter)."""
    import numpy as np
    j = bits.copy()
    if k % 2:
        j[:, k % j.shape[1]] |= np.roll(bits, 1, axis=0)[:, k % j.shape[1]]
    else:
        j[k % j.shape[0], :] |= np.roll(bits, 1, axis=1)[k % j.shape[0], :]
    return j


def test_prototype_median_mask_and_matching(tmp_path):
    """Samples of a letter accumulate into a median that equals the clean glyph and a mask that
    frees the flickering edge pixels; a new jitter variant matches the prototype, a different
    letter does not; the prototype survives save/load and the private corpus keeps the counts."""
    import numpy as np
    from vobsub_to_srt.glyphdb import GlyphDB, PROTO_MIN, topology
    from vobsub_to_srt.recognize import near_match
    db = GlyphDB("t", tmp_path / "t.json")
    db.attach_private(tmp_path / "wm", False)
    ge = segment(render("e", size=22))[0].glyphs[0]
    gc = segment(render("c", size=22))[0].glyphs[0]
    for g, lab in ((ge, "e"), (gc, "c")):
        db.add_vote(g.key, g.bits, g.top_rel, lab, "", once=set())
        db.add_vote(g.key + "2", g.bits, g.top_rel, lab, "", once=set())
    for k in range(PROTO_MIN + 2):
        db.observe(ge.key, _jittered(ge.bits, k))
    p = db.protos[ge.key]
    assert p.n == PROTO_MIN + 2
    assert np.array_equal(p.median[1:-1, 1:-1], ge.bits)             # jitter cancels in the median
    assert (~p.stable).sum() > 0 and p.stable[1:-1, 1:-1][ge.bits].mean() > 0.5
    new = _jittered(ge.bits, PROTO_MIN + 5)
    jit = type(ge)(ge.x, ge.y, new, ge.top_rel, glyph_key(new))
    cands = db.proto_candidates(new)
    assert cands and cands[0][1] == ge.key and cands[0][0] <= 0.04
    assert not any(k == ge.key and d <= 0.04 for d, k in db.proto_candidates(gc.bits))
    key, v, d = near_match(db, jit)
    assert key == ge.key and trusted_label(v.votes)[0] == "e"
    db.save(final=True)
    db2 = GlyphDB.load(tmp_path / "t.json")
    assert db2.protos[ge.key].n == PROTO_MIN + 2 and db2.protos[ge.key].acc is None     # public: median + mask
    db2.attach_private(tmp_path / "wm", False)
    assert db2.protos[ge.key].acc is not None and db2.proto_candidates(new)[0][1] == ge.key
