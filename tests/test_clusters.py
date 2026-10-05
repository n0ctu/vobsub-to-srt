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


def test_tolerant_stage_prefers_confirmed_cluster_over_unconfirmed_twin():
    """A jittered variant of a letter that never joined its confirmed cluster (one vote of its
    own; every edge moved by a pixel puts it outside the pixel tolerance but inside the
    edge-tolerant one) must not make the edge-tolerant stage abstain when it is the nearest
    candidate; the confirmed cluster decides."""
    import numpy as np
    from collections import Counter
    from vobsub_to_srt.glyphdb import GlyphDB, Variant
    from vobsub_to_srt.recognize import near_match
    from vobsub_to_srt.segment import Glyph, glyph_key
    base = np.zeros((20, 14), bool)                 # a chunky letter with a margin column each side
    base[:, 5:9] = True; base[2:6, 1:13] = True; base[14:20, 1:6] = True
    right = base | np.roll(base, 1, axis=1)         # thickened to the right
    left = base | np.roll(base, -1, axis=1)         # thickened to the left
    def glyph(bits):
        g = Glyph(0, 0, bits); g.key = glyph_key(bits); g.top_rel = -20
        return g
    db = GlyphDB("t"); db.unit = 14.0; db.tolerant = True; db.file_geo = ({}, {})
    twin, conf, q = glyph(right), glyph(base), glyph(left)
    db.add_vote(twin.key, twin.bits, -20, "y", "")                   # first in its size bucket, one vote
    for _ in range(3):
        db.add_vote(conf.key, conf.bits, -20, "y", "")               # confirmed cluster of its own
    assert db.canonical(conf.key) != db.canonical(twin.key)
    key, v, r = near_match(db, q)
    assert v is not None and v.votes["y"] == 3                      # read via the confirmed cluster


def test_local_difference_separates_touching_letter_groups_but_not_jitter():
    import numpy as np
    from vobsub_to_srt.glyphdb import diff_ratio, local_diff_ratio, letters_in
    rng = np.random.default_rng(1)
    a = rng.random((30, 60)) < 0.5                  # a 60 px wide "three letter" glyph
    b = a.copy()
    b[:, 40:60] = rng.random((30, 20)) < 0.5        # a different third letter
    assert local_diff_ratio(a, b) > 2 * diff_ratio(a, b)
    j = a.copy()
    j[::7, :] ^= True                               # jitter spread over the whole glyph
    assert abs(local_diff_ratio(a, j) - diff_ratio(a, j)) < 0.02
    assert letters_in("fte") == 3 and letters_in("...") == 0 and letters_in(None, 'r"') == 1


def test_sequence_key_tells_high_ticks_from_low_commas():
    """Same bitmaps, different height: a " at cap height and a „ on the baseline are two sequences."""
    import numpy as np
    from vobsub_to_srt.glyphdb import GlyphDB
    from vobsub_to_srt.segment import Glyph
    db = GlyphDB("t")
    bits = np.ones((10, 6), bool)
    low = [Glyph(0, 40, bits.copy()), Glyph(8, 40, bits.copy())]
    high = [Glyph(0, 10, bits.copy()), Glyph(8, 10, bits.copy())]
    for g in low:
        g.top_rel = -5
    for g in high:
        g.top_rel = -30
    for g in low + high:
        g.key = "k" + str(g.x)
    for _ in range(3):
        db.add_sequence(low, "„")
    assert db.seq_label(low) == "„"
    assert db.seq_label(high) is None
    assert db.seq_key(high) != db.seq_key(low)


def test_jittered_sequence_parts_join_the_stored_parts_cluster():
    """A quote learned from one pair of ticks is recognised from a slightly different pair."""
    import numpy as np
    from vobsub_to_srt.glyphdb import GlyphDB
    from vobsub_to_srt.segment import Glyph
    rng = np.random.default_rng(3)
    base = rng.random((12, 7)) < 0.6
    def pair(bits_a, bits_b, name):
        gl = [Glyph(0, 10, bits_a.copy()), Glyph(9, 10, bits_b.copy())]
        for k, g in enumerate(gl):
            g.top_rel = -30
            g.key = f"{name}{k}"
        return gl
    db = GlyphDB("t")
    for _ in range(3):
        db.add_sequence(pair(base, base, "a"), '"')
    jit = base.copy(); jit[0, 3] ^= True                   # one edge pixel differs
    assert db.seq_label(pair(jit, base, "b")) == '"'


def test_jittered_sequence_parts_snap_to_the_sequence_cluster_for_the_key():
    """Ticks of ~35 px of ink scatter over several small clusters on a rescaled track; the pair's
    key still lands on the cluster that carries the sequence."""
    import numpy as np
    from vobsub_to_srt.glyphdb import GlyphDB
    from vobsub_to_srt.segment import Glyph
    rng = np.random.default_rng(5)
    base = rng.random((11, 6)) < 0.6
    def pair(a, b, name):
        gl = [Glyph(0, 10, a.copy()), Glyph(9, 10, b.copy())]
        for k, g in enumerate(gl):
            g.top_rel = -30
            g.key = f"{name}{k}"
        return gl
    db = GlyphDB("t")
    for _ in range(3):
        db.add_sequence(pair(base, base, "a"), '"')
    far = base.copy()
    far[0, :] ^= True                    # a whole row differs: beyond the cluster tolerance for 35 px, within the key tolerance
    assert db.find_cluster(far, -30) is None
    assert db.seq_label(pair(far, base, "b")) == '"'


def test_strict_letter_relaxes_once_the_lookalike_is_known():
    """A jittered l one row off is a question while the set knows no I; once an I cluster exists
    and sits well apart, the variant matches l (rival-free), and a bitmap next to the I matches I."""
    from vobsub_to_srt.recognize import near_match
    db = GlyphDB("t")
    gl = segment(render("l"))[0].glyphs[0]
    for _ in range(2):
        db.add_vote(gl.key, gl.bits, gl.top_rel, "l", "", once=set())
    taller = np.vstack([gl.bits[:1], gl.bits])                      # one row more: jitter of the l
    assert db.find_cluster(taller, gl.top_rel - 1, "l") is None     # no I known yet: strict
    short = gl.bits[2:]                                              # two rows shorter: the font's I
    for _ in range(2):
        db.add_vote(glyph_key(short), short, gl.top_rel + 2, "I", "", once=set())
    assert db.lookalike_known("l", gl.bits.shape[0])
    assert db.find_cluster(taller, gl.top_rel - 1, "l") == db.canonical(gl.key)      # rival I far enough
    near_i = short.copy(); near_i[0, 0] = not near_i[0, 0]            # one pixel off the I
    assert db.find_cluster(near_i, gl.top_rel + 2, None) == db.canonical(glyph_key(short))
    assert db.find_cluster(near_i, gl.top_rel + 2, "l") is None     # read as l by the model: kept apart


def test_stray_vote_does_not_hide_an_il_cluster():
    """The font's I/l cluster read as l 121 times, I 14 times and once as J is still the I/l pair:
    a jittered l matches it (the word decides the letter) instead of going to the model."""
    from collections import Counter
    from vobsub_to_srt.recognize import voted_labels, confusable, near_match
    assert voted_labels(Counter({"l": 121, "I": 14, "J": 1})) == {"l", "I"}
    assert confusable(voted_labels(Counter({"l": 121, "I": 14, "J": 1})))
    assert voted_labels(Counter({"l": 3, "J": 1})) == {"l", "J"}             # few reads: nothing is a stray
    db = GlyphDB("t"); db.tolerant = True
    g = segment(render("l"))[0].glyphs[0]
    db.add_vote(g.key, g.bits, g.top_rel, "l", "", weight=121)
    db.add_vote(g.key, g.bits, g.top_rel, "I", "", weight=14)
    db.add_vote(g.key, g.bits, g.top_rel, "J", "", weight=1)
    jit = g.bits.copy()
    ys, xs = np.nonzero(jit)
    jit[ys[0], xs[0]] = False                                             # one pixel of jitter
    gj = Glyph(g.x, g.y, jit, g.top_rel, glyph_key(jit))
    key, v, r = near_match(db, gj)
    assert key == g.key and v is not None
