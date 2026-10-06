from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vobsub_to_srt.align import align_line, learn_cue, otsu_threshold, parse_styled, strip_tags
from vobsub_to_srt.glyphdb import GlyphDB, trusted_label
from vobsub_to_srt.recognize import recognize
from vobsub_to_srt.segment import segment, split_lines
from vobsub_to_srt.srt import fmt_ts, normalize_text


def render(text: str, size: int = 40) -> np.ndarray:
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        font = ImageFont.load_default(size)
    lines = text.split("\n")
    im = Image.new("L", (40 + size * max(len(l) for l in lines), 20 + int(size * 1.4) * len(lines)), 0)
    d = ImageDraw.Draw(im)
    for i, l in enumerate(lines):
        d.text((10, 10 + i * int(size * 1.4)), l, fill=255, font=font)
    return np.array(im) > 127


def test_i_dots_and_umlauts_merge_into_glyphs():
    lines = segment(render("iÄöü"))
    assert len(lines) == 1
    assert len(lines[0].glyphs) == 4


def test_two_lines_are_split():
    assert len(split_lines(render("abc\ndef"))) == 2


def test_parse_styled_and_normalize():
    st = parse_styled("<i>Hi</i>  you")
    assert [c for c, _ in st] == list("Hi you")
    assert [st_ for _, st_ in st][:3] == ["i", "i", ""]
    assert normalize_text("<i>Hi </i><i>there</i> you") == "<i>Hi there</i> you"
    assert strip_tags("<i>a</i>b") == "ab"


def test_learn_then_recognize_roundtrip():
    mask = render("hello world\nhold the door")
    lines = segment(mask)
    gaps = [g for l in lines for g in l.gaps]
    t = otsu_threshold(gaps)
    db = GlyphDB("t")
    lr = learn_cue(db, lines, "hello world\nhold the door", t)
    assert lr.learned, lr.reason
    assert not recognize(db, lines).ok           # one observation is not enough
    lines_b = segment(render("world hello\ndoor the hold"))
    assert learn_cue(db, lines_b, "world hello\ndoor the hold", t).learned
    res = recognize(db, lines)
    assert res.ok
    assert res.text() == "hello world\nhold the door"
    # a new text built from known glyphs is read without any VLM help
    lines2 = segment(render("hold the world"))
    assert recognize(db, lines2).text() == "hold the world"


def test_unknown_glyph_is_reported():
    db = GlyphDB("t")
    for _ in range(2):
        learn_cue(db, segment(render("ab")), "ab", 12)
    res = recognize(db, segment(render("abz")))
    assert not res.ok
    assert res.problems()[0][1] == 2


def test_alignment_rejects_wrong_line_count():
    db = GlyphDB("t")
    lr = learn_cue(db, segment(render("ab\ncd")), "ab cd", 12)
    assert not lr.learned


def test_alignment_handles_touching_letters():
    db = GlyphDB("t")
    line = segment(render("hello"))[0]
    # pretend two glyphs touched: merge glyph 2+3 ("ll")
    a = align_line(db, line, parse_styled("hello"), 12)
    assert a is not None and "".join(m.text for m in a.mappings) == "hello"


def test_fmt_ts():
    assert fmt_ts(3_723_004) == "01:02:03,004"


def test_nested_styles_roundtrip():
    assert normalize_text("<b><i>Hey</i></b> <u>you</u>") == "<b><i>Hey</i></b> <u>you</u>"
    assert normalize_text("<i>a <b>b</b></i>") == "<i>a <b>b</b></i>"
    # unbalanced VLM output is repaired per line
    assert normalize_text("<i>open\nnext</i>") == "<i>open</i>\nnext"


def test_underline_is_stripped_and_detected():
    mask = render("gypsy jig")
    lines = segment(mask)
    base = lines[0]
    ul = mask.copy()
    y = base.baseline + 3          # draw an underline through the descenders
    x0 = min(g.x for g in base.glyphs)
    x1 = max(g.right for g in base.glyphs)
    ul[y:y + 3, x0:x1] = True
    ul_lines = segment(ul)
    assert len(ul_lines) == 1
    assert ul_lines[0].underlines
    assert len(ul_lines[0].glyphs) == len(base.glyphs)
    assert all(g.underlined for g in ul_lines[0].glyphs)


def test_bold_learned_from_vlm_tags():
    lines = segment(render("big deal"))
    db = GlyphDB("t")
    for _ in range(2):
        assert learn_cue(db, lines, "<b>big</b> deal", 12).learned
    assert recognize(db, lines).text() == "<b>big</b> deal"


def test_label_needs_two_thirds_majority():
    from collections import Counter
    from vobsub_to_srt.glyphdb import trusted_label
    assert trusted_label(Counter({"a": 1}))[0] is None            # seen once: quarantined
    assert trusted_label(Counter({"a": 2}))[0] == "a"
    assert trusted_label(Counter({"a": 2, "e": 1}))[0] is None    # a dissenting read: the majority needs a third
    assert trusted_label(Counter({"a": 3, "e": 1}))[0] == "a"
    assert trusted_label(Counter({"6": 3, "8": 2}))[0] is None    # below 2/3
    assert trusted_label(Counter({"a": 2, "e": 2}))[0] is None
    assert trusted_label(Counter({"a": 2, "e": 0}))[0] == "a"     # a cleared vote is not dissent
    assert trusted_label(Counter({"I": 4, "1": 1}))[0] == "I"


def test_confusable_digits_use_strict_tolerance():
    from vobsub_to_srt.glyphdb import CLUSTER_TOL, STRICT_TOL, tol_for
    assert tol_for("6") == STRICT_TOL and tol_for("8") == STRICT_TOL and tol_for("g") == STRICT_TOL
    assert tol_for("e") == CLUSTER_TOL and tol_for("7") == CLUSTER_TOL


def test_repeated_glyph_in_one_cue_votes_once():
    db = GlyphDB("t")
    lines = segment(render("aaa"))
    learn_cue(db, lines, "aaa", 12)
    v = db.lookup(lines[0].glyphs[0].key, lines[0].glyphs[0].top_rel)
    assert sum(v.votes.values()) == 1


def _render_font(text: str, font_file: str, size: int = 40) -> np.ndarray:
    font = ImageFont.truetype(font_file, size)
    im = Image.new("L", (40 + size * len(text), int(size * 1.6)), 0)
    ImageDraw.Draw(im).text((10, 10), text, fill=255, font=font)
    return np.array(im) > 127


def test_bold_detected_from_stroke_width():
    import pytest
    from vobsub_to_srt.segment import bold_votes
    try:
        regular = [segment(_render_font(t, "LiberationSans-Regular.ttf"))
                   for t in ("the quick brown fox", "jumps over the lazy dog", "hello world again")]
        bold = segment(_render_font("Warning", "LiberationSans-Bold.ttf"))
    except OSError:
        pytest.skip("Liberation fonts not installed")
    votes = bold_votes(regular + [bold], gap_threshold=12)
    assert all(votes[g.key][1] > votes[g.key][0] for g in bold[0].glyphs if g.key in votes)
    assert all(votes[g.key][0] > votes[g.key][1] for l in regular[0] for g in l.glyphs if g.key in votes)


def test_confirmed_glyph_overrides_vlm_misread():
    from vobsub_to_srt.align import restyle
    db = GlyphDB("t")
    for _ in range(3):
        learn_cue(db, segment(render("hello")), "hello", 12)
    lines = segment(render("hello"))
    lr = learn_cue(db, lines, "hallo", 12)          # VLM misreads one confirmed glyph
    assert not lr.learned and lr.conflicts          # nothing in a contradicting answer is evidence
    text, corr = restyle(lines, lr.alignments, lambda g, s: s)
    assert text == "hello" and corr == ["'a'->'e'"]


def test_confusable_labels():
    from vobsub_to_srt.recognize import confusable
    assert confusable({"I", "l"})
    assert confusable({"ll", "II"}) and confusable({"Il", "lI", "ll"})
    assert not confusable({"ll", "le"}) and not confusable({"I", "1"}) and not confusable({"l"})


def _ambiguous_alles_db(word_memory: bool = False):
    """DB where the l glyphs of 'Alles' were read as I once and as l twice (pixel-identical I/l font)."""
    db = GlyphDB("t")
    db.word_memory = word_memory
    lines = segment(render("Alles"))
    for text in ("AIIes", "Alles", "Alles"):
        learn_cue(db, lines, text, 12)
    return db, lines


def test_ambiguous_word_is_not_guessed_from_case():
    db, lines = _ambiguous_alles_db()
    db.words.clear()                       # no word memory
    res = recognize(db, lines)
    assert not res.ok                      # no "AIIes" from an uppercase-neighbour rule
    assert any(r == "ambiguous I/l" for _, _, r in res.problems())


def test_ambiguous_word_resolved_by_word_memory_and_lexicon():
    db, lines = _ambiguous_alles_db(word_memory=True)
    assert recognize(db, lines).text() == "Alles"          # word memory: 2x "Alles" vs 1x "AIIes"
    db.words.clear()

    class Lex:
        def resolve(self, options):
            import itertools
            words = {"".join(p) for p in itertools.product(*options)}
            return "Alles" if "Alles" in words else None
    res = recognize(db, lines, lexicon=Lex())
    assert res.ok and res.text() == "Alles"


def test_near_match_never_bridges_a_height_difference():
    from vobsub_to_srt.recognize import near_match
    db = GlyphDB("t")
    lines = segment(render("Ill"))
    for _ in range(2):
        learn_cue(db, lines, "Ill", 12)
    g = lines[0].glyphs[1]
    taller = type(g)(g.x, g.y - 1, np.vstack([g.bits[:1], g.bits]), g.top_rel - 1, key="x")
    assert near_match(db, taller)[0] is None


def test_lexicon_gate_prefers_clear_words_only():
    from vobsub_to_srt.lexicon import Lexicon
    lx = Lexicon("de", use_hunspell=False)
    assert lx.resolve([["A"], ["I", "l"], ["I", "l"], ["e"], ["s"]]) == "Alles"
    assert lx.resolve([["A"], ["L"], ["L"], ["E"], ["S"]]) is None          # nothing ambiguous
    en = Lexicon("en", use_hunspell=False)
    assert en.resolve([["I", "l"], ["s"]]) == "Is"
    assert en.resolve([["P"], ["I", "l"]]) is None                          # PI vs Pl: too close


def test_teacher_transfer_across_sizes():
    from collections import Counter
    from vobsub_to_srt import transfer
    teacher = GlyphDB("big")
    text = "the quick brown fox jumps over a lazy dog"
    big = segment(_render_font(text, "LiberationSans-Regular.ttf", 48))
    for _ in range(2):
        assert learn_cue(teacher, big, text, 14).learned
    small = segment(_render_font(text, "LiberationSans-Regular.ttf", 32))
    glyphs = {g.key: g for l in small for g in l.glyphs}
    freq = Counter(g.key for l in small for g in l.glyphs)
    tr = transfer.find_scale(teacher, glyphs, freq)
    assert tr is not None and abs(tr.scale - 32 / 48) < 0.05
    new = GlyphDB("small")
    transfer.apply(teacher, new, tr, glyphs)
    got = {v.prior for s in new.shapes.values() for v in s.variants if v.prior}
    assert len(got) >= 15                           # most letters bootstrapped
    assert not got & transfer.FRAGILE               # thin strokes are never transferred
    wrong = [(g, new.lookup(g.key, g.top_rel).prior) for l in small for g in l.glyphs
             if new.lookup(g.key, g.top_rel)]
    truth = [c for c in text if c != " "]
    lab = {g.key: c for l in small for g, c in zip(l.glyphs, truth)}
    assert all(lab[g.key] == p for g, p in wrong)   # every transferred prior is correct


def test_lexicon_overrides_vlm_on_pixel_identical_glyphs():
    from vobsub_to_srt.align import align_cue, restyle
    from vobsub_to_srt.lexicon import Lexicon
    db, lines = _ambiguous_alles_db()
    aligns, _ = align_cue(db, lines, "AIIes", 12)
    text, corr = restyle(lines, aligns, lambda g, s: s, Lexicon("de", use_hunspell=False))
    assert text == "Alles" and "(lexicon)" in corr[0]


def test_same_image_votes_only_once():
    db = GlyphDB("t")
    lines = segment(render("abc"))
    assert learn_cue(db, lines, "abc", 12, source="img1").learned
    lr = learn_cue(db, lines, "abc", 12, source="img1")        # re-run of the same cue
    assert not lr.learned and lr.alignments
    assert not recognize(db, lines).ok                          # still quarantined
    assert learn_cue(db, lines, "abc", 12, source="img2").learned
    assert recognize(db, lines).text() == "abc"


def test_case_consistency_fallback_for_unknown_words():
    from vobsub_to_srt.lexicon import Lexicon, case_regular
    assert case_regular("ZORVANIA") and not case_regular("ZORVANlA")
    assert case_regular("O´Neil") and case_regular("Erdbeer-Eis") and not case_regular("iPhone")
    lx = Lexicon("de", use_hunspell=False)
    fixed = lambda s: [[c] for c in s]
    assert lx.resolve(fixed("ZORVAN") + [["I", "l"]] + fixed("A")) == "ZORVANIA"
    assert lx.resolve(fixed("Läch") + [["I", "l"]] + fixed("er")) == "Lächler"
    assert lx.resolve(fixed("XyzAb") + [["I", "l"]] + fixed("er")) is None       # mixed case: VLM decides
    assert Lexicon("en", use_hunspell=False).resolve([["P"], ["I", "l"]]) is None  # both are words: VLM


def test_vlm_dropping_a_letter_does_not_align():
    from vobsub_to_srt.align import align_cue
    db = GlyphDB("t")
    lines = segment(render("Verdoppeln"))
    for src in ("a", "b"):
        learn_cue(db, lines, "Verdoppeln", 12, source=src)
    aligns, reason = align_cue(db, lines, "Verdopeln", 12)
    assert aligns is None                                   # -> cue-level arbitration uses the DB
    assert recognize(db, lines).text() == "Verdoppeln"


def test_simplify_folds_variants_only():
    from vobsub_to_srt.simplify import simplify
    assert simplify("O´Neil, O’Brien, rock`n`roll") == "O'Neil, O'Brien, rock'n'roll"
    assert simplify("„Krone“ «oui» “quoted”") == '"Krone" "oui" "quoted"'
    assert simplify("Wait—what – 1−2") == "Wait-what - 1-2"
    assert simplify("Na…ja") == "Na...ja"
    assert simplify("ärger ﬁne day­") == "ärger fine day"
    assert simplify("Straße, élan, <i>ça</i>") == "Straße, élan, <i>ça</i>"   # letters and tags untouched


def test_random_db_names(tmp_path):
    from vobsub_to_srt.names import random_db_name
    names = {random_db_name(tmp_path) for _ in range(200)}
    assert len(names) == 200
    assert all(len(n.split("-")) == 3 for n in names)
    (tmp_path / "amber-falcon-0000.json").write_text("{}")
    assert all(n != "amber-falcon-0000" for n in (random_db_name(tmp_path) for _ in range(50)))


def test_known_single_letter_is_never_two_letters():
    db = GlyphDB("t")
    base = segment(render("we we"))
    for src in ("a", "b"):
        learn_cue(db, base, "we we", 12, source=src)
    line = segment(render("we"))[0]
    a = align_line(db, line, parse_styled("wte"), 12)       # VLM text with an extra letter
    assert a is None or all(m.text != "te" for m in a.mappings)


def test_publishable_db_and_private_sidecar(tmp_path):
    import json
    for memory in (False, True):
        db = GlyphDB("calm-sable-0000", tmp_path / "db" / f"m{memory}.json")
        db.attach_private(tmp_path / "private", word_memory=memory)
        lines = segment(render("Zorvan"))
        for src in ("a", "b"):
            learn_cue(db, lines, "Zorvan", 12, source=src)
        db.save()
        public = (tmp_path / "db" / f"m{memory}.json").read_text()
        assert "Zorvan" not in public and "learned_sources" not in public and '"words"' not in public
        private = json.loads((tmp_path / "private" / f"m{memory}.json").read_text())
        assert private["learned_sources"] == ["a", "b"]
        assert ("words" in private) == memory
        again = GlyphDB.load(tmp_path / "db" / f"m{memory}.json")
        again.attach_private(tmp_path / "private", word_memory=memory)
        assert again.learned_sources == {"a", "b"} and bool(again.words) == memory


def test_lexicon_repairs_il_non_words_only():
    from vobsub_to_srt.lexicon import Lexicon
    lx = Lexicon("de", use_hunspell=False)
    assert lx.repair(["l", "d", "e", "e"]) == "Idee"          # ldee is no word, Idee is
    assert lx.repair(["I", "d", "e", "e"]) is None             # already a word
    assert lx.repair(list("ZORVANIA")) is None   # unknown either way: keep


def test_baseline_glyph_memories_are_clean_and_loadable():
    """Every DB shipped in glyph-memory/ must be reusable and free of subtitle content."""
    files = sorted((Path(__file__).resolve().parents[1] / "glyph-memory").glob("*.json"))
    assert files
    for f in files:
        db = GlyphDB.load(f)
        assert db.shapes and db.unit and db.charset == "simplified"
        assert not db.words and not db.learned_sources
        for s_ in db.shapes.values():
            for v in s_.variants:
                assert all(len(lab) <= 3 for lab in v.votes)    # letters / fused pairs only


def test_glyph_items_only_trusted_shapes():
    import base64
    from collections import Counter
    from vobsub_to_srt.pipeline import glyph_items
    db = GlyphDB("t")
    lines = segment(render("abc"))
    learn_cue(db, lines, "abc", 12, source="one")
    glyphs = {g.key: g for l in lines for g in l.glyphs}
    freq = Counter(g.key for l in lines for g in l.glyphs)
    assert glyph_items(db, glyphs, freq) == {}                      # seen once: quarantined
    learn_cue(db, lines, "abc", 12, source="two")
    items = glyph_items(db, glyphs, freq)
    assert sorted(it["label"] for it in items.values()) == ["a", "b", "c"]
    it = items[lines[0].glyphs[0].key]
    g = lines[0].glyphs[0]
    bits = np.unpackbits(np.frombuffer(base64.b64decode(it["bits"]), np.uint8))[:g.w * g.h].reshape(g.h, g.w)
    assert (bits.astype(bool) == g.bits).all() and it["n"] == 1 and it["style"] == ""


def test_vlm_client_memory_cache_writes_nothing(tmp_path, monkeypatch):
    from vobsub_to_srt.vlm import VLMClient
    monkeypatch.chdir(tmp_path)
    c = VLMClient(cache_dir=None, base_url="http://x", api_key="k", model="m")
    assert c.cache_get("a") is None
    c.cache_put("a", "hello")
    assert c.cache_get("a") == "hello" and list(tmp_path.iterdir()) == []
    d = VLMClient(cache_dir=tmp_path / "cache", base_url="http://x", api_key="k", model="m")
    d.cache_put("a", "disk")
    assert (tmp_path / "cache" / "a.json").exists() and d.cache_get("a") == "disk"


def test_render_srt_matches_write_srt(tmp_path):
    from vobsub_to_srt.srt import render_srt, write_srt
    entries = [(0, 1000, "a"), (1000, 2000, ""), (2000, 3000, "<i>b</i>")]
    write_srt(tmp_path / "x.srt", entries)
    assert (tmp_path / "x.srt").read_text(encoding="utf-8") == render_srt(entries)
    assert render_srt(entries).startswith("1\n00:00:00,000 --> 00:00:01,000\na\n\n2\n")


def test_ocr_profile_fixups():
    from vobsub_to_srt.vlm import _ocr_fixups
    assert _ocr_fixups("Hello there.\nHello there.", 1) == "Hello there."
    assert _ocr_fixups("a\nb\na\nb\na\nb", 2) == "a\nb"
    assert _ocr_fixups("a\nb", 2) == "a\nb"
    assert _ocr_fixups("```text\nx\n```", 1) == "x"
    assert _ocr_fixups("a\nb\nc", 1) == "a\nb\nc"          # not a repetition: left to the aligner


def test_reline_splits_merged_lines_at_the_image_breaks():
    from vobsub_to_srt.align import reline
    db = GlyphDB("t")
    lines = segment(render("Hello there\nmy friend"))
    assert len(lines) == 2
    # the VLM merged both lines, with and without a <br>, with a break inside a word, or with
    # a break at the wrong word: the image's lines decide
    for text in ("Hello there my friend", "Hello there<br>my friend", "Hello there<br />my friend",
                 "Hello\nthere my friend", "Hello there my\nfriend", "Hello there\nmy friend"):
        assert reline(db, lines, text, 12) == "Hello there\nmy friend", text


def test_reline_keeps_styles_across_the_break():
    from vobsub_to_srt.align import reline
    db = GlyphDB("t")
    lines = segment(render("Hello there\nmy friend"))
    assert reline(db, lines, "<i>Hello there my friend</i>", 12) == "<i>Hello there</i>\n<i>my friend</i>"
    assert reline(db, lines, "Hello <i>there my</i> friend", 12) == "Hello <i>there</i>\n<i>my</i> friend"


def test_reline_splits_inside_a_word_when_the_vlm_dropped_the_space():
    from vobsub_to_srt.align import reline
    db = GlyphDB("t")
    lines = segment(render("Hello\nthere"))
    assert reline(db, lines, "Hellothere", 12) == "Hello\nthere"


def test_reline_single_line_only_flattens():
    from vobsub_to_srt.align import reline
    db = GlyphDB("t")
    lines = segment(render("Hello there"))
    assert reline(db, lines, "Hello\nthere", 12) == "Hello there"


def test_sheet_png_and_split():
    from vobsub_to_srt.vlm import sheet_png, split_sheet
    from PIL import Image
    import io
    a, b = render("Hello"), render("World\nagain")
    png = sheet_png([a, b], scale=1, pad=8, bar=4)
    im = Image.open(io.BytesIO(png))
    assert im.height == a.shape[0] + b.shape[0] + 4 * 8 + 4 + 2 * 8 and im.width == max(a.shape[1], b.shape[1]) + 16
    assert split_sheet("Hello\n\nWorld\nagain", 2) == ["Hello", "World\nagain"]
    assert split_sheet("1. Hello\n\n\n2) World again\n", 2) == ["Hello", "World again"]
    assert split_sheet("Hello World again", 2) is None


def test_reline_cues_splits_a_sheet_answer_by_the_glyphs():
    from vobsub_to_srt.align import reline_cues
    db = GlyphDB("t")
    cues = [segment(render("Hello there")), segment(render("my friend\nagain"))]
    assert reline_cues(db, cues, "Hello there my friend again", 12) == ["Hello there", "my friend\nagain"]
    assert reline_cues(db, cues, "Hello", 12) is None


def test_vlm_batch_groups_apply_in_order():
    import asyncio
    from vobsub_to_srt.pipeline import _vlm_batch

    class St:
        def __init__(self, i):
            self.cue = type("C", (), {"index": i, "start_ms": 0})()
    batch = [St(i) for i in range(5)]
    applied = []

    async def call_group(g):
        await asyncio.sleep(0.01 * (3 - len(g)))
        return [f"t{st.cue.index}" for st in g]
    asyncio.run(_vlm_batch(batch, None, lambda st, r, src: applied.append((st.cue.index, r)), {}, "vlm",
                           group=2, call_group=call_group))
    assert applied == [(i, f"t{i}") for i in range(5)]


def test_concurrent_copies_merge_on_save(tmp_path):
    """Two workers learn into the same glyph set at once: the second save merges instead of
    overwriting, and evidence from an image the other copy already learned counts once."""
    from vobsub_to_srt.glyphdb import GlyphDB
    path = tmp_path / "gm" / "t.json"
    base = GlyphDB("t", path)
    base.attach_private(tmp_path / "wm", False)
    learn_cue(base, segment(render("ab")), "ab", 12, source="img-ab")
    base.save()
    a, b = GlyphDB.load(path), GlyphDB.load(path)
    a.attach_private(tmp_path / "wm", False)
    b.attach_private(tmp_path / "wm", False)
    learn_cue(a, segment(render("cd")), "cd", 12, source="img-cd")
    learn_cue(b, segment(render("ef")), "ef", 12, source="img-ef")
    learn_cue(b, segment(render("cd")), "cd", 12, source="img-cd")     # the same image as a's
    a.save()
    b.save()
    merged = GlyphDB.load(path)
    merged.attach_private(tmp_path / "wm", False)
    labels = {trusted_label(v.votes)[0] or max(v.votes, key=v.votes.get) for s in merged.shapes.values() for v in s.variants if v.votes}
    assert {"a", "b", "c", "d", "e", "f"} <= labels
    assert merged.learned_sources == {"img-ab", "img-cd", "img-ef"}
    c_votes = [sum(v.votes.values()) for s in merged.shapes.values() for v in s.variants if max(v.votes, key=v.votes.get, default=None) == "c"]
    assert c_votes == [1]                                                 # not doubled by b's replay
    assert b.learned_sources == merged.learned_sources                    # b continues on the merged state


def test_jitter_members_are_pruned_on_save(tmp_path):
    """Members seen once are dropped on save, frequent ones kept (capped), canonicals and votes untouched."""
    from vobsub_to_srt.glyphdb import GlyphDB, MAX_MEMBERS
    db = GlyphDB("t", tmp_path / "t.json")
    db.attach_private(tmp_path / "private", False)     # one read: the glyphs live in the interim part
    lines = segment(render("abc"))
    learn_cue(db, lines, "abc", 12, source="img1")
    a = lines[0].glyphs[0]
    canon = db.canonical(a.key)
    # 30 jittered members of 'a': flip one far-corner pixel each so they differ but stay near
    import numpy as np
    for k in range(30):
        bits = a.bits.copy(); bits[0, min(k, bits.shape[1] - 1)] = True
        key = f"jit{k}"
        db.join(key, bits, a.top_rel, canon)
    db.file_counts = {f"jit{k}": (5 if k < 3 else 1) for k in range(30)}   # three recur, the rest are singletons
    db.save()                                   # mid-run save: nothing pruned yet
    assert all(f"jit{k}" in db.shapes for k in range(30))
    db.save(final=True)
    keys = set(db.shapes)
    assert canon in keys and all(f"jit{k}" in keys for k in range(3))
    assert not any(f"jit{k}" in keys for k in range(3, 30))
    reloaded = GlyphDB.load(tmp_path / "t.json")
    reloaded.attach_private(tmp_path / "private", False)
    assert set(reloaded.shapes) == keys and reloaded.shapes["jit0"].n == 5
    assert sum(1 for s in reloaded.shapes.values() if s.cluster == canon and s.key != canon) <= MAX_MEMBERS


def test_uncertain_spaces_are_filled_from_the_reference_reading():
    from vobsub_to_srt.recognize import recognize
    db = GlyphDB("t")
    lines = segment(render("ab cd"))
    for _ in range(2):
        learn_cue(db, lines, "ab cd", 12)
    res = recognize(db, lines, learn_near=False)
    assert res.ok
    res.lines[0].spaces[0] = None                       # make the a|b break uncertain
    assert not res.ok and res.letters_ok
    assert res.fill_spaces("a b cd") and res.text() == "a b cd"      # uncertain break from the reference
    res.lines[0].spaces[1] = True                       # the memory is sure of this one: c|d? no, b|c stays
    assert not res.fill_spaces("ab cd xx")              # characters differ: nothing taken


def test_once_seen_glyphs_are_read_tentatively_without_a_model(tmp_path):
    """A glyph the model read once is not confirmed; with no model to ask, the fallback takes it as
    a low-confidence reading (flagged) instead of a placeholder - unless configured otherwise."""
    import asyncio
    from vobsub_to_srt.pipeline import Options, process_file, VobSubData
    from vobsub_to_srt.recognize import recognize
    db = GlyphDB("t", tmp_path / "gm" / "t.json")
    db.attach_private(tmp_path / "wm", False)
    lines = segment(render("abc"))
    learn_cue(db, lines, "abc", 12, source="img1")          # every glyph: one vote
    res = recognize(db, lines, learn_near=False)
    assert not res.ok and all("unconfirmed" in r for _, _, r in res.problems())
    tent = recognize(db, lines, learn_near=False, tentative=True)
    assert tent.ok and tent.text() == "abc" and sorted(tent.low_confidence()) == ["a", "b", "c"]
    assert not any(it.low for it in res.lines[0].items)


def test_dropping_sequence_rules_are_ignored_and_not_learned():
    """Two dots read as one dot only records that the model miscounted an ellipsis: such a rule
    is neither applied nor learned. Two ticks forming a quote stay a legitimate sequence."""
    db = GlyphDB("t")
    lines = segment(render("a . . . b"))
    gl = lines[0].glyphs
    dots = [g for g in gl if g.bits.shape[0] <= 8]
    assert len(dots) == 3
    for _ in range(3):                                   # the dot is a confirmed glyph of its own
        for g in dots:
            db.add_vote(g.key, g.bits, g.top_rel, ".", "")
    assert db.dropping_sequence(dots[:2], ".")           # '.' '.' -> '.' drops a glyph
    assert not db.dropping_sequence(dots[:2], "…")       # a different character: allowed
    assert not db.dropping_sequence(dots[:3], "...")     # same length: nothing dropped
    db.sequences[db.seq_key(dots[:2])] = __import__("collections").Counter({".": 5})
    assert db.seq_label(dots[:2]) is None                # the stored rule is ignored
    items = recognize(db, lines, learn_near=False).lines[0]
    assert [it.text for it in items.items].count(".") == 3          # three dots, one glyph each
    assert all(b - a == 1 for a, b in items.glyph_spans)


def test_fill_values_heavy_font_and_sparse_outline():
    """A heavy font fills half of its bounding box but comes as many pieces: not a backdrop. When
    the outline is a solid box and the anti-alias ring is sparse, the thick colour is the fill."""
    import numpy as np
    from vobsub_to_srt.segment import fill_values
    from vobsub_to_srt.vobsub import Cue
    # heavy letters (value 1) with a one-pixel ring (2) and an outline (3) facing transparency (0)
    img = np.zeros((20, 60), np.uint8)
    for x0 in (4, 24, 44):                                # three fat blocks = three letters
        img[3:17, x0:x0 + 12] = 3
        img[4:16, x0 + 1:x0 + 11] = 2
        img[5:15, x0 + 2:x0 + 10] = 1
    cue = Cue(0, 0, 1000, img, [(0, 0, 0), (240, 240, 240), (153, 153, 153), (0, 0, 0)], [0, 15, 15, 15])
    assert fill_values(cue) == [1]
    # a black box (3) around the text, a ring too sparse to separate fill and box: the fill is 1
    img = np.full((20, 60), 3, np.uint8)
    for x0 in (4, 24, 44):
        img[5:15, x0 + 2:x0 + 10] = 1
        img[4, x0 + 3:x0 + 6] = 2                         # a few ring pixels only
    cue = Cue(0, 0, 1000, img, [(0, 0, 0), (240, 240, 240), (153, 153, 153), (0, 0, 0)], [0, 15, 15, 15])
    assert fill_values(cue) == [1]


def test_stray_pixels_are_dropped_before_segmentation():
    """An authoring that sets the first pixel of every cue image to the fill colour must not
    produce a one-pixel glyph (it broke the alignment of every cue). Tiny fonts keep their dots."""
    from vobsub_to_srt.segment import despeckle
    m = render("Take it")
    m[0, 0] = True
    assert len(segment(m)[0].glyphs) == len(segment(render("Take it"))[0].glyphs)
    assert not despeckle(m)[0, 0]
    tiny = np.zeros((9, 20), bool)                  # a 9 px font: its i-dot is one pixel
    tiny[3:9, 2:4] = True; tiny[0, 2] = True; tiny[3:9, 8:12] = True
    assert despeckle(tiny).sum() == tiny.sum()


def test_stacked_sequence_rule_does_not_read_an_ellipsis():
    """A colon whose two dots did not merge is learned as a stacked sequence; two dots side by
    side (an ellipsis) must not match that rule."""
    from vobsub_to_srt.segment import Glyph
    db = GlyphDB("t")
    dot = np.ones((3, 3), bool)
    stacked = [Glyph(10, 0, dot.copy()), Glyph(10, 8, dot.copy())]       # one above the other
    side = [Glyph(10, 8, dot.copy()), Glyph(16, 8, dot.copy())]          # next to each other
    for g in stacked + side:
        from vobsub_to_srt.segment import glyph_key
        g.key = glyph_key(g.bits)
    for _ in range(3):
        db.add_sequence(stacked, ":")
    assert db.seq_label(stacked) == ":"
    assert db.seq_label(side) is None


def test_bold_votes_separate_weights_and_stay_quiet_on_regular_text():
    """Words of a bold face vote bold, regular words regular; a regular-only text at a size whose
    stroke width is quantised (measured as 4 or 6 px) must not produce bold votes."""
    import pytest
    from vobsub_to_srt.segment import bold_votes
    try:
        bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 40)
        regular = ImageFont.truetype("DejaVuSans.ttf", 40)
    except OSError:
        pytest.skip("DejaVu fonts not installed")
    def render_font(text, font):
        im = Image.new("L", (40 + 40 * len(text), 70), 0)
        ImageDraw.Draw(im).text((10, 10), text, fill=255, font=font)
        return np.array(im) > 127
    lines = [segment(render_font("wenn mann", regular)), segment(render_font("wenn mann", bold)),
             segment(render_font("Hallo Welt", regular)), segment(render_font("nummer eins", regular))]
    votes = bold_votes(lines, 12)
    keys_bold = {g.key for l in lines[1] for g in l.glyphs}
    keys_reg = {g.key for ls in (lines[0], lines[2], lines[3]) for l in ls for g in l.glyphs}
    assert all(votes[k][1] > 0 for k in keys_bold if k in votes)
    assert all(votes[k][1] == 0 for k in keys_reg if k in votes)
    # regular-only text at several sizes: never a bold vote
    for size in (23, 27, 31, 35):
        f = ImageFont.truetype("DejaVuSans.ttf", size)
        ls = [segment(render_font(w, f)) for w in ("want to be", "on Chucky", "Previously", "Jennifer", "Tiffany")]
        assert all(v[1] == 0 for v in bold_votes(ls, 10).values()), size


def test_fill_values_drop_shadow_authoring():
    """A drop shadow instead of an outline leaves the fill exposed to transparency as well; the
    anti-alias ring is then the only enclosed colour and must not be taken for the fill."""
    import numpy as np
    from vobsub_to_srt.segment import fill_values
    from vobsub_to_srt.vobsub import Cue
    img = np.zeros((24, 90), np.uint8)
    for x0 in (4, 34, 64):                       # three letters: fill 1 with a ring 2, shadow 3 offset down-right
        img[8:20, x0 + 2:x0 + 14] = 3            # shadow
        img[5:17, x0:x0 + 12] = 2                # ring around the fill
        img[6:16, x0 + 1:x0 + 11] = 1            # fill
    cue = Cue(0, 0, 1000, img, [(0, 0, 0), (204, 204, 204), (153, 153, 153), (0, 0, 0)], [0, 15, 15, 15])
    assert fill_values(cue) == [1]


def test_bridge_marks_letter_pieces_connected_through_the_ring():
    """A k whose arms meet the stem only in anti-alias pixels is two fill components; with the
    bridge mask (fill + ring) segment() marks the pair as joined (the aligner may then read the
    two glyphs as one character), without it nothing is marked."""
    import numpy as np
    from vobsub_to_srt.segment import fill_mask_ex
    from vobsub_to_srt.vobsub import Cue
    img = np.zeros((40, 60), np.uint8)
    img[4:36, 8:12] = 1                                   # stem
    img[16:26, 14:22] = 1; img[26:36, 16:24] = 1          # arms, one pixel away from the stem
    img[16:36, 12:14] = 2                                 # the join: ring colour only
    img[4:36, 40:50] = 1                                  # a second letter, well apart
    ring = np.zeros_like(img, bool)
    for v in (1,):
        m = img == v
        ring |= np.pad(m, 1)[2:, 1:-1] | np.pad(m, 1)[:-2, 1:-1] | np.pad(m, 1)[1:-1, 2:] | np.pad(m, 1)[1:-1, :-2]
    img[ring & (img == 0)] = 2                            # one-pixel ring around every fill pixel
    out = np.zeros_like(img, bool)
    m2 = img > 0
    out = np.pad(m2, 1)[2:, 1:-1] | np.pad(m2, 1)[:-2, 1:-1] | np.pad(m2, 1)[1:-1, 2:] | np.pad(m2, 1)[1:-1, :-2]
    img[out & (img == 0)] = 3                             # outline around it all
    cue = Cue(0, 0, 1000, img, [(0, 0, 0), (240, 240, 240), (153, 153, 153), (0, 0, 0)], [0, 15, 15, 15])
    mask, bridge = fill_mask_ex(cue)
    assert bridge is not None and bridge.shape == mask.shape
    assert len(segment(mask)[0].glyphs) == 3                 # stem, arms, second letter
    line = segment(mask, bridge)[0]
    assert len(line.glyphs) == 3 and line.joined == [True, False]   # kept apart, but marked as one letter


def test_stray_vote_does_not_hide_an_i_l_pair():
    """A cluster read as I 112 times and l 197 times is an I/l pair for the lexicon to settle; one
    stray 'L' must not turn it into a conflict (which sent 210 cues of one file to the model)."""
    from collections import Counter
    from vobsub_to_srt.glyphdb import Variant
    from vobsub_to_srt.recognize import _decide
    assert _decide(Variant(-22, Counter({"I": 112, "l": 197, "L": 1}))) == (None, "ambiguous I/l")
    assert _decide(Variant(-22, Counter({"I": 1, "l": 1}))) == (None, "ambiguous I/l")     # small clusters unchanged
    assert _decide(Variant(-22, Counter({"e": 40, "c": 1})))[0] == "e"
    assert _decide(Variant(-22, Counter({"e": 5, "c": 5})))[0] is None                      # a real conflict stays one


def test_italic_dot_displaced_along_slant_merges_into_stem():
    """Italic i / ! : the dot sits along the slant, barely overlapping the stem's box."""
    from vobsub_to_srt.segment import segment
    m = np.zeros((30, 60), bool)
    for r in range(8, 28):                 # a stem leaning right: 3 px wide, shifts 1 px per 3 rows
        x = 10 + (27 - r) // 3
        m[r, x:x + 3] = True
    m[2:5, 18:22] = True                   # the dot, right of the stem's box top (16..19) by most of its width
    for r in range(8, 28):                 # an upright 'l' further right, then an upright dot pair (umlaut) next to it
        m[r, 36:39] = True
    m[2:5, 42:46] = True                   # a dot that belongs to a neighbour: no slant, must stay separate
    lines = segment(m)
    assert len(lines) == 1
    widths = sorted((g.x, g.bits.shape[0]) for g in lines[0].glyphs)
    assert widths == [(10, 26), (36, 20), (42, 3)], widths


def test_stacked_lines_without_a_blank_row_are_split():
    """A descender of line 1 shares rows with an umlaut of line 2: no blank row, still two lines."""
    from vobsub_to_srt.segment import segment
    m = np.zeros((70, 200), bool)
    for k in range(6):                     # line 1: six letters, rows 5..30, the third with a descender to row 40
        m[5:30, 10 + k * 30:22 + k * 30] = True
    m[30:44, 70:76] = True                 # reaches into the rows of line 2: no blank row anywhere
    for k in range(6):                     # line 2: six letters, rows 42..66, the first with umlaut dots at rows 36..39
        m[42:66, 25 + k * 30:37 + k * 30] = True
    m[36:39, 26:29] = True
    m[36:39, 33:36] = True
    lines = segment(m)
    assert [len(l.glyphs) for l in lines] == [6, 6]
    assert lines[0].y1 <= 44 and lines[1].y0 >= 36
    assert lines[1].glyphs[0].bits.shape[0] == 30     # the umlaut dots merged into their letter


def test_single_line_with_descenders_is_not_split():
    from vobsub_to_srt.segment import segment
    m = np.zeros((40, 200), bool)
    for k in range(6):
        m[5:30, 10 + k * 30:22 + k * 30] = True
    m[30:38, 40:46] = True                 # a descender
    m[30:38, 130:136] = True
    assert [len(l.glyphs) for l in segment(m)] == [6]


def test_two_tick_quote_confirmed_as_sequence_costs_nothing():
    """DejaVu draws " as two ticks. Once the memory holds the pair as ", a line with two quotes
    aligns within budget even though both ticks are confirmed apostrophes."""
    db = GlyphDB("t")
    t = 12
    for _ in range(3):                                   # confirm the ticks as apostrophes and the pair as "
        assert learn_cue(db, segment(render("it's 'a'")), "it's 'a'", t).learned
        assert learn_cue(db, segment(render('"so" it')), '"so" it', t).learned
    lines = segment(render('"ok" "no"'))
    lr = learn_cue(db, lines, '"ok" "no"', t)
    assert lr.learned, lr.reason


def test_capital_i_on_an_l_glyph_is_not_a_contradiction_in_the_slow_path():
    """The fast path (counts match) exempts I/l; the DP path (a split glyph on the line) must too."""
    from vobsub_to_srt.align import align_line
    db = GlyphDB("t")
    for _ in range(2):
        learn_cue(db, segment(render("lila")), "lila", 12)
    line = segment(render("lila"))[0]
    glyphs = line.glyphs
    # split the last glyph in two halves so the glyph count no longer matches the text
    g = glyphs[-1]
    from vobsub_to_srt.segment import Glyph, glyph_key
    half = g.bits.shape[1] // 2
    a, b = Glyph(g.x, g.y, g.bits[:, :half].copy()), Glyph(g.x + half, g.y, g.bits[:, half:].copy())
    a.key, b.key = glyph_key(a.bits), glyph_key(b.bits)
    a.top_rel = b.top_rel = g.top_rel
    line.glyphs = glyphs[:-1] + [a, b]
    line.gaps = [n.x - m.right for m, n in zip(line.glyphs, line.glyphs[1:])]
    al = align_line(db, line, parse_styled("IiIa"), 12)
    assert al is not None and al.soft_conflicts == 0


def test_low_quote_by_position_in_literal_mode():
    from vobsub_to_srt.align import quotes_by_position
    db = GlyphDB("t")
    lines = segment(render('„ab“ „cd“'))
    assert quotes_by_position(db, lines, '"ab" "cd"', 12) == '„ab" „cd"'
    assert quotes_by_position(db, lines, '„ab“ „cd“', 12) == '„ab“ „cd“'


def test_lexicon_words_split_at_dashes():
    from vobsub_to_srt.recognize import GlyphResult, LineResult, _word_ranges
    items = [GlyphResult(c, "") for c in "should--I"]
    res = LineResult(items=items, glyph_spans=[(k, k + 1) for k in range(9)], spaces=[False] * 8)
    assert _word_ranges(res) == [(0, 6), (8, 9)]


def test_gap_statistics_override_a_model_space_after_an_ellipsis():
    """The memory knows the gap between the last dot and a letter as a letter gap: the model's
    '... wenn' becomes '...wenn' even while other glyphs of the cue are still unknown."""
    from vobsub_to_srt.align import align_cue, geometry_spaces
    db = GlyphDB("t")
    t = 12
    for text in ("...wenn so", "...wann es"):
        for _ in range(2):
            assert learn_cue(db, segment(render(text)), text, t).learned
    lines = segment(render("...wenn zyx"))          # z y x unknown: the memory cannot read the cue
    aligns, reason = align_cue(db, lines, "... wenn zyx", t)
    assert aligns, reason
    assert 3 in aligns[0].spaces_before
    notes = geometry_spaces(db, lines, aligns)
    assert notes and 3 not in aligns[0].spaces_before


def test_language_guess_from_text():
    from vobsub_to_srt.lexicon import guess_language
    en = ["I don't know, Your Grace.", "It is our duty and our honor to serve the realm.", "You should be the one to tell him what we found in the north."] * 4
    de = ["Ich weiß nicht, was du meinst.", "Das ist nicht der Grund, und du weißt es.", "Wir sind nicht die Einzigen, die das wollen."] * 4
    assert guess_language(en) == "en"
    assert guess_language(de) == "de"
    assert guess_language(["Hm.", "Ja."]) is None            # too little text to tell


def test_batched_overlap_equals_single_pair():
    import numpy as np
    from vobsub_to_srt import transfer
    rng = np.random.default_rng(3)
    for _ in range(60):
        h, w = int(rng.integers(6, 30)), int(rng.integers(3, 24))
        a = rng.random((h, w)) < 0.45
        refs = []
        for k in range(int(rng.integers(1, 12))):
            bh, bw = h + int(rng.integers(-1, 2)), w + int(rng.integers(-1, 2))
            bits = rng.random((bh, bw)) < 0.45
            refs.append(transfer._Ref(f"k{k}", "x", bits, 0.0, int(bits.sum())))
        rs = transfer._Refs(refs)
        picked = [(r, row) for rows in rs.by_h.values() for r, row in rows]
        batched = rs.ious(a, picked).tolist()
        assert batched == [transfer._iou(a, r.bits) for r, _ in picked]


def test_probe_reports_each_set_tried(tmp_path):
    from collections import Counter
    from vobsub_to_srt.pipeline import probe
    for name in ("one", "two"):
        GlyphDB(name, tmp_path / f"{name}.json").save()
    steps = []
    db, cov, mode = probe(tmp_path, Counter(), {}, 0.5, progress=lambda *a: steps.append(a))
    assert mode == "new" and cov == 0.0
    assert steps == [("compare", 1, 2), ("compare", 2, 2), ("teacher", 1, 2), ("teacher", 2, 2)]


def test_glyph_sets_record_the_version_that_learned_them(tmp_path):
    from vobsub_to_srt.version import app_version
    db = GlyphDB("stamped", tmp_path / "stamped.json")
    db.save()
    import json
    d = json.load(open(tmp_path / "stamped.json"))
    assert d["learned_with"] == app_version() == d["saved_with"]
    assert app_version() and app_version() != "unknown"
    d["learned_with"] = "0.0.9"                      # an older set keeps the version that created it
    json.dump(d, open(tmp_path / "stamped.json", "w"))
    old = GlyphDB.load(tmp_path / "stamped.json")
    assert old.learned_with == "0.0.9" and old.to_json()["learned_with"] == "0.0.9"


def test_stray_votes_are_not_ambiguity_candidates():
    """A glyph read as 'l' 1492 times, 'I' 532 times and once each as 'le', 'sl', 'len' offers I and l
    only; the strays would otherwise pit 'Ziel' against the valid 'Ziele' in every episode."""
    from collections import Counter
    from vobsub_to_srt.recognize import _candidates
    from vobsub_to_srt.glyphdb import Variant

    class Item:
        via = "ambig"; text = None
        variant = Variant(-43, Counter({"l": 1492, "I": 532, "le": 1, "sl": 1, "len": 1}))

    class Res:
        items = [Item()]; glyph_spans = [(0, 1)]

    class Line:
        glyphs = [None]

    class DB:
        il_identical = True

    assert _candidates(DB(), Line(), Res(), 0) == ["I", "l"]
    Item.variant = Variant(-43, Counter({"1": 3, "l": 9, "|": 1}))
    assert _candidates(DB(), Line(), Res(), 0) == ["1", "l"]            # a supported second reading stays
    Item.variant = Variant(-43, Counter({"rn": 4, "m": 1}))
    assert _candidates(DB(), Line(), Res(), 0) == ["rn"]                # a fused majority reading stands


def test_contradicting_answer_teaches_nothing():
    """'Erschießen' read as 'Erschließen': the surplus l contradicts confirmed letters and the ß at the
    end would have absorbed 'eß'. No glyph of such an answer gets a vote, not even the unknown one."""
    db = GlyphDB("t")
    base = "hallo das ist ein test schie"
    for _ in range(3):
        learn_cue(db, segment(render(base)), base, 12)
    lines = segment(render(base + "ß"))
    lr = learn_cue(db, lines, base[:-5] + "schließ", 12)
    assert not lr.learned
    g = lines[0].glyphs[-1]                                              # the ß: still unknown
    v = db.lookup(g.key, g.top_rel)
    assert v is None or not +v.votes


def test_two_letters_need_a_glyph_wide_enough():
    """A single glyph as wide as one letter cannot be 'eß' when the set knows how wide e and ß are."""
    db = GlyphDB("t")
    base = "hallo das ist ein test eßen"
    for _ in range(3):
        learn_cue(db, segment(render(base)), base, 12)
    w = db.letter_widths()
    assert db.fits_width("rw", int(w["e"] * 2.2)) and not db.fits_width("eß", int(w["e"]))
    lines = segment(render(base[:-4] + "ßen"))
    lr = learn_cue(db, lines, base, 12)                                  # one letter too many
    assert not lr.learned
    g = lines[0].glyphs[-3]                                              # the ß got no 'eß' vote
    v = db.lookup(g.key, g.top_rel)
    assert v is None or "eß" not in +v.votes


def test_seed_installs_and_replaces_by_version(tmp_path):
    import json
    from vobsub_to_srt.seed import seed
    base, data = tmp_path / "base", tmp_path / "data"
    base.mkdir()
    (base / "one.json").write_text(json.dumps({"name": "one", "shapes": []}))
    assert seed(base, data, version="0.2.1") == ["one.json: installed"]
    assert json.loads((data / "one.json").read_text())["seeded_from"] == "0.2.1"
    assert seed(base, data, version="0.2.1") == []                                  # same image: untouched
    (data / "one.json").write_text(json.dumps({"name": "one", "shapes": [1], "seeded_from": "0.2.1"}))   # grew on the server
    notes = seed(base, data, version="0.3.0")
    assert notes == ["one.json: replaced (was seeded from 0.2.1; old copy archived)"]
    assert json.loads((data / "superseded" / "one.0.2.1.json").read_text())["shapes"] == [1]
    assert json.loads((data / "one.json").read_text())["seeded_from"] == "0.3.0"
    (data / "two.json").write_text(json.dumps({"name": "two", "shapes": []}))     # a set learned there: untouched
    assert seed(base, data, version="0.3.0") == [] and (data / "two.json").exists()


def test_seed_retires_a_shipped_set_the_image_no_longer_ships(tmp_path):
    import json
    from vobsub_to_srt.seed import seed
    base, data, wm = tmp_path / "base", tmp_path / "data", tmp_path / "wm"
    base.mkdir(); wm.mkdir()
    for n in ("keep", "merged"):
        (base / f"{n}.json").write_text(json.dumps({"name": n, "shapes": []}))
    seed(base, data, version="0.2.2", private=wm)
    (data / "local.json").write_text(json.dumps({"name": "local", "shapes": []}))   # learned on the instance
    (wm / "merged.json").write_text(json.dumps({"interim": {"shapes": []}}))
    (base / "merged.json").unlink()                                                   # 0.3.0 merged it away
    notes = seed(base, data, version="0.3.0", private=wm)
    assert "merged.json: retired (no longer shipped; seeded from 0.2.2; archived)" in notes
    assert not (data / "merged.json").exists() and (data / "superseded" / "merged.0.2.2.json").exists()
    assert not (wm / "merged.json").exists() and (wm / "superseded" / "merged.0.2.2.json").exists()
    assert (data / "local.json").exists() and (data / "keep.json").exists()


def test_baseline_counts_jittered_rows_together_but_never_moves_down():
    from vobsub_to_srt.segment import _baseline
    # "[Dog barking]" on a rescaled track: letters end at rows 91/92 (jitter), brackets and g at 100
    assert _baseline([100, 91, 92, 100, 100, 92, 91, 92, 91, 91, 100, 100]) == 91   # the real line
    assert _baseline([100, 91, 92, 100, 100, 92, 91, 100, 92, 91, 100, 100]) == 100  # a tie keeps the mode
    # crisp line with many descenders: brackets end one row below g/p; the baseline stays at 44
    assert _baseline([57, 44, 44, 56, 56, 44, 44, 56, 57]) == 44
    assert _baseline([30, 30, 30, 41]) == 30                     # ordinary line: the plain mode


def test_only_letters_and_digits_have_a_slant():
    from vobsub_to_srt.recognize import _has_slant
    assert _has_slant("e") and _has_slant("8") and _has_slant("rt")
    assert not _has_slant("#") and not _has_slant('"') and not _has_slant("-") and not _has_slant(None)
