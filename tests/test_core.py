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
    assert trusted_label(Counter({"a": 2, "e": 1}))[0] == "a"     # exactly 2/3
    assert trusted_label(Counter({"a": 2, "e": 2}))[0] is None
    assert trusted_label(Counter({"I": 4, "1": 1}))[0] == "I"


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
    assert lr.learned and lr.conflicts
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
