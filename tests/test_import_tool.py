"""tools/memory_import.py: classifying pulled glyph sets against the baseline."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

from vobsub_to_srt.glyphdb import GlyphDB

spec = importlib.util.spec_from_file_location("memory_import", Path(__file__).resolve().parent.parent / "tools" / "memory_import.py")
mi = importlib.util.module_from_spec(spec)
sys.modules["memory_import"] = mi
spec.loader.exec_module(mi)


def _glyph(seed: int, h: int = 14, w: int = 9) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bits = rng.random((h, w)) < 0.5
    bits[0, :] = bits[-1, :] = True
    return bits


def _set(path: Path, letters: dict[str, int], reads: int = 2, learned_with: str | None = "0.1.0") -> GlyphDB:
    """A set with one confirmed cluster per letter (bitmap chosen by `seed`), read `reads` times."""
    db = GlyphDB(path.stem, path)
    for label, seed in letters.items():
        for _ in range(reads):
            db.add_vote(f"k{seed}", _glyph(seed), 0, label, "")
    db.learned_with = learned_with
    db.journal = None
    db.save(final=True)
    return db


def test_new_update_merge_and_refusals(tmp_path):
    repo, pulled = tmp_path / "repo", tmp_path / "pulled"
    repo.mkdir(); pulled.mkdir()
    _set(repo / "shipped.json", {"a": 1, "b": 2, "c": 3})

    _set(pulled / "shipped.json", {"a": 1, "b": 2, "c": 3, "d": 4})           # grown copy of the shipped set
    _set(pulled / "sibling.json", {"a": 1, "b": 2, "e": 5})                   # same font, learned under another name
    _set(pulled / "other.json", {"x": 11, "y": 12, "z": 13})                  # a different font
    _set(pulled / "old.json", {"q": 21, "r": 22}, learned_with=None)          # from before the version stamp
    w = _set(pulled / "wrong.json", {"a": 1, "b": 2, "f": 6})
    for _ in range(5):
        w.add_vote("k3", _glyph(3), 0, "o", "")                                # the shipped 'c' read as 'o'
    w.journal = None; w.save(final=True)

    plans = {p.name: p for p in mi.plans_for(pulled, repo, "0.1.0", allow_unversioned=False)}
    assert plans["shipped"].action == "update" and [c.label for c in plans["shipped"].new] == ["d"]
    assert not plans["shipped"].refused and not plans["shipped"].changed
    assert plans["sibling"].action == "merge" and plans["sibling"].target == "shipped"
    assert [c.label for c in plans["sibling"].new] == ["e"]
    assert plans["other"].action == "new" and len(plans["other"].new) == 3
    assert plans["old"].refused and "learned_with" in plans["old"].refused[0]
    assert not mi.plans_for(pulled, repo, "0.1.0", allow_unversioned=True)[1].refused   # old.json accepted
    assert plans["wrong"].action == "merge" and plans["wrong"].changed == [("k3", "c", "o")] and plans["wrong"].refused

    out = {p.name: mi.apply_plan(p, repo, force=False) for p in plans.values()}
    assert out["wrong"].startswith("wrong: skipped") and out["old"].startswith("old: skipped")
    assert sorted(p.stem for p in repo.glob("*.json")) == ["other", "shipped"]
    assert not list(repo.glob("*.lock"))
    merged = mi.confirmed_clusters(GlyphDB.load(repo / "shipped.json"))
    assert sorted(c.label for c in merged.values()) == ["a", "b", "c", "d", "e"]
    again = {p.name: p for p in mi.plans_for(pulled, repo, "0.1.0", allow_unversioned=True)}
    assert not again["shipped"].new and not again["sibling"].new and again["other"].action == "update"
