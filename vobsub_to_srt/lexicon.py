"""Lexicon gate for glyphs that are pixel-identical across labels (I/l/| in many sans-serif fonts).

Given the possible readings of an ambiguous word, pick one only if the evidence is clear:
  1. wordfreq: the most frequent reading is a known word and >= MIN_LEAD zipf above the others,
     or it is the only reading that is a known word at all
  2. Hunspell: exactly one reading is a valid spelling
  3. case consistency, only if NO reading is a known word (names, invented words): exactly one
     reading has a regular case pattern (lower, UPPER, Capitalized per hyphen/apostrophe part),
     e.g. ZORVANIA not ZORVANlA
Otherwise return None, and the cue goes to the VLM. Dictionary lookups are case-insensitive
(subtitles may be all-caps).
"""
from __future__ import annotations

import itertools
import re
import logging
import os
from pathlib import Path

log = logging.getLogger("vobsub_to_srt")

MIN_ZIPF = 1.0          # a reading must be a known word at all
MIN_LEAD = 1.5          # zipf lead over the next reading (= ~30x more frequent)
MAX_CANDIDATES = 64
REPAIR_LEAD = 2.0       # repairing a confidently read word needs a ~100x frequency lead
PUNCT = "\"'.,;:!?()[]{}-–—…«»„“”‚‘’¿¡"

def dict_dir() -> Path:
    """Where downloaded dictionaries live (default ./dictionaries, like db/ and cache/)."""
    return Path(os.environ.get("VOBSUB_TO_SRT_DICT_DIR", "dictionaries"))


def hunspell_dirs() -> list[Path]:
    return [dict_dir()] + SYSTEM_DIRS


# LibreOffice dictionaries (licenses per language: GPL/LGPL/MPL); downloaded on first use,
# never bundled with this project.
DICT_BASE = "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/"
DICT_SOURCES = {
    "de": ("de/de_DE_frami", "de_DE_frami"),
    "en": ("en/en_US", "en_US"),
    "fr": ("fr_FR/fr", "fr_FR"),
    "es": ("es/es_ES", "es_ES"),
    "it": ("it_IT/it_IT", "it_IT"),
    "nl": ("nl_NL/nl_NL", "nl_NL"),
}


def download_hunspell(lang: str, dest: Path | None = None) -> Path | None:
    """Fetch the .aff/.dic pair for `lang` into dest; returns the dictionary base path."""
    if lang not in DICT_SOURCES:
        return None
    import httpx
    dest = dest or dict_dir()
    src, name = DICT_SOURCES[lang]
    dest.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60, follow_redirects=True) as c:
        for ext in (".aff", ".dic"):
            r = c.get(DICT_BASE + src + ext)
            r.raise_for_status()
            tmp = dest / f"{name}{ext}.part"
            tmp.write_bytes(r.content)
            tmp.replace(dest / f"{name}{ext}")
    return dest / name


SYSTEM_DIRS = [
    Path("/usr/share/hunspell"),
    Path("/usr/share/myspell/dicts"),
    Path("/usr/share/myspell"),
    Path("/Library/Spelling"),
]
# preferred dictionary names per ISO 639-1 language code
HUNSPELL_NAMES = {
    "de": ["de_DE_frami", "de_DE", "de_AT", "de_CH"],
    "en": ["en_US", "en_GB"],
    "fr": ["fr_FR", "fr"],
    "es": ["es_ES", "es"],
    "it": ["it_IT"],
    "nl": ["nl_NL"],
}


def _core(word: str) -> str:
    return word.strip(PUNCT)


def case_regular(word: str) -> bool:
    """lower, UPPER or Capitalized in every alphabetic part (parts split at - ' ´ ’ etc.)."""
    parts = [p for p in re.split(r"[^\w]|\d|_", word) if p]
    if not parts:
        return False
    return all(p.islower() or p.isupper() or (p[0].isupper() and p[1:].islower()) for p in parts)


_HUNSPELL: dict[str, object] = {}


def _hunspell(base: str):
    d = _HUNSPELL.get(base)
    if d is None:
        from spylls.hunspell import Dictionary
        d = _HUNSPELL[base] = Dictionary.from_files(base)
    return d


class Lexicon:
    def __init__(self, lang: str, use_wordfreq: bool = True, use_hunspell: bool = True, use_case: bool = True,
                 download: bool = True):
        self.lang = lang
        self.zipf = None
        self.hunspell = None
        if use_wordfreq:
            try:
                from wordfreq import available_languages, zipf_frequency
                if lang in available_languages():
                    self.zipf = lambda w: zipf_frequency(w, lang)
                else:
                    log.info("lexicon: wordfreq has no data for %r", lang)
            except ImportError:
                log.info("lexicon: wordfreq not installed")
        if use_hunspell:
            self.hunspell = self._load_hunspell(lang, download)
        self.use_case = use_case
        self.stats = {"wordfreq": 0, "hunspell": 0, "case": 0, "undecided": 0}

    @staticmethod
    def _load_hunspell(lang: str, download: bool = True):
        """The parsed dictionary is shared by every lexicon of this process (parsing a .dic takes
        seconds; a web worker converts many files, and the dictionary is only ever read)."""
        try:
            from spylls.hunspell import Dictionary
        except ImportError:
            log.info("lexicon: spylls not installed, Hunspell tier disabled")
            return None
        for d in hunspell_dirs():
            for name in HUNSPELL_NAMES.get(lang, [lang]):
                if (d / f"{name}.dic").exists() and (d / f"{name}.aff").exists():
                    log.info("lexicon: Hunspell dictionary %s", d / name)
                    return _hunspell(str(d / name))
        if download and lang in DICT_SOURCES:
            try:
                base = download_hunspell(lang)
                log.info("lexicon: downloaded Hunspell dictionary %s", base)
                return Dictionary.from_files(str(base))
            except Exception as e:     # offline etc.: the other tiers still work
                log.warning("lexicon: could not download Hunspell dictionary for %r: %s", lang, e)
                return None
        log.info("lexicon: no Hunspell dictionary for %r", lang)
        return None

    @property
    def active(self) -> bool:
        return self.zipf is not None or self.hunspell is not None or self.use_case

    def _valid(self, word: str) -> bool:
        core = _core(word)
        if not core:
            return False
        return any(self.hunspell.lookup(v) for v in {core, core.lower(), core.capitalize()})

    def resolve(self, options: list[list[str]]) -> str | None:
        """options: per glyph item, the possible texts. Returns the chosen full word or None."""
        n = 1
        for o in options:
            n *= max(1, len(o))
        if n < 2 or n > MAX_CANDIDATES:
            return None
        cands = sorted({"".join(p) for p in itertools.product(*options)})
        if self.use_case:
            # "WAILING" vs "WAlLING": a reading with a lone lower-case l inside a capital word (or a
            # capital I inside a lower-case one) is not how words are written, whatever the corpus
            # says about the rest; one case-regular reading among irregular ones decides
            regular = [c for c in cands if case_regular(_core(c))]
            if len(regular) == 1 and len(cands) > 1:
                self.stats["case"] += 1
                return regular[0]
        known = False
        if self.zipf is not None:
            scored = sorted(((self.zipf(_core(c).lower()), c) for c in cands), reverse=True)
            (z1, best), z2 = scored[0], scored[1][0]
            known = z1 >= MIN_ZIPF
            # clear lead over another known word, or the only reading that is a word at all
            if known and (z1 - z2 >= MIN_LEAD or z2 == 0.0):
                self.stats["wordfreq"] += 1
                return best
        if self.hunspell is not None:
            valid = [c for c in cands if self._valid(c)]
            if len(valid) == 1:
                self.stats["hunspell"] += 1
                return valid[0]
            known = known or bool(valid)
        if self.use_case and not known:
            regular = [c for c in cands if case_regular(_core(c))]
            if len(regular) == 1:
                self.stats["case"] += 1
                return regular[0]
        self.stats["undecided"] += 1
        return None


    def _known(self, word: str) -> bool:
        core = _core(word)
        if not core:
            return False
        if self.zipf is not None and self.zipf(core.lower()) >= MIN_ZIPF:
            return True
        return self.hunspell is not None and self._valid(word)

    def repair(self, items: list[str]) -> str | None:
        """Safety net for I/l glyphs not (yet) known to be ambiguous: replace the word as read by an
        I<->l swap only if the current spelling is not valid (Hunspell) and the swap is far more
        frequent (wordfreq, >= REPAIR_LEAD zipf; web corpora contain OCR errors like 'ldee')."""
        current = "".join(items)
        if not any(c in "Il" for c in current) or (self.zipf is None and self.hunspell is None):
            return None
        options = []
        for t in items:
            if t and all(c in "Il" for c in t):
                options.append(["".join(p) for p in itertools.product("Il", repeat=len(t))])
            else:
                options.append([t])
        n = 1
        for o in options:
            n *= len(o)
        if n < 2 or n > MAX_CANDIDATES:
            return None
        if self.hunspell is not None and self._valid(current):
            return None
        alts = sorted({"".join(p) for p in itertools.product(*options)} - {current})
        chosen = None
        if self.zipf is not None:
            zc = self.zipf(_core(current).lower())
            scored = sorted(((self.zipf(_core(a).lower()), a) for a in alts), reverse=True)
            z1, best = scored[0]
            z2 = scored[1][0] if len(scored) > 1 else 0.0
            if z1 >= MIN_ZIPF and z1 - zc >= REPAIR_LEAD and z1 - z2 >= MIN_LEAD:
                chosen = best
        elif self.hunspell is not None:
            valid = [a for a in alts if self._valid(a)]
            chosen = valid[0] if len(valid) == 1 else None
        if chosen:
            self.stats["repair"] = self.stats.get("repair", 0) + 1
        return chosen


def make_lexicon(mode: str, lang: str, download: bool = True) -> Lexicon | None:
    if mode == "off":
        return None
    lx = Lexicon(lang, use_wordfreq=mode in ("auto", "wordfreq"), use_hunspell=mode in ("auto", "hunspell"),
                 use_case=mode != "off", download=download)
    return lx if lx.active else None
