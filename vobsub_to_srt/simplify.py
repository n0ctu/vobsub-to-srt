"""Character-set simplification: fold typographic variants that do not change the meaning.

On by default (--keep-special-chars turns it off). Applied to everything the VLM returns before it
is learned, compared or written, so e.g. an acute accent used as apostrophe (O´Neil) and a real
apostrophe (O'Neil) are one class instead of competing labels for the same glyph.
"""
from __future__ import annotations

import unicodedata

_MAP = {}
for ch in "´`’‘‚′ʼʻ‹›‛":
    _MAP[ord(ch)] = "'"                      # apostrophes, single quotes
for ch in "„“”«»″‟＂〝〞":
    _MAP[ord(ch)] = '"'                      # double quotes
for ch in "‐‑‒–—―−⁃﹘﹣－":
    _MAP[ord(ch)] = "-"                      # hyphens, dashes, minus
_MAP[ord("…")] = "..."                       # ellipsis
_MAP[ord("․")] = "."                         # one-dot leader
for lig, plain in {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}.items():
    _MAP[ord(lig)] = plain                   # typographic ligatures
for ch in "            　":
    _MAP[ord(ch)] = " "                      # non-breaking / thin / wide spaces
for ch in "­​‌‍⁠﻿":
    _MAP[ord(ch)] = None                     # soft hyphen, zero-width characters

CHARSET_SIMPLIFIED = "simplified"
CHARSET_LITERAL = "literal"


def simplify(text: str) -> str:
    """NFC-compose (a + combining diaeresis -> ä), then fold variants."""
    return unicodedata.normalize("NFC", text).translate(_MAP)
