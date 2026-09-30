"""Pre-download Hunspell dictionaries for the lexicon gate (normally fetched automatically on first use).

usage: python tools/fetch_hunspell.py de en [fr es it nl]   (target: --dict-dir / $VOBSUB_TO_SRT_DICT_DIR / ./dictionaries)
"""
from __future__ import annotations

import sys

from vobsub_to_srt.lexicon import download_hunspell

if __name__ == "__main__":
    for lang in sys.argv[1:] or ["de", "en"]:
        print(lang, download_hunspell(lang))
