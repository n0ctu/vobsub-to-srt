"""CLI: python -m vobsub_to_srt Subs/*.idx"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

from .pipeline import Options, process_file
from .vlm import VLMClient


def main() -> None:
    ap = argparse.ArgumentParser(description="VobSub -> SRT with self-healing nOCR and VLM fallback")
    ap.add_argument("inputs", nargs="+", type=Path, help=".idx files")
    ap.add_argument("--glyph-memory-dir", type=Path, default=Path("glyph-memory"),
                    help="font glyph DBs (publishable: no subtitle content)")
    ap.add_argument("--out-dir", type=Path, default=Path("out"))
    ap.add_argument("--debug-dir", type=Path, default=Path("debug"))
    ap.add_argument("--cache-dir", type=Path, default=Path("cache/vlm"))
    ap.add_argument("--mode", choices=["hybrid", "vlm-only", "nocr-only"], default="hybrid")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--track", type=int, default=0)
    ap.add_argument("--context", type=int, default=12, help="previous cues given to the VLM as reference (0 = off)")
    ap.add_argument("--tool", action="store_true", help="VLM submits transcripts via a forced tool call")
    ap.add_argument("--word-memory", action="store_true",
                    help="remember resolved words (stored in --word-memory-dir, never in the glyph memory)")
    ap.add_argument("--word-memory-dir", type=Path, default=Path("word-memory"),
                    help="private per-DB data (not publishable): learned image hashes and the optional word memory")
    ap.add_argument("--keep-special-chars", action="store_true",
                    help="keep typographic variants (´ ’ „ “ – — … ligatures) instead of folding them to ' \" - ...")
    ap.add_argument("--lexicon", choices=["auto", "wordfreq", "hunspell", "off"], default="auto",
                    help="tie-break for pixel-identical I/l glyphs (default: wordfreq, then Hunspell, then case "
                         "consistency for unknown words)")
    ap.add_argument("--dict-dir", type=Path, default=Path("dictionaries"),
                    help="Hunspell dictionaries (downloaded here on first use; system dictionaries are used too)")
    ap.add_argument("--no-dict-download", action="store_true", help="never download Hunspell dictionaries")
    ap.add_argument("--debug-rescale", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    os.environ["VOBSUB_TO_SRT_DICT_DIR"] = str(a.dict_dir.resolve())
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    opts = Options(db_dir=a.glyph_memory_dir, out_dir=a.out_dir, debug_dir=a.debug_dir, batch_size=a.batch_size,
                   mode=a.mode, track=a.track, context=a.context,
                   rescale=a.debug_rescale, lexicon=a.lexicon,
                   keep_special_chars=a.keep_special_chars,
                   word_memory=a.word_memory, private_dir=a.word_memory_dir, download_dicts=not a.no_dict_download)

    problems = []
    for p in a.inputs:
        if p.suffix.lower() != ".idx":
            problems.append(f"{p}: expected a .idx file")
        elif not p.is_file():
            problems.append(f"{p}: file not found")
        elif not p.with_suffix(".sub").is_file():
            problems.append(f"{p}: companion file {p.with_suffix('.sub').name} not found next to it")
    if problems:
        raise SystemExit("\n".join(problems))

    async def run():
        if a.mode == "nocr-only":
            for p in a.inputs:
                await process_file(p, None, opts)
            return
        async with VLMClient(cache_dir=a.cache_dir, concurrency=a.concurrency, use_tool=a.tool) as client:
            for p in a.inputs:
                await process_file(p, client, opts)

    asyncio.run(run())


if __name__ == "__main__":
    main()
