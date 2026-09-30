"""Measure VLM transcription accuracy with vs. without previous-cue context.

Context comes from a verified transcript (default experiments/out-gt/), so this is the best case for context.
usage: python tools/eval_context.py Subs/X.idx [...] --context 12
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_srt import TAG, lev, parse  # noqa: E402
from vobsub_to_srt.segment import fill_mask, segment  # noqa: E402
from vobsub_to_srt.srt import fmt_ts, normalize_text  # noqa: E402
from vobsub_to_srt.vlm import VLMClient, mask_to_png  # noqa: E402
from vobsub_to_srt.vobsub import load_vobsub  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--gt-dir", type=Path, default=Path("experiments/out-gt"))
    ap.add_argument("--nocontext-dir", type=Path, default=Path("experiments/out-vlm3"))
    ap.add_argument("--context", type=int, default=12)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--tool", action="store_true")
    a = ap.parse_args()
    async with VLMClient(cache_dir=Path('cache/vlm'), concurrency=a.concurrency, use_tool=a.tool) as client:
        for idx_path in a.inputs:
            idx, cues = load_vobsub(idx_path)
            lang = idx.tracks[0].lang
            gt = parse(a.gt_dir / (idx_path.stem + ".srt"))
            base = parse(a.nocontext_dir / (idx_path.stem + ".srt"))
            keys = [fmt_ts(c.start_ms) for c in cues]
            gt_plain = [TAG.sub("", gt.get(k, "")).replace("\n", " ") for k in keys]

            async def one(i):
                m = fill_mask(cues[i])
                hist = [t for t in gt_plain[max(0, i - a.context):i] if t]
                return await client.transcribe(mask_to_png(m), max(1, len(segment(m))), lang, context=hist)

            res = await asyncio.gather(*(one(i) for i in range(len(cues))), return_exceptions=True)
            err_ctx = err_base = chars = 0
            diffs = []
            for k, r in zip(keys, res):
                g = TAG.sub("", gt.get(k, ""))
                c = TAG.sub("", normalize_text(r)) if isinstance(r, str) else ""
                b = TAG.sub("", base.get(k, ""))
                ec, eb = lev(g, c), lev(g, b)
                err_ctx += ec
                err_base += eb
                chars += len(g)
                if ec or eb:
                    diffs.append({"t": k, "truth": g, "no_context": b, "context": c})
            print(f"{idx_path.name} (tool={a.tool}): chars {chars} | no context: {err_base} errors in "
                  f"{sum(1 for d in diffs if d['no_context'] != d['truth'])} cues | "
                  f"context {a.context}: {err_ctx} errors in {sum(1 for d in diffs if d['context'] != d['truth'])} cues")
            for d in diffs:
                print(json.dumps(d, ensure_ascii=False))
        print(f"API requests {client.calls}, 429s {client.throttled}, cache hits {client.cache_hits}")


asyncio.run(main())
