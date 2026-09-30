"""Quality-control sheets: subtitle image and transcript side by side for a sample of cues.

For each track, draws random cues (reproducible seed) plus "risky" cues (VLM-transcribed, re-asked or
flagged, read from the report), each as the cue image with our transcript rendered below it
(italic slanted, bold, underlined), so a human can verify them at a glance.

usage: python tools/qc_sheet.py SUB.idx OUT_DIR/NAME.srt [--random 6] [--risky 4] [--seed 1] [--sheet out.png]
       [--page-size 10] [--exclude earlier.txt ...]
Writes the sampled timestamps to <sheet>.txt (usable as --exclude for a later, disjoint sample).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vobsub_to_srt.segment import fill_mask
from vobsub_to_srt.srt import fmt_ts
from vobsub_to_srt.styling import parse_styled
from vobsub_to_srt.vobsub import load_vobsub

FONT = {"": "DejaVuSans.ttf", "i": "DejaVuSans-Oblique.ttf", "b": "DejaVuSans-Bold.ttf",
        "bi": "DejaVuSans-BoldOblique.ttf"}
SIZE = 30
PAD = 10


def parse_srt(path: Path) -> dict[str, tuple[int, str]]:
    out = {}
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip()):
        lines = block.splitlines()
        if len(lines) >= 3 and "-->" in lines[1]:
            out[lines[1].split("-->")[0].strip()] = (int(lines[0]), "\n".join(lines[2:]))
    return out


def render_text(text: str, width: int) -> Image.Image:
    lines = text.split("\n")
    im = Image.new("L", (width, (SIZE + 8) * len(lines) + 4), 235)
    d = ImageDraw.Draw(im)
    for li, line in enumerate(lines):
        x, y = 4, 2 + li * (SIZE + 8)
        for ch, st in parse_styled(line):
            font = ImageFont.truetype(FONT["".join(f for f in "bi" if f in st)], SIZE)
            w = d.textlength(ch, font=font)
            d.text((x, y), ch, fill=0, font=font)
            if "u" in st:
                d.line((x, y + SIZE + 2, x + w, y + SIZE + 2), fill=0, width=2)
            x += w
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("idx", type=Path)
    ap.add_argument("srt", type=Path)
    ap.add_argument("--random", type=int, default=6)
    ap.add_argument("--risky", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sheet", type=Path, default=None)
    ap.add_argument("--page-size", type=int, default=10, help="cues per sheet image")
    ap.add_argument("--exclude", type=Path, nargs="*", default=[], help="timestamp lists of earlier samples")
    a = ap.parse_args()

    _, cues = load_vobsub(a.idx)
    by_ts = {fmt_ts(c.start_ms): c for c in cues}
    srt = parse_srt(a.srt)
    report_path = a.srt.with_suffix(".report.json")
    risky_idx: set[int] = set()
    if report_path.exists():
        rep = json.loads(report_path.read_text())
        risky_idx = {int(k) for k in rep.get("vlm_raw", {})} | {int(k) for k in rep.get("flagged", {})}
    rng = random.Random(a.seed)
    excluded = {line.strip() for f in a.exclude if f.exists() for line in f.read_text().splitlines()}
    stamps = sorted(ts for ts in srt if ts in by_ts and ts not in excluded)
    risky = [ts for ts in stamps if by_ts[ts].index in risky_idx]
    pick = rng.sample(risky, min(a.risky, len(risky)))
    rest = [ts for ts in stamps if ts not in pick]
    pick += rng.sample(rest, min(a.random, len(rest)))
    pick.sort()

    rows = []
    for ts in pick:
        cue = by_ts[ts]
        mask = fill_mask(cue)
        img = Image.fromarray(np.where(mask, 0, 255).astype(np.uint8))
        n, text = srt[ts]
        tag = "VLM/flagged" if cue.index in risky_idx else "random"
        rows.append((f"#{n} {ts} [{tag}]", img, text))
    out = a.sheet or a.srt.with_suffix(".qc.png")
    out.with_suffix(".txt").write_text("\n".join(pick) + "\n")
    pages = [rows[i:i + a.page_size] for i in range(0, len(rows), a.page_size)]
    for n, page in enumerate(pages, 1):
        name = out if len(pages) == 1 else out.with_name(f"{out.stem}_p{n}{out.suffix}")
        draw_sheet(page).save(name)
        print(f"{name}: {len(page)} cues ({sum('VLM' in r[0] for r in page)} risky)")


def draw_sheet(rows) -> Image.Image:
    width = max(max(r[1].width for r in rows), 900) + 2 * PAD
    parts = []
    label_font = ImageFont.truetype("DejaVuSans.ttf", 16)
    for label, img, text in rows:
        lab = Image.new("L", (width, 22), 255)
        ImageDraw.Draw(lab).text((PAD, 2), label, fill=90, font=label_font)
        cue_im = Image.new("L", (width, img.height + 8), 255)
        cue_im.paste(img, (PAD, 4))
        parts += [lab, cue_im, render_text(text, width), Image.new("L", (width, 14), 160)]
    sheet = Image.new("L", (width, sum(p.height for p in parts)), 255)
    y = 0
    for p in parts:
        sheet.paste(p, (0, y))
        y += p.height
    return sheet


if __name__ == "__main__":
    main()
