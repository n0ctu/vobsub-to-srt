"""Compare two SRT files cue by cue (by start time): CER, exact-match rate, italic agreement.

usage: python tools/compare_srt.py reference.srt candidate.srt [--show N]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

TAG = re.compile(r"</?[biu]>")


def parse(path: Path) -> dict[str, str]:
    out = {}
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip()):
        lines = block.splitlines()
        if len(lines) >= 3 and "-->" in lines[1]:
            out[lines[1].split("-->")[0].strip()] = "\n".join(lines[2:])
    return out


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--show", type=int, default=30)
    ap.add_argument("--simplify", action="store_true", help="fold typographic variants on both sides first")
    a = ap.parse_args()
    ref, cand = parse(a.reference), parse(a.candidate)
    if a.simplify:
        from vobsub_to_srt.simplify import simplify
        ref = {k: simplify(v) for k, v in ref.items()}
        cand = {k: simplify(v) for k, v in cand.items()}
    keys = sorted(set(ref) | set(cand))
    errs = chars = exact = ital_diff = 0
    diffs = []
    for k in keys:
        r, c = ref.get(k, ""), cand.get(k, "")
        rp, cp = TAG.sub("", r), TAG.sub("", c)
        e = lev(rp, cp)
        errs += e
        chars += len(rp)
        if e == 0:
            exact += 1
            if r != c:
                ital_diff += 1
        else:
            diffs.append((k, r, c))
    print(f"cues: ref {len(ref)} cand {len(cand)}  text-exact {exact}/{len(keys)}  "
          f"CER {100 * errs / max(1, chars):.3f}%  italic-tag differences {ital_diff}")
    for k, r, c in diffs[:a.show]:
        print(f"--- {k}\n  ref:  {r!r}\n  cand: {c!r}")


if __name__ == "__main__":
    main()
