"""Subtitle styling: <b>, <i>, <u> parsing and rendering."""
from __future__ import annotations

import re

_TAG = re.compile(r"</?\s*([a-zA-Z]+)[^>]*>")

STYLE_ORDER = "biu"   # tag nesting order, outermost first


def parse_styled(text: str) -> list[tuple[str, str]]:
    """'<i>Hi</i> <b>you</b>' -> [('H','i'),('i','i'),(' ',''),('y','b'),...]. Unknown tags are dropped."""
    out: list[tuple[str, str]] = []
    active: set[str] = set()
    pos = 0

    def style() -> str:
        return "".join(f for f in STYLE_ORDER if f in active)

    for m in _TAG.finditer(text):
        out += [(c, style()) for c in text[pos:m.start()]]
        tag = m.group(1).lower()
        if tag in STYLE_ORDER:
            if m.group(0).startswith("</"):
                active.discard(tag)
            else:
                active.add(tag)
        pos = m.end()
    out += [(c, style()) for c in text[pos:]]
    res: list[tuple[str, str]] = []
    for c, st in out:
        if c.isspace():
            if res and res[-1][0] != " ":
                res.append((" ", st))
        else:
            res.append((c, st))
    while res and res[-1][0] == " ":
        res.pop()
    return res


def render_styled(chars: list[tuple[str, str]]) -> str:
    """Inverse of parse_styled: balanced, properly nested tags; a space is styled only if both neighbours are."""
    out: list[str] = []
    stack: list[str] = []
    n = len(chars)
    for k, (c, st) in enumerate(chars):
        if c == " ":
            prev = chars[k - 1][1] if k else ""
            nxt = chars[k + 1][1] if k + 1 < n else ""
            st = "".join(f for f in STYLE_ORDER if f in prev and f in nxt)
        # close from the top down to the lowest tag that is no longer wanted, then open what's missing
        drop = next((k for k, f in enumerate(stack) if f not in st), len(stack))
        while len(stack) > drop:
            out.append(f"</{stack.pop()}>")
        for f in STYLE_ORDER:
            if f in st and f not in stack:
                out.append(f"<{f}>")
                stack.append(f)
        out.append(c)
    while stack:
        out.append(f"</{stack.pop()}>")
    return "".join(out)


def majority_style(styles: list[str]) -> str:
    return "".join(f for f in STYLE_ORDER if sum(f in s for s in styles) * 2 > len(styles))


