"""SRT output and text normalization."""
from __future__ import annotations

from pathlib import Path

from .styling import parse_styled, render_styled


def fmt_ts(ms: int) -> str:
    ms = max(0, ms)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def normalize_styled_line(line: str) -> str:
    """Re-render a (VLM) line so <b>/<i>/<u> tags are balanced, nested and per line."""
    return render_styled(parse_styled(line))


def normalize_text(text: str) -> str:
    return "\n".join(normalize_styled_line(l) for l in text.split("\n") if l.strip())


def write_srt(path: Path, entries: list[tuple[int, int, str]]) -> None:
    parts = []
    n = 0
    for start, end, text in entries:
        if not text.strip():
            continue
        n += 1
        parts.append(f"{n}\n{fmt_ts(start)} --> {fmt_ts(end)}\n{text}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
