"""VobSub (.idx/.sub) parsing and SPU bitmap decoding."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Cue:
    index: int
    start_ms: int
    end_ms: int | None
    image: np.ndarray            # (h, w) uint8, values 0..3 = bg, pattern, emph1, emph2
    colors: list[tuple[int, int, int]]   # rgb per 2-bit value
    alpha: list[int]             # 0..15 per 2-bit value
    x: int = 0
    y: int = 0
    forced: bool = False


@dataclass
class IdxTrack:
    lang: str
    index: int
    entries: list[tuple[int, int]] = field(default_factory=list)  # (ms, filepos)


@dataclass
class Idx:
    size: tuple[int, int]
    palette: list[tuple[int, int, int]]
    tracks: list[IdxTrack]


_TS_RE = re.compile(r"timestamp:\s*(-?)(\d+):(\d+):(\d+):(\d+),\s*filepos:\s*([0-9a-fA-F]+)")


def parse_idx(path: Path) -> Idx:
    return parse_idx_text(path.read_text(encoding="latin-1"))


def parse_idx_text(text: str) -> Idx:
    size = (720, 576)
    palette: list[tuple[int, int, int]] = []
    tracks: list[IdxTrack] = []
    offset_ms = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("size:"):
            w, h = line[5:].strip().split("x")
            size = (int(w), int(h))
        elif line.startswith("palette:"):
            palette = [(int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))
                       for c in (p.strip() for p in line[8:].split(","))]
        elif line.startswith("id:"):
            m = re.match(r"id:\s*(\w*),\s*index:\s*(\d+)", line)
            if m:
                tracks.append(IdxTrack(m.group(1), int(m.group(2))))
            offset_ms = 0
        elif line.startswith("delay:"):
            m = re.match(r"delay:\s*(-?)(\d+):(\d+):(\d+):(\d+)", line)
            if m:
                ms = ((int(m.group(2)) * 60 + int(m.group(3))) * 60 + int(m.group(4))) * 1000 + int(m.group(5))
                offset_ms += -ms if m.group(1) else ms
        elif line.startswith("timestamp:"):
            m = _TS_RE.match(line)
            if not m:
                continue
            if not tracks:
                tracks.append(IdxTrack("", 0))
            ms = ((int(m.group(2)) * 60 + int(m.group(3))) * 60 + int(m.group(4))) * 1000 + int(m.group(5))
            if m.group(1):
                ms = -ms
            tracks[-1].entries.append((ms + offset_ms, int(m.group(6), 16)))
    if len(palette) < 16:
        palette += [(0, 0, 0)] * (16 - len(palette))
    return Idx(size, palette, tracks)


def _read_spu_packet(data: bytes, pos: int, stream_id: int) -> bytes | None:
    """Collect PES payloads for one SPU starting at the pack at `pos`."""
    buf = bytearray()
    spu_size = None
    n = len(data)
    while pos + 4 <= n:
        if data[pos:pos + 3] != b"\x00\x00\x01":
            pos += 1
            continue
        code = data[pos + 3]
        if code == 0xBA:  # pack header
            if data[pos + 4] & 0xC0 == 0x40:   # MPEG-2
                pos += 14 + (data[pos + 13] & 7)
            else:                              # MPEG-1
                pos += 12
            continue
        if code in (0xB9,):
            pos += 4
            continue
        plen = int.from_bytes(data[pos + 4:pos + 6], "big")
        body_start = pos + 6
        next_pos = body_start + plen
        if code == 0xBD:
            hdr_len = data[body_start + 2]
            payload = data[body_start + 3 + hdr_len:next_pos]
            if payload and payload[0] == stream_id:
                chunk = payload[1:]
                if spu_size is None:
                    spu_size = int.from_bytes(chunk[0:2], "big")
                buf += chunk
                if len(buf) >= spu_size:
                    return bytes(buf[:spu_size])
        pos = next_pos
    return bytes(buf) if spu_size and buf else None


def _decode_rle(nibbles: list[int], offset: int, width: int, height: int, out: np.ndarray, start_row: int) -> None:
    """Decode one interlaced field (every other row) into `out`. `nibbles` is the SPU as a nibble list."""
    p = offset * 2
    total = len(nibbles)
    for y in range(start_row, height, 2):
        row = out[y]
        x = 0
        while x < width and p < total:
            v = nibbles[p]
            p += 1
            if v < 0x4:
                v = (v << 4) | nibbles[p]
                p += 1
                if v < 0x10:
                    v = (v << 4) | nibbles[p]
                    p += 1
                    if v < 0x40:
                        v = (v << 4) | nibbles[p]
                        p += 1
            run = v >> 2
            if run == 0 or x + run > width:
                run = width - x
            color = v & 3
            if color:
                row[x:x + run] = color
            x += run
        if p & 1:
            p += 1  # byte-align at end of line


def decode_spu(spu: bytes, palette: list[tuple[int, int, int]]):
    """Returns (image, colors, alpha, x, y, start_delay_ms, stop_delay_ms, forced)."""
    ctrl = int.from_bytes(spu[2:4], "big")
    color_idx = [0, 1, 2, 3]
    alpha = [0, 15, 15, 15]
    x1 = y1 = 0
    x2 = y2 = -1
    off_top = off_bot = 0
    start_delay = 0
    stop_delay = None
    forced = False
    pos = ctrl
    seen = set()
    while pos + 4 <= len(spu) and pos not in seen:
        seen.add(pos)
        delay = int.from_bytes(spu[pos:pos + 2], "big")
        nxt = int.from_bytes(spu[pos + 2:pos + 4], "big")
        delay_ms = delay * 1024 // 90
        p = pos + 4
        while p < len(spu):
            cmd = spu[p]
            p += 1
            if cmd == 0x00:
                forced = True
                start_delay = delay_ms
            elif cmd == 0x01:
                start_delay = delay_ms
            elif cmd == 0x02:
                stop_delay = delay_ms
            elif cmd == 0x03:
                b0, b1 = spu[p], spu[p + 1]
                color_idx = [b1 & 0xF, b1 >> 4, b0 & 0xF, b0 >> 4]
                p += 2
            elif cmd == 0x04:
                b0, b1 = spu[p], spu[p + 1]
                alpha = [b1 & 0xF, b1 >> 4, b0 & 0xF, b0 >> 4]
                p += 2
            elif cmd == 0x05:
                b = spu[p:p + 6]
                x1 = (b[0] << 4) | (b[1] >> 4)
                x2 = ((b[1] & 0xF) << 8) | b[2]
                y1 = (b[3] << 4) | (b[4] >> 4)
                y2 = ((b[4] & 0xF) << 8) | b[5]
                p += 6
            elif cmd == 0x06:
                off_top = int.from_bytes(spu[p:p + 2], "big")
                off_bot = int.from_bytes(spu[p + 2:p + 4], "big")
                p += 4
            elif cmd == 0x07:  # CHG_COLCON: skip its parameter block
                p += int.from_bytes(spu[p:p + 2], "big")
            else:  # 0xFF end or unknown
                break
        if nxt == pos:
            break
        pos = nxt
    w, h = x2 - x1 + 1, y2 - y1 + 1
    img = np.zeros((max(h, 0), max(w, 0)), dtype=np.uint8)
    if w > 0 and h > 0:
        arr = np.frombuffer(spu, np.uint8)
        nibbles = np.empty(len(arr) * 2 + 8, np.uint8)
        nibbles[0:len(arr) * 2:2] = arr >> 4
        nibbles[1:len(arr) * 2:2] = arr & 0xF
        nibbles[len(arr) * 2:] = 0
        nl = nibbles.tolist()
        _decode_rle(nl, off_top, w, h, img, 0)
        _decode_rle(nl, off_bot, w, h, img, 1)
    colors = [palette[i] for i in color_idx]
    return img, colors, alpha, x1, y1, start_delay, stop_delay, forced


def load_vobsub(idx_path: Path, track: int = 0) -> tuple[Idx, list[Cue]]:
    """Load an .idx/.sub pair from disk."""
    sub_path = idx_path.with_suffix(".sub")
    if not sub_path.is_file():
        raise FileNotFoundError(f"{idx_path}: companion file {sub_path.name} not found next to it")
    return load_vobsub_bytes(idx_path.read_bytes(), sub_path.read_bytes(), track)


def load_vobsub_bytes(idx_bytes: bytes, sub_bytes: bytes, track: int = 0) -> tuple[Idx, list[Cue]]:
    """Load an .idx/.sub pair from memory (nothing is written anywhere)."""
    idx = parse_idx_text(idx_bytes.decode("latin-1"))
    data = sub_bytes
    trk = idx.tracks[track]
    cues: list[Cue] = []
    for n, (ms, filepos) in enumerate(trk.entries):
        spu = _read_spu_packet(data, filepos, 0x20 + trk.index)
        if not spu:
            continue
        img, colors, alpha, x, y, sdel, edel, forced = decode_spu(spu, idx.palette)
        start = ms + sdel
        end = ms + edel if edel else None
        cues.append(Cue(n, start, end, img, colors, alpha, x, y, forced))
    # Fill missing end times from the next cue start.
    for a, b in zip(cues, cues[1:]):
        if a.end_ms is None or a.end_ms > b.start_ms:
            a.end_ms = b.start_ms if a.end_ms is None else min(a.end_ms, b.start_ms)
    if cues and cues[-1].end_ms is None:
        cues[-1].end_ms = cues[-1].start_ms + 4000
    return idx, cues


def to_rgba(cue: Cue) -> np.ndarray:
    lut = np.array([(*c, a * 17) for c, a in zip(cue.colors, cue.alpha)], dtype=np.uint8)
    return lut[cue.image]
