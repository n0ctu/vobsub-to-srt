"""Async DeepSeek (OpenAI-compatible) vision client with on-disk cache."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import random
import time
import re
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from .prompts import (PROMPT_VERSION, SUBMIT_TOOL, TRANSCRIBE_CONTEXT, TRANSCRIBE_SHEET_USER, TRANSCRIBE_STRICT_ADDENDUM,
                      TRANSCRIBE_SYSTEM, TRANSCRIBE_TOOL_ADDENDUM, TRANSCRIBE_USER)

log = logging.getLogger(__name__)

LANG_NAMES = {"en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian"}


ENV_KEYS = ("VLM_BASE_URL", "VLM_API_KEY", "VLM_MODEL")


def env(key: str) -> str | None:
    return os.environ.get(key) or None


def endpoint_configured() -> bool:
    load_env()
    return all(env(k) for k in ENV_KEYS)


def load_env(path: Path = Path(".env")) -> None:
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def mask_to_png(mask: np.ndarray, scale: int = 2, pad: int = 8) -> bytes:
    """Dark text on white, padded and upscaled (nearest)."""
    img = np.full((mask.shape[0] + 2 * pad, mask.shape[1] + 2 * pad), 255, np.uint8)
    img[pad:pad + mask.shape[0], pad:pad + mask.shape[1]][mask] = 0
    im = Image.fromarray(img)
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def mask_to_png_smooth(mask: np.ndarray, scale: int = 2, pad: int = 8) -> bytes:
    """Like mask_to_png, but upscaled with Lanczos (anti-aliased edges instead of blocks)."""
    img = np.full((mask.shape[0] + 2 * pad, mask.shape[1] + 2 * pad), 255, np.uint8)
    img[pad:pad + mask.shape[0], pad:pad + mask.shape[1]][mask] = 0
    im = Image.fromarray(img).resize((img.shape[1] * scale, img.shape[0] * scale), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def original_png(cue, fill: list[int], scale: int = 1, pad: int = 8, margin: int = 2) -> bytes:
    """The cue as drawn (fill, anti-alias ring, outline), as grey on white: the fill colour becomes
    black, the colour farthest from it in luminance (the outline) white, the ring in between;
    transparent pixels are white. Cropped to the fill's bounding box plus a margin."""
    lum = np.array([0.299 * r + 0.587 * g + 0.114 * b for r, g, b in cue.colors])
    opaque = np.array([a > 0 for a in cue.alpha])
    lf = lum[fill].mean()
    far = max((abs(lum[v] - lf) for v in range(4) if opaque[v]), default=1.0) or 1.0
    grey = np.array([255 if not opaque[v] else int(round(255 * min(1.0, abs(lum[v] - lf) / far)))
                     for v in range(4)], np.uint8)
    ink = np.isin(cue.image, fill)
    rows, cols = np.nonzero(ink.any(axis=1))[0], np.nonzero(ink.any(axis=0))[0]
    h, w = cue.image.shape
    y0, y1 = max(0, rows[0] - margin), min(h, rows[-1] + 1 + margin)
    x0, x1 = max(0, cols[0] - margin), min(w, cols[-1] + 1 + margin)
    img = grey[cue.image[y0:y1, x0:x1]]
    img = np.pad(img, pad, constant_values=255)
    im = Image.fromarray(img)
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


RENDERS = ("mask2", "mask1", "mask2s", "orig1", "orig2")


def render_cue_png(cue, mask: np.ndarray, fill: list[int], render: str) -> bytes:
    """The image a VLM gets for a cue: see RENDERS (mask = fill bitmask, nearest 2x by default)."""
    if render == "mask1":
        return mask_to_png(mask, scale=1)
    if render == "mask2s":
        return mask_to_png_smooth(mask, scale=2)
    if render in ("orig1", "orig2"):
        return original_png(cue, fill, scale=1 if render == "orig1" else 2)
    return mask_to_png(mask, scale=2)


def sheet_png(masks: list[np.ndarray], scale: int = 2, pad: int = 8, bar: int = 4) -> bytes:
    """Several cue masks stacked top to bottom (dark text on white), separated by thick black bars."""
    width = max(m.shape[1] for m in masks) + 2 * pad
    rows: list[np.ndarray] = []
    for k, m in enumerate(masks):
        if k:
            rows.append(np.full((pad, width), 255, np.uint8))
            rows.append(np.zeros((bar, width), np.uint8))
            rows.append(np.full((pad, width), 255, np.uint8))
        img = np.full((m.shape[0] + 2 * pad, width), 255, np.uint8)
        img[pad:pad + m.shape[0], pad:pad + m.shape[1]][m] = 0
        rows.append(img)
    im = Image.fromarray(np.vstack(rows))
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


_NUMBERED = re.compile(r"^\s*(?:\(?\d{1,2}[.):]|\d{1,2}\s*[-\u2013])\s+")


def split_sheet(text: str, n: int) -> list[str] | None:
    """The answer for a sheet: n transcriptions separated by empty lines (a leading number is
    dropped). None when the count does not match (the caller splits by the glyphs instead)."""
    parts = [pt.strip() for pt in re.split(r"\n\s*\n", text.strip()) if pt.strip()]
    parts = [_NUMBERED.sub("", pt) for pt in parts]
    return parts if len(parts) == n else None


def _clean(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```\w*\n?|\n?```$", "", text).strip()
    return "\n".join(l.rstrip() for l in text.splitlines() if l.strip())


def _extract(message: dict) -> str:
    """Transcript from a submit_transcript tool call, falling back to plain content."""
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        if fn.get("name") == "submit_transcript":
            lines = json.loads(fn.get("arguments") or "{}").get("lines")
            if isinstance(lines, list):
                return _clean("\n".join(str(l) for l in lines))
    return _clean(message.get("content") or "")


OCR_USER = "OCR:"
OCR_PROMPT_VERSION = "v5"      # the bare OCR prompt has not changed since PROMPT_VERSION v5


def _ocr_fixups(text: str, n_lines: int) -> str:
    """Document OCR models sometimes answer with the whole block repeated, or with markdown.
    Collapse an answer that is the expected number of lines repeated k times."""
    lines = [l.rstrip() for l in text.strip().split("\n") if l.strip()]
    lines = [l for l in lines if l not in ("```", "```text")]
    if n_lines and len(lines) > n_lines and len(lines) % n_lines == 0:
        block = lines[:n_lines]
        if all(lines[i:i + n_lines] == block for i in range(0, len(lines), n_lines)):
            lines = block
    return "\n".join(lines)


COOLDOWN_MAX = 120.0    # seconds: longest pause after repeated HTTP 429


class AdaptiveLimiter:
    """Concurrency limiter whose limit shrinks on HTTP 429 and slowly grows back on success."""

    def __init__(self, limit: int):
        self.max_limit = limit
        self.limit = limit
        self.active = 0
        self.ok_streak = 0
        self.cond = asyncio.Condition()

    async def acquire(self) -> None:
        async with self.cond:
            await self.cond.wait_for(lambda: self.active < self.limit)
            self.active += 1

    async def release(self, throttled: bool) -> None:
        async with self.cond:
            self.active -= 1
            if throttled:
                self.ok_streak = 0
                if self.limit > 1:
                    self.limit -= 1
                    log.debug("rate limited: concurrency -> %d", self.limit)
            else:
                self.ok_streak += 1
                if self.ok_streak >= 20 and self.limit < self.max_limit:
                    self.limit += 1
                    self.ok_streak = 0
            self.cond.notify_all()


class VLMClient:
    def __init__(self, cache_dir: Path | None = None, concurrency: int = 2,
                 base_url: str | None = None, api_key: str | None = None, model: str | None = None,
                 timeout: float = 60.0, max_attempts: int = 4, use_tool: bool = False,
                 profile: str | None = None, limiter=None):
        load_env()
        missing = [k for k in ENV_KEYS if not env(k)]
        if missing and not (base_url and api_key and model):
            raise SystemExit(f"VLM endpoint not configured: set {', '.join(missing)} in .env "
                             "(see .env.example); any OpenAI-compatible vision model works")
        self.base_url = (base_url or env("VLM_BASE_URL")).rstrip("/")
        self.api_key = api_key or env("VLM_API_KEY")
        self.model = model or env("VLM_MODEL")
        self.use_tool = use_tool
        # Prompt profile. "chat": the instruction prompt tuned for chat VLMs (DeepSeek & co).
        # "ocr": the bare "OCR:" task prompt of document OCR models (PaddleOCR-VL), which choke on
        # instructions, context and line counts. Chosen by VLM_PROMPT, else by the model name.
        self.profile = profile or env("VLM_PROMPT") or ("ocr" if any(
            w in (self.model or "").lower() for w in ("ocr", "paddle")) else "chat")
        # answer cache: in memory for this client's lifetime (default: nothing of the subtitle text
        # touches a disk), or on disk under cache_dir (CLI --diagnostics: re-runs are free and
        # reproducible across processes)
        self.cache_dir = cache_dir
        self._mem: dict[str, str] = {}
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        # request slots: this client's own adaptive limiter, or one shared across worker
        # processes (pool.PipeLimiter) with the same acquire()/release(throttled) interface
        self.limiter = limiter if limiter is not None else AdaptiveLimiter(concurrency)
        # HTTP 429 handling: a cooldown shared by every request of this client. Each throttled
        # answer doubles it (from the server's retry-after, at least 5 s, at most COOLDOWN_MAX);
        # nothing is sent before it has passed; a successful answer ends it. Without this, every
        # in-flight request probed the endpoint again on its own schedule (1763 throttled
        # requests for 127 cues on one file).
        self.max_throttle_retries = 12
        self._cooldown = 0.0
        self._cooldown_until = 0.0
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.calls = 0          # real API requests (incl. throttled ones)
        self.throttled = 0
        self.cache_hits = 0
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *exc):
        await self._client.aclose()

    def cache_get(self, key: str) -> str | None:
        if self.cache_dir is None:
            return self._mem.get(key)
        f = self.cache_dir / f"{key}.json"
        if f.exists():
            return json.loads(f.read_text())["text"]
        return None

    def cache_put(self, key: str, text: str) -> None:
        if self.cache_dir is None:
            self._mem[key] = text
            return
        f = self.cache_dir / f"{key}.json"
        tmp = f.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"text": text}, ensure_ascii=False))
        tmp.replace(f)

    def _key(self, png: bytes, prompt: str) -> str:
        h = hashlib.sha256()
        version = PROMPT_VERSION if self.profile == "chat" else OCR_PROMPT_VERSION
        for part in (self.model, version, prompt):
            h.update(part.encode())
            h.update(b"\0")
        h.update(png)
        return h.hexdigest()

    async def transcribe(self, png: bytes, n_lines: int, lang: str = "en", strict: bool = False,
                         context: list[str] | None = None) -> str:
        text, _ = await self.transcribe_ex(png, n_lines, lang, strict, context)
        return text

    async def transcribe_ex(self, png: bytes, n_lines: int, lang: str = "en", strict: bool = False,
                            context: list[str] | None = None, sheet: int = 0) -> tuple[str, bool]:
        """Like transcribe(); also returns whether the answer came from the cache (no API request).
        `sheet` > 1: the image stacks that many cues (sheet_png); the answer holds all of them."""
        if self.profile == "ocr":
            user, context, system = OCR_USER, None, ""
        else:
            system = TRANSCRIBE_SYSTEM
            if sheet > 1:
                user = TRANSCRIBE_SHEET_USER.format(lang=LANG_NAMES.get(lang, lang), n=sheet)
            else:
                user = TRANSCRIBE_USER.format(lang=LANG_NAMES.get(lang, lang), n=n_lines)
            if strict:
                user += TRANSCRIBE_STRICT_ADDENDUM
            if self.use_tool:
                user += TRANSCRIBE_TOOL_ADDENDUM
        mode = ("\0tool" if self.use_tool else "") + "\0" + self.profile
        # image-level key (without the context history): a re-run whose context differs reuses the
        # earlier answer for the same image instead of asking again -> deterministic, no API call
        image_key = "img-" + self._key(png, system + chr(0) + user + mode)
        if context:
            user += TRANSCRIBE_CONTEXT.format(k=len(context), history="\n".join(context))
        key = self._key(png, system + "\0" + user + mode)
        cached = self.cache_get(key)
        if cached is None:
            cached = self.cache_get(image_key)
        if cached is not None:
            self.cache_hits += 1
            return cached, True
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}
        if self.profile == "ocr":
            messages = [{"role": "user", "content": [image, {"type": "text", "text": user}]}]
        else:
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": [{"type": "text", "text": user}, image]}]
        payload = {"model": self.model, "temperature": 0, "messages": messages}
        if self.profile == "ocr":
            payload["max_tokens"] = 400          # a derailed OCR model floods newlines
        elif sheet > 1:
            payload["max_tokens"] = 300 * sheet
        if self.use_tool:
            payload["tools"] = [SUBMIT_TOOL]
            payload["tool_choice"] = {"type": "function", "function": {"name": "submit_transcript"}}
        last_err: Exception | None = None
        attempt = throttles = 0
        while attempt < self.max_attempts:
            await self._wait_cooldown()
            await self.limiter.acquire()
            throttled = False
            try:
                self.calls += 1
                r = await self._client.post(
                    f"{self.base_url}/chat/completions", json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"})
                if r.status_code == 429:
                    throttled = True
                    self.throttled += 1
                    throttles += 1
                    retry_after = float(r.headers.get("retry-after", "5") or 5)
                    self._throttled_now(retry_after)
                    if throttles > self.max_throttle_retries:
                        raise httpx.HTTPStatusError("HTTP 429 (gave up)", request=r.request, response=r)
                elif r.status_code >= 500:
                    raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
                else:
                    r.raise_for_status()
                    text = _extract(r.json()["choices"][0]["message"])
                    if self.profile == "ocr":
                        text = _ocr_fixups(text, n_lines)
                    if not text:
                        raise ValueError("empty response")
                    self.cache_put(key, text)
                    if self.cache_get(image_key) is None:
                        self.cache_put(image_key, text)
                    self._cooldown = 0.0
                    return text, False
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
                last_err = e
                attempt += 1
                log.debug("VLM attempt %d failed: %s", attempt, e)
            finally:
                await self.limiter.release(throttled)
            if not throttled:
                await asyncio.sleep(min(2 ** attempt, 20) + random.random())
        raise RuntimeError(f"VLM failed after {self.max_attempts} attempts: {last_err}")

    def _throttled_now(self, retry_after: float) -> None:
        self._cooldown = min(COOLDOWN_MAX, max(retry_after, 5.0, 2 * self._cooldown))
        until = time.monotonic() + self._cooldown + random.random()
        if until > self._cooldown_until:
            self._cooldown_until = until
            log.info("rate limited (HTTP 429): pausing requests for %.0f s", self._cooldown)

    async def _wait_cooldown(self) -> None:
        while True:
            left = self._cooldown_until - time.monotonic()
            if left <= 0:
                return
            await asyncio.sleep(left)
