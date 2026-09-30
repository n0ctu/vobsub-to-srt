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
import re
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from .prompts import (PROMPT_VERSION, SUBMIT_TOOL, TRANSCRIBE_CONTEXT, TRANSCRIBE_STRICT_ADDENDUM,
                      TRANSCRIBE_SYSTEM, TRANSCRIBE_TOOL_ADDENDUM, TRANSCRIBE_USER)

log = logging.getLogger(__name__)

LANG_NAMES = {"en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian"}


ENV_KEYS = ("DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL")


def endpoint_configured() -> bool:
    load_env()
    return all(os.environ.get(k) for k in ENV_KEYS)


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
    def __init__(self, cache_dir: Path = Path("cache/vlm"), concurrency: int = 2,
                 base_url: str | None = None, api_key: str | None = None, model: str | None = None,
                 timeout: float = 60.0, max_attempts: int = 4, use_tool: bool = False):
        load_env()
        missing = [k for k in ENV_KEYS if not os.environ.get(k)]
        if missing and not (base_url and api_key and model):
            raise SystemExit(f"VLM endpoint not configured: set {', '.join(missing)} in .env "
                             "(see .env.example); any OpenAI-compatible vision model works")
        self.base_url = (base_url or os.environ["DEEPSEEK_BASE_URL"]).rstrip("/")
        self.api_key = api_key or os.environ["DEEPSEEK_API_KEY"]
        self.model = model or os.environ["DEEPSEEK_MODEL"]
        self.use_tool = use_tool
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.limiter = AdaptiveLimiter(concurrency)
        self.max_throttle_retries = 60
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

    def _key(self, png: bytes, prompt: str) -> str:
        h = hashlib.sha256()
        for part in (self.model, PROMPT_VERSION, prompt):
            h.update(part.encode())
            h.update(b"\0")
        h.update(png)
        return h.hexdigest()

    async def transcribe(self, png: bytes, n_lines: int, lang: str = "en", strict: bool = False,
                         context: list[str] | None = None) -> str:
        text, _ = await self.transcribe_ex(png, n_lines, lang, strict, context)
        return text

    async def transcribe_ex(self, png: bytes, n_lines: int, lang: str = "en", strict: bool = False,
                            context: list[str] | None = None) -> tuple[str, bool]:
        """Like transcribe(); also returns whether the answer came from the cache (no API request)."""
        user = TRANSCRIBE_USER.format(lang=LANG_NAMES.get(lang, lang), n=n_lines)
        if strict:
            user += TRANSCRIBE_STRICT_ADDENDUM
        if self.use_tool:
            user += TRANSCRIBE_TOOL_ADDENDUM
        mode = "\0tool" if self.use_tool else ""
        # image-level key (without the context history): a re-run whose context differs reuses the
        # earlier answer for the same image instead of asking again -> deterministic, no API call
        image_file = self.cache_dir / f"img-{self._key(png, TRANSCRIBE_SYSTEM + chr(0) + user + mode)}.json"
        if context:
            user += TRANSCRIBE_CONTEXT.format(k=len(context), history="\n".join(context))
        key = self._key(png, TRANSCRIBE_SYSTEM + "\0" + user + mode)
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.exists() and image_file.exists():
            cache_file = image_file
        if cache_file.exists():
            self.cache_hits += 1
            return json.loads(cache_file.read_text())["text"], True
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": TRANSCRIBE_SYSTEM}, {"role": "user", "content": [
                {"type": "text", "text": user},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
            ]}],
        }
        if self.use_tool:
            payload["tools"] = [SUBMIT_TOOL]
            payload["tool_choice"] = {"type": "function", "function": {"name": "submit_transcript"}}
        last_err: Exception | None = None
        attempt = throttles = 0
        while attempt < self.max_attempts:
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
                    if throttles > self.max_throttle_retries:
                        raise httpx.HTTPStatusError("HTTP 429 (gave up)", request=r.request, response=r)
                elif r.status_code >= 500:
                    raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
                else:
                    r.raise_for_status()
                    text = _extract(r.json()["choices"][0]["message"])
                    if not text:
                        raise ValueError("empty response")
                    tmp = cache_file.with_suffix(f".{os.getpid()}.tmp")
                    tmp.write_text(json.dumps({"text": text}, ensure_ascii=False))
                    tmp.replace(cache_file)
                    if not image_file.exists():
                        image_file.write_text(json.dumps({"text": text}, ensure_ascii=False))
                    return text, False
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
                last_err = e
                attempt += 1
                log.debug("VLM attempt %d failed: %s", attempt, e)
            finally:
                await self.limiter.release(throttled)
            if throttled:
                await asyncio.sleep(retry_after + random.random())
            else:
                await asyncio.sleep(min(2 ** attempt, 20) + random.random())
        raise RuntimeError(f"VLM failed after {self.max_attempts} attempts: {last_err}")
