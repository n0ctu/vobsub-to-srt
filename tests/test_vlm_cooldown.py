"""HTTP 429 handling: one cooldown shared by all requests of a client, doubling per throttle."""
import asyncio

import httpx
import pytest

from vobsub_to_srt import vlm


def test_429_cooldown_is_shared_and_doubles(monkeypatch):
    clock = {"t": 1000.0}
    sleeps: list[float] = []

    async def fake_sleep(s):
        sleeps.append(round(s, 1))
        clock["t"] += s

    monkeypatch.setattr(vlm.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(vlm.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(vlm.random, "random", lambda: 0.0)
    answers = iter([429, 429, 200, 200])

    def handler(request):
        code = next(answers)
        if code == 429:
            return httpx.Response(429, headers={"retry-after": "7"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "Hallo"}}]})

    async def go():
        c = vlm.VLMClient(base_url="http://x", api_key="k", model="m", concurrency=2, profile="chat")
        c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        a, b = await asyncio.gather(c.transcribe_ex(b"png-a", 1), c.transcribe_ex(b"png-b", 1))
        return c, a, b

    c, a, b = asyncio.run(go())
    assert a[0] == "Hallo" and b[0] == "Hallo"
    assert c.throttled == 2 and c.calls == 4
    # first throttle: 7 s (retry-after), second: doubled to 14 s; both requests waited the pause
    assert c._cooldown == 0.0          # the successes ended it
    assert sleeps and max(sleeps) >= 14.0 and all(s >= 7.0 for s in sleeps)
