"""The VLM batch: concurrent requests, answers applied in batch order, progress callbacks."""
import asyncio
from types import SimpleNamespace

from vobsub_to_srt.pipeline import _vlm_batch


def _cue(i):
    return SimpleNamespace(cue=SimpleNamespace(index=i, start_ms=i * 1000))


def test_batch_applies_in_order_and_reports_progress():
    batch = [_cue(i) for i in range(4)]
    delays = {0: 0.06, 1: 0.01, 2: 0.03, 3: 0.02}          # the first answer arrives last

    async def call(st):
        await asyncio.sleep(delays[st.cue.index])
        if st.cue.index == 2:
            raise RuntimeError("boom")
        return f"text {st.cue.index}"

    applied, progress, applies = [], [], []
    failures = {2: "old", 3: "old"}
    asyncio.run(_vlm_batch(batch, call, lambda st, r, src: applied.append((st.cue.index, r, src)), failures, "vlm",
                           on_progress=lambda k, n: progress.append((k, n)), on_apply=lambda: applies.append(1)))
    assert applied == [(0, "text 0", "vlm"), (1, "text 1", "vlm"), (3, "text 3", "vlm")]   # batch order, not arrival order
    assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]                                     # one per answered request
    assert len(applies) == 3
    assert list(failures) == [2] and failures[2].startswith("RuntimeError")                 # 3 cleared, 2 recorded


def test_batch_without_callbacks():
    batch = [_cue(0)]

    async def call(st):
        return "x"

    out = []
    asyncio.run(_vlm_batch(batch, call, lambda st, r, src: out.append(r), {}, "vlm"))
    assert out == ["x"]
