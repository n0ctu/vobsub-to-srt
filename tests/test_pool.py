import asyncio

import pytest

from vobsub_to_srt.pipeline import VobSubData
from vobsub_to_srt.pool import WorkerPool


def test_pool_runs_jobs_caps_slots_and_replaces_a_hung_worker():
    async def main():
        pool = WorkerPool(size=2, slots=4, per_job=2, entry="fake_entry:run_job")
        await pool.start()
        events = {"a": [], "b": []}
        try:
            ra, rb = await asyncio.gather(
                pool.run(VobSubData("slots-a", b"", b""), None, events["a"].append, timeout=30),
                pool.run(VobSubData("slots-b", b"", b""), None, events["b"].append, timeout=30))
            assert ra.srt and rb.srt
            peaks = [e["peak"] for k in events for e in events[k] if e["event"] == "slots"]
            assert peaks == [2, 2]                    # 3 requests per job, never more than 2 at once
            assert [e["event"] for e in events["a"]][-1] == "done"
            with pytest.raises(asyncio.TimeoutError):
                await pool.run(VobSubData("hang", b"", b""), None, lambda e: None, timeout=0.5)
            assert len(pool.workers) == 2             # the hung worker was replaced
            r = await pool.run(VobSubData("x", b"", b""), None, lambda e: None, timeout=30)
            assert r.srt
        finally:
            await pool.stop()
    asyncio.run(main())
