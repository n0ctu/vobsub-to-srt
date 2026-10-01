"""A stand-in for job.run_job, imported by worker processes in the tests (see pool.WorkerPool)."""
import asyncio

from vobsub_to_srt.job import JobResult


async def run_job(source, config, progress, limiter):
    stem = getattr(source, "name", "")
    progress({"event": "probe", "db": "calm-sable-0000", "mode": "exact", "coverage": 1.0})
    if stem.startswith("slow"):
        progress({"event": "cues", "items": [], "resolved": 2, "cues": 5, "round": 1})
        await asyncio.sleep(1.2)
    if stem.startswith("hang"):
        await asyncio.sleep(60)
    if stem.startswith("slots"):                 # three requests at once: how many ran together?
        peak = active = 0

        async def one():
            nonlocal peak, active
            await limiter.acquire()
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.2)
            active -= 1
            await limiter.release(False)
        await asyncio.gather(one(), one(), one())
        progress({"event": "slots", "peak": peak})
    progress({"event": "cues", "items": [{"i": 0, "start": 0, "end": 1000, "text": "Hello", "src": "nocr"}], "resolved": 1})
    progress({"event": "glyphs", "items": [{"key": "k1", "label": "H", "w": 4, "h": 5, "bits": "8A==", "n": 3,
                                            "style": "", "new": False}], "known": 1, "pending": 0})
    progress({"event": "done", "unresolved": 0, "seconds": 0.1})
    return JobResult(srt="1\n00:00:00,000 --> 00:00:01,000\nHello\n",
                     report={"by_source": {"nocr": 4, "vlm": 1}, "cues": 5, "vlm_requests": 3, "seconds": 0.1,
                             "language": "en", "max_vlm_cues": getattr(config, "max_vlm_cues", None)},
                     unresolved=0, vlm_used=3, budget_exhausted=False)
