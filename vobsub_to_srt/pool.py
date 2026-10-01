"""Worker processes for the web app.

The web process stays the only owner of the queue, the job records and the event streams; it
hands each job to one of a few worker processes (real CPU parallelism: segmentation and
alignment are CPU-bound Python) and streams the job's progress events back over the worker's
queue. The same channel carries the vision-model request slots: a worker asks for a slot before
every request and returns it afterwards, so the orchestrator enforces one global ceiling and a
per-job cap however many jobs run (the endpoint rate-limits at ~7 concurrent requests). A job
that times out is killed with its process and the process is replaced; nothing keeps running.

Workers learn into the shared glyph memory concurrently; GlyphDB.save() merges under a file lock.
"""
from __future__ import annotations

import asyncio
import importlib
import logging
import multiprocessing as mp
import threading
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger(__name__)


def _resolve(entry: str) -> Callable:
    mod, _, fn = entry.partition(":")
    return getattr(importlib.import_module(mod), fn)


class PipeLimiter:
    """Request slots granted by the orchestrator (same interface as vlm.AdaptiveLimiter)."""

    def __init__(self, outq, loop: asyncio.AbstractEventLoop):
        self.outq, self.loop = outq, loop
        self.grants: asyncio.Queue = asyncio.Queue()

    async def acquire(self) -> None:
        self.outq.put(("acquire",))
        await self.grants.get()

    async def release(self, throttled: bool) -> None:
        self.outq.put(("release", bool(throttled)))

    def granted(self) -> None:                      # called from the worker's main thread
        self.loop.call_soon_threadsafe(self.grants.put_nowait, None)


def _worker_main(inq, outq, entry: str) -> None:
    """Worker process: one job at a time. The main thread reads the inbound queue (jobs, grants);
    the job runs in its own thread with its own event loop and reports through the outbound queue.
    Queues (not raw pipes) carry the multi-megabyte uploads and results without blocking anyone."""
    fn = _resolve(entry)
    current: list[PipeLimiter] = []

    def run(payload: dict) -> None:
        async def go() -> None:
            limiter = PipeLimiter(outq, asyncio.get_running_loop())
            current.append(limiter)
            try:
                res = await fn(payload["source"], payload["config"], lambda e: outq.put(("event", e)), limiter)
                outq.put(("done", res))
            except BaseException as e:  # noqa: BLE001 - reported to the orchestrator
                outq.put(("error", f"{type(e).__name__}: {e}"[:300]))
            finally:
                current.clear()
        asyncio.run(go())

    while True:
        try:
            msg = inq.get()
        except (EOFError, OSError):
            return
        if msg[0] == "job":
            threading.Thread(target=run, args=(msg[1],), daemon=True).start()
        elif msg[0] == "grant" and current:
            current[0].granted()
        elif msg[0] == "stop":
            return


@dataclass
class _Running:
    fut: asyncio.Future
    progress: Callable[[dict], None]


class _Worker:
    def __init__(self, proc, inq, outq):
        self.proc, self.inq, self.outq = proc, inq, outq
        self.job: _Running | None = None
        self.pending = 0        # slot requests not yet granted
        self.active = 0         # slots granted


class WorkerPool:
    """`size` worker processes; `slots` vision requests in flight overall, at most `per_job` per job.
    The ceiling shrinks on HTTP 429 and grows back on sustained success (like AdaptiveLimiter)."""

    def __init__(self, size: int = 2, slots: int = 4, per_job: int = 2,
                 entry: str = "vobsub_to_srt.job:run_job"):
        self.size, self.slots, self.per_job, self.entry = size, slots, per_job, entry
        self.limit = slots
        self.ok_streak = 0
        self.throttled = 0
        self.workers: list[_Worker] = []
        self.ctx = mp.get_context("spawn")
        self.loop: asyncio.AbstractEventLoop | None = None
        self.idle: asyncio.Queue | None = None

    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.idle = asyncio.Queue()
        for _ in range(self.size):
            self.idle.put_nowait(self._spawn())

    async def stop(self) -> None:
        for w in list(self.workers):
            self._drop(w, "pool stopped", respawn=False)
        for w in list(self.workers):
            w.proc.join(2)

    def _spawn(self) -> _Worker:
        inq, outq = self.ctx.Queue(), self.ctx.Queue()
        proc = self.ctx.Process(target=_worker_main, args=(inq, outq, self.entry), daemon=True)
        proc.start()
        w = _Worker(proc, inq, outq)
        threading.Thread(target=self._drain, args=(w,), daemon=True).start()
        self.loop.add_reader(proc.sentinel, self._exited, w)     # readable once the process ended
        self.workers.append(w)
        return w

    def _drain(self, w: _Worker) -> None:
        """Thread: forward the worker's messages to the event loop."""
        while True:
            try:
                msg = w.outq.get()
            except (EOFError, OSError):
                return
            if msg[0] == "__exit__":
                return
            self.loop.call_soon_threadsafe(self._handle, w, msg)

    def in_flight(self) -> int:
        return sum(w.active for w in self.workers)

    # ---- orchestrator side
    def _exited(self, w: _Worker) -> None:
        self._drop(w, "worker process died")

    def _handle(self, w: _Worker, msg: tuple) -> None:
        kind = msg[0]
        if kind == "event":
            if w.job:
                w.job.progress(msg[1])
        elif kind == "acquire":
            w.pending += 1
            self._grant()
        elif kind == "release":
            w.active = max(0, w.active - 1)
            self._feedback(bool(msg[1]))
            self._grant()
        elif kind in ("done", "error"):
            job, w.job = w.job, None
            w.pending = w.active = 0
            self._grant()
            if job and not job.fut.done():
                if kind == "done":
                    job.fut.set_result(msg[1])
                else:
                    job.fut.set_exception(RuntimeError(msg[1]))
            if w in self.workers:
                self.idle.put_nowait(w)

    def _feedback(self, throttled: bool) -> None:
        if throttled:
            self.throttled += 1
            self.ok_streak = 0
            if self.limit > 1:
                self.limit -= 1
                log.info("vision endpoint rate limited: slots -> %d", self.limit)
        else:
            self.ok_streak += 1
            if self.ok_streak >= 20 and self.limit < self.slots:
                self.limit += 1
                self.ok_streak = 0

    def _grant(self) -> None:
        """Hand out free slots round-robin to workers that asked, within the per-job cap."""
        progress = True
        while progress and self.in_flight() < self.limit:
            progress = False
            for w in self.workers:
                if w.pending and w.active < self.per_job and self.in_flight() < self.limit:
                    w.pending -= 1
                    w.active += 1
                    w.inq.put(("grant",))
                    progress = True

    def _drop(self, w: _Worker, why: str, respawn: bool = True) -> None:
        if w not in self.workers:
            return
        self.workers.remove(w)
        try:
            self.loop.remove_reader(w.proc.sentinel)
        except (OSError, ValueError):
            pass
        if w.proc.is_alive():
            w.proc.terminate()
        try:
            w.outq.put(("__exit__",))                 # ends the drain thread
        except (OSError, ValueError):
            pass
        if w.job and not w.job.fut.done():
            w.job.fut.set_exception(RuntimeError(why))
        w.job = None
        for q in (w.inq, w.outq):
            try:
                q.close()
            except (OSError, ValueError):
                pass
        if respawn:
            self.idle.put_nowait(self._spawn())

    # ---- jobs
    async def run(self, source: Any, config: Any, progress: Callable[[dict], None],
                  timeout: float | None = None) -> Any:
        """Run one job on a free worker; progress events arrive on the event loop thread.
        On timeout the worker process is killed and replaced, and TimeoutError is raised."""
        w = await self.idle.get()
        fut = self.loop.create_future()
        w.job = _Running(fut, progress)
        w.inq.put(("job", {"source": source, "config": config}))
        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout)
        except asyncio.TimeoutError:
            w.job = None
            self._drop(w, "timed out")
            raise
