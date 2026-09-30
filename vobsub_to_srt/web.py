"""Minimal web app: drop a VobSub, get an SRT.

One process, no database, and nothing of a user's subtitles ever touches a disk: uploads are held
in memory until their job ran, results are held in memory until fetched or VTS_JOB_TTL seconds old.
A single worker converts jobs one after another (it is the only writer of the shared glyph memory and
the global VLM throttle). Per-IP limits: jobs per hour and VLM requests per day; when a client's daily
VLM allowance is used up its jobs still run, teacher-less. A queue cap bounds memory use.

Environment: VTS_DATA (default "."; glyph memory, word memory, dictionaries), VTS_JOB_TTL=3600,
VTS_MAX_VLM_CUES=300, VTS_JOBS_PER_HOUR=6, VTS_VLM_PER_DAY=600, VTS_MAX_SUB_MB=64, VTS_MAX_CUES=3000,
VTS_MAX_QUEUE=20, VTS_TRUST_PROXY=0, VTS_HOST, VTS_PORT.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from . import job as jobmod
from .job import JobConfig
from .pipeline import VobSubData
from .names import random_db_name  # noqa: F401  (re-exported for the stats page)

log = logging.getLogger("vobsub_to_srt.web")


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


DATA = Path(os.environ.get("VTS_DATA", "."))
JOB_TTL = _env_int("VTS_JOB_TTL", 3600)
MAX_VLM_CUES = _env_int("VTS_MAX_VLM_CUES", 300)
JOBS_PER_HOUR = _env_int("VTS_JOBS_PER_HOUR", 6)
VLM_PER_DAY = _env_int("VTS_VLM_PER_DAY", 600)
MAX_SUB_BYTES = _env_int("VTS_MAX_SUB_MB", 64) * 1024 * 1024
MAX_IDX_BYTES = 2 * 1024 * 1024
MAX_CUES = _env_int("VTS_MAX_CUES", 3000)
TRUST_PROXY = os.environ.get("VTS_TRUST_PROXY", "0") == "1"
JOB_TIMEOUT = _env_int("VTS_JOB_TIMEOUT", 900)
MAX_QUEUE = _env_int("VTS_MAX_QUEUE", 20)

STATIC = Path(__file__).parent / "static"
_STEM = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class Job:
    id: str
    stem: str
    ip: str
    upload: VobSubData | None = None  # held in memory until the job ran, then dropped
    created: float = field(default_factory=time.time)
    status: str = "queued"           # queued | running | done | error
    events: list[dict] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    result: jobmod.JobResult | None = None
    error: str | None = None
    vlm_used: int = 0

    def push(self, event: dict) -> None:
        event = {k: v for k, v in event.items() if k not in ("srt", "db")}   # no server paths / DB names
        self.events.append(event)
        for q in list(self.subscribers):
            q.put_nowait(event)


class Limiter:
    """Per-IP: sliding hour window on job submissions, daily counter on VLM requests."""

    def __init__(self) -> None:
        self.jobs: dict[str, deque] = defaultdict(deque)
        self.vlm: dict[str, tuple[str, int]] = {}

    def check_job(self, ip: str) -> int | None:
        """None if allowed, else seconds until the next slot."""
        now = time.time()
        d = self.jobs[ip]
        while d and d[0] < now - 3600:
            d.popleft()
        if len(d) >= JOBS_PER_HOUR:
            return int(d[0] + 3600 - now) + 1
        d.append(now)
        return None

    def vlm_left(self, ip: str) -> int:
        day = time.strftime("%Y-%m-%d")
        d, used = self.vlm.get(ip, (day, 0))
        if d != day:
            used = 0
        return max(0, VLM_PER_DAY - used)

    def add_vlm(self, ip: str, n: int) -> None:
        day = time.strftime("%Y-%m-%d")
        d, used = self.vlm.get(ip, (day, 0))
        self.vlm[ip] = (day, (used if d == day else 0) + n)


from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    global queue
    queue = asyncio.Queue()          # bound to this process's event loop
    jobs.clear()
    tasks = [asyncio.create_task(worker()), asyncio.create_task(sweeper())]
    yield
    for t in tasks:
        t.cancel()


app = FastAPI(title="vobsub-to-srt", docs_url=None, redoc_url=None, lifespan=lifespan)
jobs: dict[str, Job] = {}
queue: asyncio.Queue = None  # type: ignore[assignment]  # created in lifespan()
limiter = Limiter()


def client_ip(request: Request) -> str:
    if TRUST_PROXY:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else "?"


def count_cues(idx_bytes: bytes) -> int:
    return sum(1 for line in idx_bytes.splitlines() if line.startswith(b"timestamp:"))


async def _read_limited(up: UploadFile, limit: int, what: str) -> bytes:
    data = await up.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"{what} larger than {limit // (1024 * 1024)} MB")
    return data


@app.post("/api/jobs")
async def create_job(request: Request, idx: UploadFile, sub: UploadFile):
    ip = client_ip(request)
    if not (idx.filename or "").lower().endswith(".idx") or not (sub.filename or "").lower().endswith(".sub"):
        raise HTTPException(400, "upload the .idx and the matching .sub file")
    idx_bytes = await _read_limited(idx, MAX_IDX_BYTES, ".idx")
    sub_bytes = await _read_limited(sub, MAX_SUB_BYTES, ".sub")
    n = count_cues(idx_bytes)
    if n == 0:
        raise HTTPException(400, "no cues found in the .idx file")
    if n > MAX_CUES:
        raise HTTPException(413, f"{n} cues; the limit is {MAX_CUES}")
    if queue.qsize() >= MAX_QUEUE:
        raise HTTPException(503, "the queue is full; try again in a few minutes", headers={"Retry-After": "120"})
    wait = limiter.check_job(ip)
    if wait is not None:
        raise HTTPException(429, f"job limit reached; try again in {wait} s", headers={"Retry-After": str(wait)})
    stem = _STEM.sub("_", Path(idx.filename or "subtitle").stem)[:80] or "subtitle"
    job_id = uuid.uuid4().hex[:12] + secrets.token_hex(2)
    job = Job(job_id, stem, ip, upload=VobSubData(stem, idx_bytes, sub_bytes))
    jobs[job_id] = job
    job.push({"event": "queued", "position": queue.qsize() + 1, "cues": n})
    await queue.put(job_id)
    return {"id": job_id, "cues": n, "position": queue.qsize()}


def _public(job: Job) -> dict:
    return {"id": job.id, "status": job.status, "stem": job.stem, "error": job.error,
            "unresolved": job.result.unresolved if job.result else None,
            "vlm_used": job.vlm_used, "created": job.created}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown or expired job")
    return _public(job)


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown or expired job")

    async def stream():
        q: asyncio.Queue = asyncio.Queue()
        for e in job.events:
            yield f"data: {json.dumps(e)}\n\n"
        if job.status in ("done", "error"):
            return
        job.subscribers.append(q)
        try:
            while True:
                try:
                    e = await asyncio.wait_for(q.get(), timeout=20)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(e)}\n\n"
                if e.get("event") in ("done", "error"):
                    return
        finally:
            if q in job.subscribers:
                job.subscribers.remove(q)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/jobs/{job_id}/srt")
async def job_srt(job_id: str):
    job = jobs.get(job_id)
    if not job or job.status != "done":
        raise HTTPException(404, "no result for this job")
    return Response(job.result.srt, media_type="text/plain; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{job.stem}.srt"'})


@app.get("/api/jobs/{job_id}/report")
async def job_report(job_id: str):
    job = jobs.get(job_id)
    if not job or job.status != "done":
        raise HTTPException(404, "no result for this job")
    return JSONResponse(job.result.report)


@app.get("/api/stats")
async def stats():
    gm = DATA / "glyph-memory"
    return {"fonts": len(list(gm.glob("*.json"))) if gm.is_dir() else 0, "queued": queue.qsize(),
            "running": sum(1 for j in jobs.values() if j.status == "running"),
            "limits": {"jobs_per_hour": JOBS_PER_HOUR, "vlm_per_day": VLM_PER_DAY, "max_vlm_cues": MAX_VLM_CUES,
                       "max_cues": MAX_CUES, "job_ttl": JOB_TTL}}


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


_STATIC_TYPES = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml"}


@app.get("/static/{name}")
async def static_file(name: str):
    path = STATIC / name
    if "/" in name or "\\" in name or path.suffix not in _STATIC_TYPES or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type=_STATIC_TYPES[path.suffix], headers={"Cache-Control": "public, max-age=3600"})


# ---------------------------------------------------------------- worker & housekeeping

async def _run(job: Job) -> None:
    loop = asyncio.get_running_loop()
    job.status = "running"
    job.push({"event": "started"})
    budget = min(MAX_VLM_CUES, limiter.vlm_left(job.ip))
    config = JobConfig(glyph_memory_dir=DATA / "glyph-memory", word_memory_dir=DATA / "word-memory",
                       dict_dir=DATA / "dictionaries", max_vlm_cues=budget)

    def progress(e: dict) -> None:            # called from the worker thread
        loop.call_soon_threadsafe(job.push, e)

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(jobmod.run_job_sync, job.upload, config, progress), JOB_TIMEOUT)
        job.result = result
        job.vlm_used = result.vlm_used
        limiter.add_vlm(job.ip, result.vlm_used)
        job.status = "done"
        if job.events[-1].get("event") != "done":
            job.push({"event": "done", "unresolved": result.unresolved})
    except Exception as e:  # noqa: BLE001
        log.exception("job %s failed", job.id)
        job.status = "error"
        job.error = f"{type(e).__name__}: {e}"[:300]
        job.push({"event": "error", "message": job.error})
    finally:
        job.upload = None                                   # uploads never outlive the job


async def worker() -> None:
    while True:
        job_id = await queue.get()
        job = jobs.get(job_id)
        if job:
            await _run(job)
        queue.task_done()


async def sweeper() -> None:
    while True:
        await asyncio.sleep(60)
        cutoff = time.time() - JOB_TTL
        for job_id, job in list(jobs.items()):
            if job.created < cutoff and job.status in ("done", "error"):
                del jobs[job_id]


def main() -> None:
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    uvicorn.run(app, host=os.environ.get("VTS_HOST", "0.0.0.0"), port=_env_int("VTS_PORT", 8000),
                proxy_headers=TRUST_PROXY, log_level="info")


if __name__ == "__main__":
    main()
