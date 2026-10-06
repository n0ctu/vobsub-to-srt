"""Minimal web app: drop a VobSub, get an SRT.

One process, no database, and nothing of a user's subtitles ever touches a disk: uploads are held
in memory until their job ran, results are held in memory until VTS_JOB_TTL (10 min) old.
A few worker processes convert jobs concurrently (pool.WorkerPool); the orchestrator hands out the
vision-model request slots (global ceiling, per-job cap) and the glyph memory merges concurrent
learning on save. Per-IP limits: jobs per hour and VLM requests per day; when a client's daily
VLM allowance is used up its jobs still run, teacher-less. A queue cap bounds memory use.

Environment: VTS_DATA (default "."; glyph memory, word memory, dictionaries), VTS_JOB_TTL=600,
VTS_MAX_VLM_CUES=0 (no per-job cap), VTS_JOBS_PER_HOUR=20, VTS_VLM_PER_DAY=1000, VTS_MAX_SUB_MB=64, VTS_MAX_CUES=6000,
VTS_MAX_QUEUE=20, VTS_TRUST_PROXY=0, VTS_HOST, VTS_PORT, VTS_BASELINE_DIR (fonts shipped with the image).
Limits and aggregate usage counters live in <data>/stats.sqlite (see Store: addresses only as daily salted hashes).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import hashlib
import secrets
import sqlite3
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from . import job as jobmod
from .job import JobConfig
from .version import app_version
from .pipeline import VobSubData
from .pool import WorkerPool
from .names import random_db_name  # noqa: F401  (re-exported for the stats page)

log = logging.getLogger("vobsub_to_srt.web")


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


DATA = Path(os.environ.get("VTS_DATA", "."))
JOB_TTL = _env_int("VTS_JOB_TTL", 600)
MAX_VLM_CUES = _env_int("VTS_MAX_VLM_CUES", 0) or None   # 0 = no per-job cap, only the daily allowance
JOBS_PER_HOUR = _env_int("VTS_JOBS_PER_HOUR", 20)
VLM_PER_DAY = _env_int("VTS_VLM_PER_DAY", 1000)
MAX_SUB_BYTES = _env_int("VTS_MAX_SUB_MB", 64) * 1024 * 1024
MAX_IDX_BYTES = 2 * 1024 * 1024
MAX_CUES = _env_int("VTS_MAX_CUES", 6000)
TRUST_PROXY = os.environ.get("VTS_TRUST_PROXY", "0") == "1"
JOB_TIMEOUT = _env_int("VTS_JOB_TIMEOUT", 900)
MAX_QUEUE = _env_int("VTS_MAX_QUEUE", 20)
WORKERS = _env_int("VTS_WORKERS", 2)                  # worker processes (jobs converted at once)
VLM_SLOTS = _env_int("VTS_VLM_SLOTS", 4)              # vision requests in flight, all jobs together
VLM_SLOTS_PER_JOB = _env_int("VTS_VLM_SLOTS_PER_JOB", 2)
POOL_ENTRY = "vobsub_to_srt.job:run_job"              # what a worker runs (tests substitute a stand-in)
BASELINE_DIR = Path(os.environ.get("VTS_BASELINE_DIR", "/app/glyph-memory"))   # fonts shipped with the image

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
    snap: dict = field(default_factory=dict)   # latest progress (cues, resolved, round) for the queue view

    def push(self, event: dict) -> None:
        event = {k: v for k, v in event.items() if k not in ("srt", "db")}   # no server paths / DB names
        self.events.append(event)
        for q in list(self.subscribers):
            q.put_nowait(event)


class Store:
    """Limits and usage statistics in one SQLite file (<data>/stats.sqlite), so they survive restarts
    and rebuilds. Clients are identified only by a salted hash of their address; the salt is random
    per day, and a day's hashes are folded into a plain count once the day is over."""

    def __init__(self, path: Path) -> None:
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS counters   (key TEXT PRIMARY KEY, value REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS languages  (lang TEXT PRIMARY KEY, jobs INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS days       (day TEXT PRIMARY KEY, salt BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS user_hashes(day TEXT NOT NULL, hash TEXT NOT NULL, PRIMARY KEY (day, hash));
            CREATE TABLE IF NOT EXISTS daily_users(day TEXT PRIMARY KEY, users INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS submissions(hash TEXT NOT NULL, ts REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS vlm_usage  (day TEXT NOT NULL, hash TEXT NOT NULL, used INTEGER NOT NULL,
                                                   PRIMARY KEY (day, hash));
        """)
        self.db.execute("INSERT OR IGNORE INTO counters VALUES ('since', ?)", (time.time(),))
        self._import_legacy(path.with_name("stats.json"))
        self._roll()

    def _import_legacy(self, legacy: Path) -> None:
        """One-time import of the earlier stats.json counters."""
        if not legacy.exists() or self.db.execute("SELECT COUNT(*) FROM counters").fetchone()[0] > 1:
            return
        try:
            d = json.loads(legacy.read_text())
            for k in ("jobs", "cues", "memory_cues", "vision_cues", "unresolved_cues", "vlm_requests", "seconds", "errors"):
                self.db.execute("INSERT OR REPLACE INTO counters VALUES (?, ?)", (k, float(d.get(k, 0))))
            self.db.execute("INSERT OR REPLACE INTO counters VALUES ('since', ?)", (float(d.get("since", time.time())),))
            if d.get("users"):
                self.db.execute("INSERT OR REPLACE INTO daily_users VALUES ('legacy', ?)", (int(d["users"]),))
            for lang, n in d.get("languages", {}).items():
                self.db.execute("INSERT OR REPLACE INTO languages VALUES (?, ?)", (lang, int(n)))
            legacy.rename(legacy.with_suffix(".json.imported"))
        except (OSError, ValueError):
            pass

    # ---- day handling & hashing ----
    def _roll(self) -> tuple[str, bytes]:
        today = time.strftime("%Y-%m-%d")
        row = self.db.execute("SELECT salt FROM days WHERE day = ?", (today,)).fetchone()
        if row is None:
            self.db.execute("INSERT OR REPLACE INTO daily_users SELECT day, COUNT(*) FROM user_hashes "
                            "WHERE day < ? GROUP BY day", (today,))
            self.db.execute("DELETE FROM user_hashes WHERE day < ?", (today,))
            self.db.execute("DELETE FROM vlm_usage WHERE day < ?", (today,))
            self.db.execute("DELETE FROM days WHERE day < ?", (today,))
            self.db.execute("DELETE FROM submissions WHERE ts < ?", (time.time() - 3600,))
            salt = secrets.token_bytes(16)
            self.db.execute("INSERT INTO days VALUES (?, ?)", (today, salt))
            return today, salt
        return today, row[0]

    def _hash(self, ip: str) -> tuple[str, str]:
        day, salt = self._roll()
        return day, hashlib.sha256(salt + ip.encode()).hexdigest()[:24]

    # ---- limits ----
    def check_job(self, ip: str) -> int | None:
        """None if allowed (and recorded), else seconds until the next slot."""
        _, h = self._hash(ip)
        now = time.time()
        self.db.execute("DELETE FROM submissions WHERE ts < ?", (now - 3600,))
        rows = self.db.execute("SELECT ts FROM submissions WHERE hash = ? ORDER BY ts", (h,)).fetchall()
        if len(rows) >= JOBS_PER_HOUR:
            return int(rows[0][0] + 3600 - now) + 1
        self.db.execute("INSERT INTO submissions VALUES (?, ?)", (h, now))
        return None

    def vlm_left(self, ip: str) -> int:
        day, h = self._hash(ip)
        row = self.db.execute("SELECT used FROM vlm_usage WHERE day = ? AND hash = ?", (day, h)).fetchone()
        return max(0, VLM_PER_DAY - (row[0] if row else 0))

    def add_vlm(self, ip: str, n: int) -> None:
        day, h = self._hash(ip)
        self.db.execute("INSERT INTO vlm_usage VALUES (?, ?, ?) ON CONFLICT(day, hash) DO UPDATE SET used = used + ?",
                        (day, h, n, n))

    # ---- statistics ----
    def touch_user(self, ip: str) -> None:
        day, h = self._hash(ip)
        self.db.execute("INSERT OR IGNORE INTO user_hashes VALUES (?, ?)", (day, h))

    def _inc(self, key: str, n: float) -> None:
        self.db.execute("INSERT INTO counters VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = value + ?", (key, n, n))

    def add_job(self, report: dict) -> None:
        by = report.get("by_source", {})
        self._inc("jobs", 1)
        self._inc("cues", report.get("cues", 0))
        self._inc("memory_cues", by.get("nocr", 0) + by.get("nocr-arbitrated", 0))
        self._inc("vision_cues", sum(v for k, v in by.items() if k.startswith("vlm")))
        self._inc("unresolved_cues", by.get("fallback", 0))
        self._inc("vlm_requests", report.get("vlm_requests", 0))
        self._inc("seconds", report.get("seconds", 0))
        lang = report.get("language") or "?"
        self.db.execute("INSERT INTO languages VALUES (?, 1) ON CONFLICT(lang) DO UPDATE SET jobs = jobs + 1", (lang,))

    def add_error(self) -> None:
        self._inc("errors", 1)

    def snapshot(self) -> dict:
        today, _ = self._roll()
        d = {k: 0 for k in ("jobs", "cues", "memory_cues", "vision_cues", "unresolved_cues", "vlm_requests",
                            "seconds", "errors")}
        d.update({k: v for k, v in self.db.execute("SELECT key, value FROM counters")})
        for k in ("jobs", "cues", "memory_cues", "vision_cues", "unresolved_cues", "vlm_requests", "errors"):
            d[k] = int(d[k])
        d["languages"] = dict(self.db.execute("SELECT lang, jobs FROM languages"))
        past = self.db.execute("SELECT COALESCE(SUM(users), 0) FROM daily_users").fetchone()[0]
        today_n = self.db.execute("SELECT COUNT(*) FROM user_hashes WHERE day = ?", (today,)).fetchone()[0]
        d["users"] = int(past + today_n)
        return d


_glyph_cache: dict = {"key": None, "value": None}


def glyph_stats() -> dict:
    """Fonts and shapes in the glyph memory (cached by file names + mtimes; DBs are parsed lazily)."""
    gm = DATA / "glyph-memory"
    files = sorted(gm.glob("*.json")) if gm.is_dir() else []
    key = tuple((f.name, f.stat().st_mtime_ns) for f in files)
    if _glyph_cache["key"] == key:
        return _glyph_cache["value"]
    baseline = {f.name for f in BASELINE_DIR.glob("*.json")} if BASELINE_DIR.is_dir() else set()
    shapes = italic = fused = 0
    sets: list[dict] = []
    for f in files:
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        n_shapes = n_clusters = 0
        for sh in d.get("shapes", []):
            shapes += 1
            n_shapes += 1
            n_clusters += not sh.get("cluster")
            for v in sh.get("variants", []):
                if v.get("geo_italic", [0, 0])[1] > v.get("geo_italic", [0, 0])[0]:
                    italic += 1
                    break
            if any(len(lab) > 1 for v in sh.get("variants", []) for lab in v.get("votes", {})):
                fused += 1
        sets.append({"name": f.stem, "shapes": n_shapes, "clusters": n_clusters, "unit": d.get("unit"),
                     "updated": int(f.stat().st_mtime), "baseline": f.name in baseline,
                     "learned_with": d.get("learned_with")})
    sets.sort(key=lambda s: -s["updated"])
    value = {"fonts": len(files), "fonts_learned_here": sum(1 for f in files if f.name not in baseline),
             "shapes": shapes, "italic_shapes": italic, "fused_shapes": fused, "sets": sets}
    _glyph_cache.update(key=key, value=value)
    return value


_inventory_cache: dict[str, tuple[int, dict]] = {}


def glyph_inventory(name: str) -> dict | None:
    """The reviewable content of one glyph set (see pipeline.glyph_inventory), cached per file version."""
    if not name or _STEM.search(name) or name.startswith("."):
        return None
    path = DATA / "glyph-memory" / f"{name}.json"
    if not path.is_file():
        return None
    stamp = path.stat().st_mtime_ns
    hit = _inventory_cache.get(name)
    if hit and hit[0] == stamp:
        return hit[1]
    from .glyphdb import GlyphDB
    from .pipeline import glyph_inventory as inventory
    db = GlyphDB.load(path)
    db.load_interim(DATA / "word-memory")       # this server's glyphs read once (not published)
    value = inventory(db)
    value["updated"] = int(path.stat().st_mtime)
    if len(_inventory_cache) >= 8:                 # a handful of recently viewed sets
        _inventory_cache.pop(next(iter(_inventory_cache)))
    _inventory_cache[name] = (stamp, value)
    return value


from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    global queue, store, pool
    queue = asyncio.Queue()          # bound to this process's event loop
    jobs.clear()
    DATA.mkdir(parents=True, exist_ok=True)
    store = Store(DATA / "stats.sqlite")
    pool = WorkerPool(WORKERS, VLM_SLOTS, VLM_SLOTS_PER_JOB, POOL_ENTRY)
    await pool.start()
    tasks = [asyncio.create_task(worker()) for _ in range(WORKERS)]
    tasks += [asyncio.create_task(sweeper()), asyncio.create_task(queue_ticker())]
    yield
    for t in tasks:
        t.cancel()
    await pool.stop()


app = FastAPI(title="vobsub-to-srt", docs_url=None, redoc_url=None, lifespan=lifespan)
jobs: dict[str, Job] = {}
queue: asyncio.Queue = None  # type: ignore[assignment]  # created in lifespan()
pool: WorkerPool = None  # type: ignore[assignment]  # created in lifespan()
store: Store = None  # type: ignore[assignment]  # opened in lifespan()


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
    wait = store.check_job(ip)
    if wait is not None:
        raise HTTPException(429, f"job limit reached; try again in {wait} s", headers={"Retry-After": str(wait)})
    stem = _STEM.sub("_", Path(idx.filename or "subtitle").stem)[:80] or "subtitle"
    job_id = uuid.uuid4().hex[:12] + secrets.token_hex(2)
    job = Job(job_id, stem, ip, upload=VobSubData(stem, idx_bytes, sub_bytes))
    store.touch_user(ip)
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
    return JSONResponse(job.result.report,
                        headers={"Content-Disposition": f'attachment; filename="{job.stem}.report.json"'})


@app.get("/api/stats")
async def stats():
    g = glyph_stats()
    return {"version": app_version(), "fonts": g["fonts"], "queued": queue.qsize(),
            "running": sum(1 for j in jobs.values() if j.status == "running"),
            "limits": {"jobs_per_hour": JOBS_PER_HOUR, "vlm_per_day": VLM_PER_DAY, "max_vlm_cues": MAX_VLM_CUES,
                       "max_cues": MAX_CUES, "job_ttl": JOB_TTL},
            "usage": store.snapshot(), "glyphs": {k: v for k, v in g.items() if k != "sets"}}


@app.get("/api/glyphs")
async def glyph_sets():
    """The glyph sets in memory, newest first (what the converter has learned so far)."""
    return {"sets": glyph_stats()["sets"]}


@app.get("/api/glyphs/{name}")
async def glyph_set(name: str):
    inv = await asyncio.to_thread(glyph_inventory, name)
    if inv is None:
        raise HTTPException(404, "unknown glyph set")
    return inv


@app.get("/api/glyphs/{name}/download")
async def glyph_set_download(name: str, request: Request):
    """The publishable glyph set as a file: letter bitmaps, labels, votes and gap statistics. Word
    memory, source hashes and training counts stay on the server (they live in a separate file), so
    anyone may fetch a set and contribute it to the repository's baseline (tools/memory_import.py)."""
    if not name or _STEM.search(name) or name.startswith("."):
        raise HTTPException(404, "unknown glyph set")
    path = DATA / "glyph-memory" / f"{name}.json"
    if not path.is_file():
        raise HTTPException(404, "unknown glyph set")
    etag = f'"{path.stat().st_mtime_ns}-{path.stat().st_size}"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
    from .glyphdb import GlyphDB
    body = await asyncio.to_thread(lambda: json.dumps(GlyphDB.load(path).to_json()))
    return Response(body, media_type="application/json",
                    headers={"ETag": etag, "Cache-Control": "public, max-age=300",
                             "Content-Disposition": f'attachment; filename="{name}.json"'})


@app.get("/healthz")
async def healthz():
    return {"ok": True}


def _asset_version(name: str) -> str:
    """Short content hash, appended to asset links so browsers never keep a stale stylesheet after
    a deploy (the files themselves are cached for an hour)."""
    try:
        return hashlib.sha256((STATIC / name).read_bytes()).hexdigest()[:10]
    except OSError:
        return "0"


@app.get("/", response_class=HTMLResponse)
async def index():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    return html.replace('href="/static/app.css"', f'href="/static/app.css?v={_asset_version("app.css")}"', 1)


_STATIC_TYPES = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml"}


@app.get("/static/{name}")
async def static_file(name: str):
    path = STATIC / name
    if "/" in name or "\\" in name or path.suffix not in _STATIC_TYPES or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type=_STATIC_TYPES[path.suffix], headers={"Cache-Control": "public, max-age=3600"})


# ---------------------------------------------------------------- worker & housekeeping

async def _run(job: Job) -> None:
    job.status = "running"
    job.push({"event": "started"})
    budget = store.vlm_left(job.ip) if MAX_VLM_CUES is None else min(MAX_VLM_CUES, store.vlm_left(job.ip))
    config = JobConfig(glyph_memory_dir=DATA / "glyph-memory", word_memory_dir=DATA / "word-memory",
                       dict_dir=DATA / "dictionaries", max_vlm_cues=budget)

    def progress(e: dict) -> None:            # events arrive on the event loop (pool pipe reader)
        job.push(e)
        _note_progress(job, e)

    try:
        result = await pool.run(job.upload, config, progress, timeout=JOB_TIMEOUT)
        job.result = result
        job.vlm_used = result.vlm_used
        store.add_vlm(job.ip, result.vlm_used)
        store.add_job(result.report)
        job.status = "done"
        if job.events[-1].get("event") != "done":
            job.push({"event": "done", "unresolved": result.unresolved})
    except Exception as e:  # noqa: BLE001
        log.exception("job %s failed", job.id)
        job.status = "error"
        job.error = f"{type(e).__name__}: {e}"[:300]
        store.add_error()
        job.push({"event": "error", "message": job.error})
    finally:
        job.upload = None                                   # uploads never outlive the job


QUEUE_PUSH_EVERY = 3.0        # seconds between queue updates to waiting jobs


def _note_progress(job: Job, e: dict) -> None:
    """Remember the running job's progress for the queue view."""
    for k in ("cues", "resolved", "round"):
        if e.get(k) is not None:
            job.snap[k] = e[k]


async def queue_ticker() -> None:
    """Waiting jobs hear where they stand every few seconds, whether or not the running job emits
    progress (a long vision round may be quiet)."""
    last = time.monotonic()
    while True:
        await asyncio.sleep(min(QUEUE_PUSH_EVERY, 0.5))
        if time.monotonic() - last >= QUEUE_PUSH_EVERY and any(j.status == "queued" for j in jobs.values()):
            last = time.monotonic()
            broadcast_queue()


def broadcast_queue() -> None:
    """Tell every waiting job where it stands: its position, how many jobs run ahead of it and how
    far the running job is. Without this a waiting user sees a frozen page until their turn."""
    running = [j for j in jobs.values() if j.status == "running"]
    waiting = sorted((j for j in jobs.values() if j.status == "queued"), key=lambda j: j.created)
    ahead = {"cues": sum(j.snap.get("cues", 0) for j in running),
             "resolved": sum(j.snap.get("resolved", 0) for j in running),
             "round": max((j.snap.get("round", 0) for j in running), default=0)} if running else None
    for k, j in enumerate(waiting, 1):
        j.push({"event": "queue", "position": k, "ahead": len(running) + k - 1, "running": ahead})


async def worker() -> None:
    while True:
        job_id = await queue.get()
        job = jobs.get(job_id)
        if job:
            job.status = "running"
            broadcast_queue()
            await _run(job)
            broadcast_queue()
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
