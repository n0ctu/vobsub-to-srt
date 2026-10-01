"""One conversion as a self-contained, memory-only job: bytes in, SRT text out.

Meant for services (web app, queue workers) that run many untrusted files against one shared glyph
memory. Privacy by design:
  - the input pair is read from memory (or from disk, when given a path) and never copied anywhere,
  - the VLM answer cache lives in the client object for this job only, never on disk,
  - no output, report or debug file is written; the SRT and the report are returned,
  - the VLM budget bounds the cost of a single job; progress goes to a callback.
The only persistent state a job touches is the glyph memory (letter shapes) and, next to it, the
private sidecar (hashes of learned cue images, so re-runs never count an image twice).
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .pipeline import Options, VobSubData, process_file
from .vlm import VLMClient, endpoint_configured


@dataclass
class JobResult:
    srt: str                       # SRT text
    report: dict                   # the report (no raw VLM answers)
    unresolved: int                # cues written with the placeholder
    vlm_used: int                  # API requests made for this job
    budget_exhausted: bool


@dataclass
class JobConfig:
    glyph_memory_dir: Path = Path("glyph-memory")
    word_memory_dir: Path = Path("word-memory")
    dict_dir: Path | None = None           # None: default lookup (see lexicon.py)
    max_vlm_cues: int | None = 300
    concurrency: int = 2
    context: int = 12
    keep_special_chars: bool = False
    word_memory: bool = False
    track: int = 0
    # explicit endpoint (e.g. a user's own key); None = from the environment / .env
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    extra_options: dict = field(default_factory=dict)


def _source(source: Path | VobSubData | str) -> VobSubData:
    if isinstance(source, VobSubData):
        return source
    idx_path = Path(source)
    sub_path = idx_path.with_suffix(".sub")
    if not idx_path.is_file() or not sub_path.is_file():
        raise FileNotFoundError(f"{idx_path} and {sub_path.name} are both required")
    return VobSubData(idx_path.stem, idx_path.read_bytes(), sub_path.read_bytes())


async def run_job(source: Path | VobSubData | str, config: JobConfig = JobConfig(),
                  progress: Callable[[dict], None] | None = None, limiter=None) -> JobResult:
    """Convert one .idx/.sub pair. Raises on unreadable input; a missing VLM endpoint is not an error
    (the job runs teacher-less)."""
    data = _source(source)
    opts = Options(db_dir=config.glyph_memory_dir, private_dir=config.word_memory_dir,
                   out_dir=None, debug_dir=None, diagnostics=False, batch_size=16, mode="hybrid",
                   track=config.track, context=config.context, max_vlm_cues=config.max_vlm_cues,
                   keep_special_chars=config.keep_special_chars, word_memory=config.word_memory,
                   progress=progress, **config.extra_options)
    if config.dict_dir is not None:
        os.environ["VOBSUB_TO_SRT_DICT_DIR"] = str(config.dict_dir)
    explicit = bool(config.base_url and config.api_key and config.model)
    if explicit or endpoint_configured():
        async with VLMClient(cache_dir=None, concurrency=config.concurrency, base_url=config.base_url,
                             api_key=config.api_key, model=config.model, limiter=limiter) as client:
            res = await process_file(data, client, opts)
    else:
        opts.mode = "nocr-only"
        if progress:
            progress({"event": "no_vlm", "file": data.name})
        res = await process_file(data, None, opts)
    budget = res.report.get("vlm_budget", {})
    return JobResult(srt=res.srt, report=res.report, unresolved=res.report["by_source"].get("fallback", 0),
                     vlm_used=budget.get("used", 0), budget_exhausted=bool(budget.get("exhausted")))


def run_job_sync(source: Path | VobSubData | str, config: JobConfig = JobConfig(),
                 progress: Callable[[dict], None] | None = None) -> JobResult:
    return asyncio.run(run_job(source, config, progress))
