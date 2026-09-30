"""One conversion as a self-contained job: input files in, SRT + report out, nothing else left behind.

Meant for services (web app, queue workers) that run many untrusted files against one shared glyph
memory:
  - the VLM answer cache is per job (answers contain subtitle text and must not persist across users),
  - uploaded files and outputs live in a temporary directory that is removed with the job,
  - the VLM budget bounds the cost of a single job,
  - progress is reported through a callback instead of the log.
Only the glyph memory (and, if enabled, the private word memory) is shared and persistent.
"""
from __future__ import annotations

import asyncio
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .pipeline import Options, process_file
from .vlm import VLMClient, endpoint_configured


@dataclass
class JobResult:
    srt: str                       # SRT text
    report: dict                   # the .report.json content
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


async def run_job(idx_path: Path, config: JobConfig = JobConfig(),
                  progress: Callable[[dict], None] | None = None) -> JobResult:
    """Convert one .idx/.sub pair. Raises on unreadable input; a missing VLM endpoint is not an error
    (the job runs teacher-less)."""
    idx_path = Path(idx_path)
    sub_path = idx_path.with_suffix(".sub")
    if not idx_path.is_file() or not sub_path.is_file():
        raise FileNotFoundError(f"{idx_path} and {sub_path.name} are both required")
    work = Path(tempfile.mkdtemp(prefix="vobsub-job-"))
    try:
        opts = Options(db_dir=config.glyph_memory_dir, private_dir=config.word_memory_dir,
                       out_dir=work / "out", debug_dir=None, batch_size=16, mode="hybrid",
                       track=config.track, context=config.context, max_vlm_cues=config.max_vlm_cues,
                       keep_special_chars=config.keep_special_chars, word_memory=config.word_memory,
                       progress=progress, **config.extra_options)
        if config.dict_dir is not None:
            import os
            os.environ["VOBSUB_TO_SRT_DICT_DIR"] = str(config.dict_dir)
        explicit = bool(config.base_url and config.api_key and config.model)
        if explicit or endpoint_configured():
            async with VLMClient(cache_dir=work / "cache", concurrency=config.concurrency,
                                 base_url=config.base_url, api_key=config.api_key,
                                 model=config.model) as client:
                report = await process_file(idx_path, client, opts)
        else:
            opts.mode = "nocr-only"
            if progress:
                progress({"event": "no_vlm", "file": idx_path.name})
            report = await process_file(idx_path, None, opts)
        srt = (work / "out" / (idx_path.stem + ".srt")).read_text(encoding="utf-8")
        report.pop("vlm_raw", None)                      # raw answers are subtitle text: not for callers
        budget = report.get("vlm_budget", {})
        return JobResult(srt=srt, report=report, unresolved=report["by_source"].get("fallback", 0),
                         vlm_used=budget.get("used", 0), budget_exhausted=bool(budget.get("exhausted")))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_job_sync(idx_path: Path, config: JobConfig = JobConfig(),
                 progress: Callable[[dict], None] | None = None) -> JobResult:
    return asyncio.run(run_job(idx_path, config, progress))
