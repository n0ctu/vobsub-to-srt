# VobSub to SRT Converter

> **Experimental.** This pipeline is under active development: recognition strategies, the glyph
> memory format and the private sidecar format change often and without migration. Expect glyph
> sets learned by one version to be rebuilt by the next, and treat results, configuration flags and
> the web app's behaviour as subject to change.

VobSub (`.idx`/`.sub`) to SRT. Subtitle glyphs are read by deterministic bitmap matching (in the spirit of Subtitle Edit's binary image compare). A vision LLM (any OpenAI-compatible endpoint, developed against DeepSeek) acts as the *teacher* for glyphs the database doesn't know yet.
A new font costs a few dozen VLM calls. Later files with the same font need none.

## Online Version

**[vts.canihaz.cloud](https://vts.canihaz.cloud)**

Try it online. Drop a `.idx`/`.sub` pair, get the `.srt`. Uploads are never stored on disk. Only new glyphs learned from new sources get automatically added to the online tools shared glyph memory and get eventually added to this repository. 


## Local Setup

```sh
uv sync --locked                     # reproducible venv (.venv) from uv.lock, Python 3.12
cp .env.example .env                 # then optionally fill in the VLM endpoint (below)
```

The teacher is any OpenAI-compatible chat endpoint with a vision model. `.env`:

```
VLM_BASE_URL=https://api.example.com/v1   # base URL of the /chat/completions endpoint
VLM_API_KEY=...
VLM_MODEL=...                             # a model that accepts image input
```

Hunspell dictionaries for the lexicon gate are downloaded on first use into `dictionaries/` (`--dict-dir`, `--no-dict-download`); system dictionaries in `/usr/share/hunspell` are used too.
They are not bundled because of their per-language licenses.

## Usage

```sh
uv run vobsub-to-srt Subs/*.idx                   # hybrid (default)
uv run vobsub-to-srt --mode nocr-only X.idx       # never call the VLM API; unknown glyphs become �
uv run vobsub-to-srt --mode vlm-only X.idx        # VLM for every cue (reference / comparison)
```

Without a VLM endpoint configured, the default mode runs teacher-less: fonts in the glyph memory are read as usual, and a file whose font is unknown is written with `�` for the unknown glyphs plus a warning that a VLM would be needed to learn it. So a clone works **offline** for the shipped baseline fonts.

| Option | Default | |
|---|---|---|
| `--mode` | `hybrid` | `hybrid`, `nocr-only` (no API), `vlm-only` (VLM for every cue) |
| `--concurrency` | 2 | parallel VLM requests (429s lower it automatically) |
| `--batch-size` | 16 | cues sent to the VLM per round before re-matching |
| `--track` | 0 | subtitle track of a multi-track `.idx` |
| `--context` | 12 | previous cues given to the VLM as reference, 0 = off |
| `--max-vlm-cues` | none | cap on VLM requests per file; beyond it the file is finished teacher-less (`�` for unknown glyphs) |
| `--lexicon` | `auto` | tie-break for pixel-identical I/l: `auto` (wordfreq, then Hunspell), `wordfreq`, `hunspell`, `off` |
| `--diagnostics` | off | write diagnostics to disk: the VLM answer cache (`cache/vlm/`, re-runs free and reproducible), PNGs of unreadable cues (`debug/`), raw VLM answers in the report |
| `--keep-special-chars` | off | keep typographic variants instead of folding them (see below) |
| `--tool` | off | VLM submits transcripts via a forced tool call instead of plain text |
| `--word-memory` | off | remember resolved words (private, see below) |
| `--dict-dir`, `--no-dict-download` | `dictionaries/` | Hunspell dictionaries (downloaded on first use) |
| `--glyph-memory-dir`, `--word-memory-dir` | `glyph-memory/`, `word-memory/` | |
| `--out-dir`, `--cache-dir`, `--debug-dir` | `out/`, off, off | `--cache-dir`/`--debug-dir` switch on that part of the diagnostics with a custom location |

Outputs: `out/<name>.srt`; `out/<name>.report.json` (VLM calls, where each cue's text came from, flagged cues, failures). 

**Baseline fonts.** `glyph-memory/` is tracked in the repo and ships glyph memories for fonts already learned (currently 17 sets: the common Blu-ray sans-serif families at 1080p and 720p, upright and italic, plus raw Blu-ray, SDH and rescaled 720p tracks, each learned on two releases and spot-checked against the images). A clone therefore reads those fonts without any VLM (`--mode nocr-only` works offline for them). On-the-fly VLM learned glyph-sets are automatically added there. Feel free to submit them as a PR so other users can use them as well! 

**Glyph memory vs. word memory.** `glyph-memory/<name>.json` holds one learned glyph-set: glyph bitmaps (including fused letter pairs such as `rt`), their labels, the gap model and multi-glyph characters. `word-memory/<name>.json` is private: hashes of the cue images learned from and, with `--word-memory`, the words already resolved per glyph sequence (e.g. names), which helps with pixel-identical `I`/`l` on your own library.

## How it works

1. **Decode** `.idx/.sub` (MPEG-PS demux, SPU RLE) and isolate the glyph fill colour (the colour with the fewest edges facing other colours) into a binary mask.
2. **Segment** lines and glyphs (connected components; i-dots, umlauts, `:` `;` `!` `?` merged;
   touching letters become multi-character glyphs such as `rt`). Underline strokes are detected and
   removed before splitting.
3. **Probe** the glyph DBs in `glyph-memory/`:
   - *same font and resolution*: share of glyph occurrences with an exact bitmap match (≥50 % → reuse);
   - *same font, other resolution*: a DB is used as **teacher**. The scale is estimated from glyph
     heights and refined by match coverage; teacher glyphs are rescaled and compared by overlap.
     Transferred labels are **priors**: used until a VLM read contradicts them, never allowed to
     override the VLM. Thin strokes (`I l | i 1 ! j`) are never transferred.
   - otherwise a new DB. Each DB stores its font unit (x-height in px); tolerances are relative to it.
4. **Train / infer loop**: all cues are matched; unresolved cues go to the VLM in batches chosen by
   greedy set cover over the most frequent unknown glyphs. The VLM text is aligned to the glyphs
   and learned: label votes per bitmap, gap statistics, multi-glyph characters, word memory.
   **Quarantine:** a label is trusted only with ≥2 reads from different cues and a ≥2/3 majority.
5. **Arbitration**: the VLM text is aligned to the glyphs. Where it contradicts a strongly confirmed
   glyph (≥3 reads, ≥75 %) or multi-glyph character, the DB wins and the cue is flagged. If the VLM
   text does not fit the glyphs at all (dropped/added letters: *Verdopeln* over `p p`) and the DB can
   read the whole cue, the DB reading is used. A final pass re-checks every VLM cue, and every nOCR
   cue is re-read, against the final DB. On the test material the VLM "autocorrects" (*canvass →
   canvas*, *quit → quitt*) and normalizes `O´Neil` to `O'Neil`; the glyph DB keeps the literal text.
6. **Pixel-identical `I`/`l`** (e.g. Arial: both are the same vertical bar): with `--word-memory`,
   a word seen before with the exact glyph sequence decides; otherwise the lexicon gate: (1) wordfreq - a clear
   frequency lead, or the only reading that is a word; (2) Hunspell - the only valid spelling;
   (3) only if no reading is a known word (names, invented words): the only reading without case
   changes inside a word part (`MILLER`, not `MlLLER`). Otherwise the VLM decides. The same gate
   overrides the VLM's own I/l reading on such glyphs.
7. **Retries** with a stricter prompt at 3× scale; anything left is logged with cue, line and glyph
   position and rendered with `�`.
8. **Styles** come from geometry, VLM tags only break ties: `<i>` from word slant (voted per glyph
   shape over the whole file), `<b>` from stroke width (≥1.25× the file median), `<u>` from the
   underline stroke.
9. **Spaces**: additive side-bearing model (letter gap(a,b) ≈ R[a] + L[b]; a space adds a learned
   offset). Ambiguous gaps go to the VLM.

**Character simplification** (default; `--keep-special-chars` turns it off): everything the VLM returns is folded before it is learned, compared or written. Apostrophes/single quotes `´ ’ ‘ ‚ ′ ‹ ›` to `'`, double quotes `„ “ ” « » ″` to `"`, dashes `‐ – — ― −` to `-`, `…` to `...`, ligatures `ﬁ ﬂ ﬀ` to letters, non-breaking/thin spaces to space, soft hyphens and zero-width characters removed, Unicode NFC. With simplification off, double quotes are placed by the image rather than by the model's habit: a quote drawn on the baseline is written `„`, one at cap height keeps its high form.

Prompts live in `vobsub_to_srt/prompts.py`. The transcription prompt demands a literal, letter-by-letter reading (typos kept); a planned optional spell-check pass will get its own prompt.

## Docker

```sh
mkdir -p data/in && cp .env.example .env                      # endpoint optional: without it, teacher-less
docker compose --profile cli run --rm vobsub-to-srt /data/in/movie.idx   # -> data/out/movie.srt
```

The image `ghcr.io/n0ctu/vobsub-to-srt:latest` is built by CI from `main` (tags `vX.Y.Z` from releases). All state lives in the `data/` volume: `glyph-memory/` is seeded from the image's baseline on first start and grows from there, `word-memory/` and `dictionaries/` next to it (`out/` for CLI results). `compose.yml` includes Watchtower, which pulls a new `:latest` and replaces the container automatically.

## Web app (self-hosting)

The same image runs the web app: `docker compose up -d` starts it on the host port set in `compose.yml` (default `127.0.0.1:8787`; bind a private address if a reverse proxy on another host terminates TLS, and give it `client_max_body_size 80m`, `proxy_read_timeout 900s` and `proxy_buffering off`). Without Docker: `uv sync --extra web && uv run vobsub-to-srt-web`.

Jobs run in `VTS_WORKERS` worker processes. The web process owns the queue and hands out the vision-model request slots, so the endpoint never sees more than `VTS_VLM_SLOTS` requests at once however many jobs run; waiting jobs are told their position every few seconds. Workers learn into the shared glyph memory concurrently: each save merges the job's learning into the file's current content under a file lock.

Everything a user uploads stays in memory: uploads are dropped when their job ran, results are held in memory for `VTS_JOB_TTL` (10 min) so the browser can fetch them, the VLM cache is per job. Only the shared `glyph-memory/` grows. Limits and usage statistics live in `<data>/stats.sqlite`; clients are identified by a salted hash of their address whose salt changes daily.

The page's "Glyph memory" section lists every glyph set learned so far and renders one on request (bitmap, reading, style, how often it was seen), so what the shared memory holds is open to review: `GET /api/glyphs` and `GET /api/glyphs/<set>` return the same data. Private sidecar data (word memory, learned image hashes, per-pixel training counts) is never served.

| Variable | Default | |
|---|---|---|
| `VTS_DATA` | `.` (`/data` in Docker) | glyph memory, word memory, dictionaries, `stats.sqlite` |
| `VTS_JOBS_PER_HOUR`, `VTS_VLM_PER_DAY` | 20, 1000 | limits per user; an exhausted daily allowance runs jobs teacher-less |
| `VTS_MAX_VLM_CUES` | 0 (off) | optional cap on VLM requests per job |
| `VTS_MAX_CUES`, `VTS_MAX_SUB_MB`, `VTS_MAX_QUEUE` | 6000, 64, 20 | input and queue caps |
| `VTS_JOB_TTL`, `VTS_JOB_TIMEOUT` | 600, 900 | seconds |
| `VTS_WORKERS` | 2 | worker processes: jobs converted at the same time (one CPU core each) |
| `VTS_VLM_SLOTS`, `VTS_VLM_SLOTS_PER_JOB` | 4, 2 | vision requests in flight: all jobs together, and per job |
| `VTS_TRUST_PROXY` | 0 | `1` to take the client address from `X-Forwarded-For` (only behind your own proxy) |

## As a library / service

```python
from vobsub_to_srt.job import run_job_sync, JobConfig
res = run_job_sync("movie.idx", JobConfig(max_vlm_cues=300), progress=print)
res.srt, res.unresolved, res.vlm_used, res.budget_exhausted, res.report
```

`run_job` converts one file entirely in memory (a path or a `VobSubData(name, idx_bytes, sub_bytes)`): per-job VLM cache, no temp files, a VLM budget, progress events (`probe`, `round`, `cues`, `glyphs`, `retry`, `budget_exhausted`, `done`, `no_vlm`), and an optional per-job endpoint (`JobConfig(base_url=, api_key=, model=)`). Only the shared glyph memory is written.

## Roadmap

- Drop the per-image hash sidecar for web jobs, so the only thing a job leaves behind is letter shapes.
- Bring-your-own-key (a user's own OpenAI-compatible endpoint for their jobs).
- A candidates tier for fonts learned on the server before they are promoted into the baseline.
- Dictionaries for more languages in the lexicon gate.

## Development

```sh
uv run pytest
uv run python tools/compare_srt.py reference.srt out/X.srt      # CER, exact cues, style diffs
uv run python tools/eval_context.py X.idx --context 12          # VLM accuracy with/without context
uv run python tools/qc_sheet.py X.idx out/X.srt --random 6 --risky 4   # image vs. transcript sheet
tools/build_css.sh                                              # rebuild static/app.css (Tailwind CLI) after editing index.html
uv run vobsub-to-srt --debug-rescale 0.6667 X.idx               # simulate another resolution
```

## Status

Verified on latin 1080p VobSubs (two sans-serif fonts, English and German, upright and italic): the output matched the images on every cue of four fully checked tracks (2,791 cues) and on 302 sampled cues of further episodes. Bold and underline are only tested on synthetic images.
Other resolutions are handled through the teacher transfer (tested by downscaling). Other scripts than Latin should work, but are untested as of the current release.

## License

MIT (see `LICENSE`). Hunspell dictionaries are fetched at runtime under their own licenses and are not part of this repository.
