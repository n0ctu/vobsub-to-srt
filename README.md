# vobsub-to-srt

**Try it online: [vts.canihaz.cloud](https://vts.canihaz.cloud)** — drop a `.idx`/`.sub` pair, get the `.srt`.
Every font learned there becomes part of the shared glyph memory in this repository.

VobSub (`.idx`/`.sub`) → SRT. Subtitle glyphs are read by deterministic bitmap matching
(in the spirit of Subtitle Edit's binary image compare); a vision LLM (any OpenAI-compatible
endpoint, developed against DeepSeek) acts as the *teacher* for glyphs the database doesn't know yet.
A new font costs a few dozen VLM calls; later files with the same font need none.

## Setup

```sh
uv sync --locked                     # reproducible venv (.venv) from uv.lock, Python 3.12
cp .env.example .env                 # then fill in the VLM endpoint (below)
```

The teacher is any OpenAI-compatible chat endpoint with a vision model. `.env` (never committed):

```
DEEPSEEK_BASE_URL=https://api.example.com/v1   # base URL of the /chat/completions endpoint
DEEPSEEK_API_KEY=...
DEEPSEEK_MODEL=...                             # a model that accepts image input
```

Hunspell dictionaries for the lexicon gate are downloaded on first use into `dictionaries/`
(`--dict-dir`, `--no-dict-download`); system dictionaries in `/usr/share/hunspell` are used too.
They are not bundled because of their per-language licenses.

## Usage

```sh
uv run vobsub-to-srt Subs/*.idx                   # hybrid (default)
uv run vobsub-to-srt --mode nocr-only X.idx       # never call the API; unknown glyphs become �
uv run vobsub-to-srt --mode vlm-only X.idx        # VLM for every cue (reference / comparison)
```

Without a VLM endpoint configured, the default mode runs teacher-less: fonts in the glyph memory
are read as usual, and a file whose font is unknown is written with `�` for the unknown glyphs plus a
warning that a VLM would be needed to learn it. So a clone works offline for the shipped baseline fonts.

| Option | Default | |
|---|---|---|
| `--mode` | `hybrid` | `hybrid`, `nocr-only` (no API), `vlm-only` (VLM for every cue) |
| `--concurrency` | 2 | parallel VLM requests (429s lower it automatically) |
| `--batch-size` | 16 | cues sent to the VLM per round before re-matching |
| `--track` | 0 | subtitle track of a multi-track `.idx` |
| `--context` | 12 | previous cues given to the VLM as reference, 0 = off |
| `--max-vlm-cues` | none | cap on VLM requests per file; beyond it the file is finished teacher-less (`�` for unknown glyphs) |
| `--lexicon` | `auto` | tie-break for pixel-identical I/l: `auto` (wordfreq, then Hunspell), `wordfreq`, `hunspell`, `off` |
| `--keep-special-chars` | off | keep typographic variants instead of folding them (see below) |
| `--tool` | off | VLM submits transcripts via a forced tool call instead of plain text |
| `--word-memory` | off | remember resolved words (private, see below) |
| `--dict-dir`, `--no-dict-download` | `dictionaries/` | Hunspell dictionaries (downloaded on first use) |
| `--glyph-memory-dir`, `--word-memory-dir` | `glyph-memory/`, `word-memory/` | |
| `--out-dir`, `--cache-dir`, `--debug-dir` | `out/`, `cache/vlm/`, `debug/` | |

Outputs: `out/<name>.srt`; `out/<name>.report.json` (VLM calls, where each cue's text came from,
flagged cues, failures, raw VLM answers); `debug/` (images of cues that could not be read).

**Baseline fonts.** `glyph-memory/` is tracked in the repo and ships glyph memories for fonts
already learned (currently two common sans-serif subtitle fonts at 1080p, upright and italic). A
clone therefore reads those fonts without any VLM (`--mode nocr-only` works offline for them). New
fonts you process are written next to them; since a glyph memory contains no subtitle text (see
below), please contribute them back with a pull request to strengthen the baseline.

**Glyph memory vs. word memory.** `glyph-memory/<name>.json` holds one learned font: glyph bitmaps
(including fused letter pairs such as `rt`), their labels, the gap model and multi-glyph characters.
Names are random (e.g. `calm-sable-4e11`) and nothing in it is subtitle content, so it can be shared
and reused by anyone. `word-memory/<name>.json` is private: hashes of the cue images learned from (so
re-runs never count the same image twice) and, with `--word-memory`, the words already resolved per
glyph sequence (e.g. names), which helps with pixel-identical `I`/`l` on your own library.

## How it works

1. **Decode** `.idx/.sub` (MPEG-PS demux, SPU RLE) and isolate the glyph fill colour
   (the colour with the fewest edges facing other colours) into a binary mask.
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
   a word seen before with the exact glyph sequence decides; otherwise the lexicon gate: (1) wordfreq — a clear
   frequency lead, or the only reading that is a word; (2) Hunspell — the only valid spelling;
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

**Character simplification** (default; `--keep-special-chars` turns it off): everything the VLM returns
is folded before it is learned, compared or written — apostrophes/single quotes `´ ` ’ ‘ ‚ ′ ‹ ›` → `'`,
double quotes `„ “ ” « » ″` → `"`, dashes `‐ – — ― −` → `-`, `…` → `...`, ligatures `ﬁ ﬂ ﬀ` → letters,
non-breaking/thin spaces → space, soft hyphens and zero-width characters removed, Unicode NFC.
The VLM is inconsistent between these variants (it wrote `O'Neil` for `O´Neil` in 3 of 4 cues);
folded, they are one class and cannot outvote each other. The prompt stays literal. Each glyph DB
records its character set, so simplified and literal DBs (`*.literal.json`) never mix.

Prompts live in `vobsub_to_srt/prompts.py`. The transcription prompt demands a literal,
letter-by-letter reading (typos kept); a planned spell-check pass will get its own prompt.

## Docker

```sh
mkdir -p data/in && cp .env.example .env            # endpoint optional: without it, teacher-less
docker compose run --rm vobsub-to-srt /data/in/movie.idx      # -> data/out/movie.srt
```

The image `ghcr.io/n0ctu/vobsub-to-srt:latest` is built by CI from `main` (tags `vX.Y.Z` from
releases). All state lives in the `data/` volume: `glyph-memory/` is seeded from the image's baseline on
first start and grows from there, `word-memory/`, `dictionaries/`, `cache/` and `out/` next to it.
`compose.yml` includes Watchtower, which pulls a new `:latest` and replaces the container automatically.

## Web app

`uv sync --extra web && uv run vobsub-to-srt-web` (or `docker compose up -d`) serves a one-page app on
port 8000: drop the `.idx` and `.sub`, watch the progress, download the SRT. Design:

- one worker converts jobs one after another (it is the only writer of the shared glyph memory and
  the global VLM throttle); the page shows the queue position;
- uploads and results live under `<data>/jobs/` and are deleted after `VTS_JOB_TTL` (1 h); the
  uploaded files are removed as soon as the job ends; the VLM cache is per job, so no subtitle text
  survives a job — only the glyph memory grows;
- per-IP limits: `VTS_JOBS_PER_HOUR` (6) and `VTS_VLM_PER_DAY` (600 VLM requests); when a client's
  daily allowance is used up its jobs still run, teacher-less; per job `VTS_MAX_VLM_CUES` (300);
- input caps: `.sub` ≤ `VTS_MAX_SUB_MB` (64), ≤ `VTS_MAX_CUES` (3000) cues; job timeout 15 min;
- `VTS_TRUST_PROXY=1` takes the client address from `X-Forwarded-For` (only behind your own proxy).
- the page keeps a personal queue in the browser's localStorage: several tracks can be added and are
  submitted one after another; each finished SRT is fetched and stored client-side, so it can be
  downloaded again after the server's copy expired. Nothing about users is stored on the server.

Reverse proxy (nginx on another host, TLS terminated there):

```nginx
location / {
    proxy_pass http://10.0.0.5:8787;            # the private address bound in compose.yml
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header Host $host;
    client_max_body_size 80m;                   # a .sub is 5-10 MB per track
    proxy_read_timeout 900s;                    # progress stream while a new font is learned
    proxy_buffering off;
}
```

## As a library / service

```python
from vobsub_to_srt.job import run_job_sync, JobConfig
res = run_job_sync("movie.idx", JobConfig(max_vlm_cues=300), progress=print)
res.srt, res.unresolved, res.vlm_used, res.budget_exhausted, res.report
```

`run_job` isolates one conversion: per-job VLM cache and temp files (removed afterwards, so no
subtitle text persists across jobs), a VLM budget, progress events (`probe`, `round`, `retry`,
`budget_exhausted`, `done`, `no_vlm`), and an optional per-job endpoint (`JobConfig(base_url=,
api_key=, model=)` for bring-your-own-key). Only the shared glyph memory is written.

## Development

```sh
uv run pytest
uv run python tools/compare_srt.py reference.srt out/X.srt      # CER, exact cues, style diffs
uv run python tools/eval_context.py X.idx --context 12          # VLM accuracy with/without context
uv run python tools/qc_sheet.py X.idx out/X.srt --random 6 --risky 4   # image vs. transcript sheet
tools/build_css.sh                                              # rebuild static/app.css (Tailwind CLI) after editing index.html
uv run vobsub-to-srt --debug-rescale 0.6667 X.idx               # simulate another resolution
```

Experiments (test subtitles, reference transcripts, logs) live in the git-ignored `experiments/`.
Tests use synthetic subtitle images rendered with DejaVu/Liberation fonts; no subtitle files are needed.

## Status

Verified on 1080p VobSubs of two releases (two sans-serif fonts, English and German, upright and
italic): the output matched the images on every cue of four fully checked tracks (2,791 cues) and
on 302 sampled cues of further episodes. Bold and underline are only tested on synthetic images.
Other resolutions are handled through the teacher transfer (tested by downscaling). Other scripts
than Latin are untested.

## License

MIT (see `LICENSE`). Hunspell dictionaries are fetched at runtime under their own licenses and are
not part of this repository.
