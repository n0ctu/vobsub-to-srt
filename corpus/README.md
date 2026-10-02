# Benchmark corpus (not in git)

Subtitle tracks for regression and optimisation runs. Everything in this folder except this
README and `index.example.csv` is ignored by git.

## Layout

```
corpus/
  <release>/                  one folder per release / authoring, e.g. banshee-s01-720p-intention
    <stem>.idx + <stem>.sub   one or more tracks (all languages of the release are welcome)
    <stem>.gt.srt             optional: a trusted ground-truth SRT for that track
  index.csv                   one row per track (generated, see below)
```

Folder names: lowercase, `title-season-resolution-source` is a good pattern (`dvd`, `bluray`,
`web`, `rescaled` if the track was downscaled from another master). Drop whole releases rather
than single tracks: the forced/full/other-language tracks of one release share the glyph set and
show what re-use costs.

What helps most, in this order:

1. **Fonts and rasters we do not have yet**: DVD PAL (720x576) and NTSC (720x480), other Blu-ray
   authorings, outlined or shadowed styles, rescaled tracks.
2. **Languages with accents** (French, Spanish, Scandinavian, Eastern European) and all-caps styles.
3. **Ground truth** for a handful of tracks (`<stem>.gt.srt`): the only way to measure accuracy
   instead of agreement with the previous version. A carefully checked VLM-only output
   (`--mode vlm-only`, then proofread against the images) is fine.

A few hundred tracks are plenty. Every first contact with a new glyph set costs real vision
requests; the point is variety, not volume.

## Index

`uv run python tools/corpus_index.py` scans the folder and writes `corpus/index.csv` (release,
stem, languages, cue count, resolution, ground truth yes/no) while keeping the hand-written
columns `source`, `style` and `notes` of an existing index. Fill those in when you know them
(e.g. `source=bluray`, `style=outlined italic-heavy`, `notes=rescaled from 1080p`).
