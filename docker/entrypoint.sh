#!/bin/sh
# Seed the shared glyph memory from the image's baseline: missing sets are installed, sets the
# image ships in a newer version replace the instance's copy (the old copy goes to
# /data/glyph-memory/superseded/ so its learning can still be harvested). See vobsub_to_srt/seed.py.
set -e
mkdir -p /data/glyph-memory /data/word-memory /data/dictionaries
python -m vobsub_to_srt.seed /app/glyph-memory /data/glyph-memory
if [ "$1" = "web" ]; then
    shift
    exec vobsub-to-srt-web "$@"
fi
exec vobsub-to-srt "$@"
