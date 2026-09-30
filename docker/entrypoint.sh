#!/bin/sh
# Seed the shared glyph memory from the image's baseline on first start; never overwrite learned DBs.
set -e
mkdir -p /data/glyph-memory /data/word-memory /data/dictionaries /data/cache /data/out
for f in /app/glyph-memory/*.json; do
    [ -e "/data/glyph-memory/$(basename "$f")" ] || cp "$f" /data/glyph-memory/
done
if [ "$1" = "web" ]; then
    shift
    exec vobsub-to-srt-web "$@"
fi
exec vobsub-to-srt "$@"
