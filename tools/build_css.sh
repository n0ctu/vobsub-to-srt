#!/bin/sh
# Rebuild vobsub_to_srt/static/app.css from src.css + the classes used in index.html.
# Uses the Tailwind standalone CLI (no node); the binary is cached in .cache/. Commit app.css.
set -e
cd "$(dirname "$0")/.."
V=v4.1.14
BIN=.cache/tailwindcss
if [ ! -x "$BIN" ]; then
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64)  A=linux-x64 ;;
        Linux-aarch64) A=linux-arm64 ;;
        Darwin-arm64)  A=macos-arm64 ;;
        Darwin-x86_64) A=macos-x64 ;;
        *) echo "unsupported platform, see https://github.com/tailwindlabs/tailwindcss/releases"; exit 1 ;;
    esac
    mkdir -p .cache
    curl -fsSL -o "$BIN" "https://github.com/tailwindlabs/tailwindcss/releases/download/$V/tailwindcss-$A"
    chmod +x "$BIN"
fi
"$BIN" -i vobsub_to_srt/static/src.css -o vobsub_to_srt/static/app.css --minify
ls -l vobsub_to_srt/static/app.css
