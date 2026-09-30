# vobsub-to-srt CLI image. State lives in /data (mount it): glyph-memory/, word-memory/,
# dictionaries/, cache/, out/. The baseline glyph memory from the repo seeds /data/glyph-memory
# on first start (see docker/entrypoint.sh).
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_NO_CACHE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# dependencies first (cached layer), then the project
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY vobsub_to_srt ./vobsub_to_srt
COPY glyph-memory ./glyph-memory
COPY README.md LICENSE ./
RUN uv sync --locked --no-dev

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && mkdir -p /data
VOLUME ["/data"]
WORKDIR /data
ENV PATH="/app/.venv/bin:$PATH" VOBSUB_TO_SRT_DICT_DIR=/data/dictionaries
ENTRYPOINT ["/entrypoint.sh"]
CMD ["--help"]
