# App image only — PostgreSQL and Jaeger run as separate services (see
# docker-compose.yml). Unlike the original DuckDB version, the warehouse can't
# be baked into this image at build time: seeding needs a live Postgres to
# connect to, so it happens as a one-off command against DATABASE_URL instead
# (see README Quickstart) — `data/seed.py` ships in the final image for
# exactly that, run via `docker exec <container> python data/seed.py`.
#
# syntax=docker/dockerfile:1

# ---- builder: resolve dependencies into an isolated venv ------------------
# Split from the runtime stage so pip's own build environment (hatchling, the
# wheel it produces, any transient build backend) never ends up in the image
# that actually gets deployed — only the installed venv is copied forward.
# This project's dependencies are all pure-Python/manylinux-wheel (no gcc
# needed on this base image), so the size win here is modest, but keeping the
# split means adding a compiled dependency later is a one-line change to this
# stage alone, never the runtime one.
FROM python:3.11-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
# pyproject.toml's build backend (hatchling) needs the actual package source
# present to build the wheel — there's no separate lockfile/requirements.txt
# to install from in isolation, so this layer is invalidated by any app/
# change, not just a dependency change. Still only ever installs into the
# venv, never system site-packages.
COPY pyproject.toml ./
COPY app ./app
RUN pip install .

# ---- runtime: minimal image, no build tooling, non-root -------------------
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Unprivileged user — this process accepts network input and never needs root
# once its dependencies are installed, so running it as root would be an
# avoidable privilege-escalation surface.
RUN groupadd --gid 1000 app && useradd --uid 1000 --gid app --create-home app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
# Only what the running service (and the one-off seed command) actually
# need: app/ is the service itself, data/ is seed.py. Tests, the eval
# harness, docs/screenshots, and repo/CI metadata are intentionally left out
# of the image — see .dockerignore — a production artifact ships what runs,
# not what was used to build and verify it.
COPY app ./app
COPY data ./data

RUN chown -R app:app /app
USER app

EXPOSE 8000

# GET / is unauthenticated by design (serves the static login page), so it
# doubles as a health check without needing a dedicated route or credentials
# baked into the image.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/', timeout=2)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
