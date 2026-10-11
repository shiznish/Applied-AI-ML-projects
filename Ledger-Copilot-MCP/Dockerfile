FROM python:3.12-slim AS builder

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim AS runtime

RUN groupadd --gid 1000 ledger && \
    useradd --uid 1000 --gid ledger --create-home --shell /usr/sbin/nologin ledger

COPY --from=builder /install /usr/local

# ErpClient resolves its seed-data fixtures relative to its own installed
# location (parents[3]/tests/fixtures) rather than the repo root, so a
# non-editable install needs the fixtures copied to that same resolved path.
COPY tests/fixtures /usr/local/lib/python3.12/tests/fixtures

WORKDIR /app
RUN mkdir -p /data && chown ledger:ledger /data

USER ledger

ENV LEDGER_COPILOT_HOST=0.0.0.0 \
    LEDGER_COPILOT_PORT=8000 \
    LEDGER_COPILOT_AUDIT_DATABASE_PATH=/data/audit.db

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"LEDGER_COPILOT_PORT\"]}/healthz', timeout=3)" || exit 1

CMD ["python", "-m", "ledger_copilot.server"]
