# ---- Stage 1: build wheels -------------------------------------------------
# Multi-stage build: compile/download dependencies in a throwaway layer so the
# final image ships no build toolchain — smaller and a reduced attack surface.
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ---- Stage 2: runtime ------------------------------------------------------
FROM python:3.12-slim

# Run as a non-root user: baseline container security practice.
RUN useradd --create-home --shell /usr/sbin/nologin werby
WORKDIR /srv/werby

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

COPY app ./app
COPY scripts ./scripts

RUN mkdir -p data/chroma && chown -R werby:werby /srv/werby
USER werby

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
