# syntax=docker/dockerfile:1

# ── Stage 1: install dependencies ────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /install

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install/pkgs -r requirements.txt


# ── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install/pkgs /usr/local

# Copy application source and the baked-in model artifact
COPY src/ ./src/
COPY models/ ./models/

# Non-root user — principle of least privilege
RUN useradd --no-create-home --shell /bin/false appuser
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
