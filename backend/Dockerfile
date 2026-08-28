cd /mnt/dataD/PaperAxiom

cat > backend/Dockerfile <<'EOF'
# ============================================================
# ResearchGPT Backend - Production/Render
# ============================================================

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libpq-dev \
        curl && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Python dependencies
# ------------------------------------------------------------

COPY backend/requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# ------------------------------------------------------------
# Backend application
# ------------------------------------------------------------

COPY backend/app /app/app

# ------------------------------------------------------------
# Frontend
# ------------------------------------------------------------

COPY frontend /app/frontend

# ------------------------------------------------------------
# Runtime
# ------------------------------------------------------------

ENV PORT=8000

EXPOSE 8000

# ------------------------------------------------------------
# Health check
# ------------------------------------------------------------

HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=60s \
    --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# ------------------------------------------------------------
# Start FastAPI
# ------------------------------------------------------------

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
EOF