# ==============================================================================
# HelixMind Multi-Stage Production Dockerfile
# Stage 1: Build modern React + Vite frontend
# Stage 2: Python 3.11 biomedical runtime with RDKit, ReportLab, and FastAPI
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Frontend Builder
# ------------------------------------------------------------------------------
FROM node:22-slim AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ------------------------------------------------------------------------------
# Stage 2: Production Python Runtime
# ------------------------------------------------------------------------------
FROM python:3.11-slim

LABEL maintainer="HelixMind Team"
LABEL description="Autonomous Scientific Research Platform with Real-Time WebSockets"

# Install OS libraries required for RDKit, ReportLab fonts, and networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libxrender1 \
    libxext6 \
    libfreetype6 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend source code and install as package
COPY helixmind/ ./helixmind/
RUN pip install --no-cache-dir -e .

# Copy compiled frontend from Stage 1 into the location expected by FastAPI
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Ensure persistent artifact directories exist
RUN mkdir -p data/reports data/papers data/structures data/uploads data/memory data/figures

EXPOSE 8000

# Health check verifying API and backend readiness
HEALTHCHECK --interval=20s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["helixmind", "serve", "--host", "0.0.0.0", "--port", "8000"]
