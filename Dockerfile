# EngageIQ - container for Google Cloud Run.
# FastAPI app: Sentence-Transformers + FAISS retrieval, cross-encoder re-rank,
# server-rendered topic maps, and a WeasyPrint HTML->PDF engagement brief.
FROM python:3.12-slim

# Native libraries WeasyPrint needs (Pango / Cairo / GDK-PixBuf / libffi) plus fonts
# for the brief PDF. fonts-lmodern provides Latin Modern (the Computer Modern serif
# the brief targets) so the PDF matches the academic format even off the dev machine.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libpango-1.0-0 libpangocairo-1.0-0 libcairo2 libgdk-pixbuf-2.0-0 \
      libffi-dev shared-mime-info \
      fonts-dejavu-core fonts-liberation fonts-lmodern \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf \
    OMP_NUM_THREADS=1 \
    KMP_DUPLICATE_LIB_OK=TRUE

WORKDIR /app

# CPU-only PyTorch first (the default PyPI wheel bundles CUDA and is far larger).
RUN pip install --no-cache-dir torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu

# The remaining runtime deps (torch already satisfied -> pip will not pull the CUDA build).
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Bake the two transformer models into the image so cold starts never re-download them.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Application code, the SPA, and the offline data snapshot (.dockerignore drops the
# raw ingestion inputs and the read-write user DB; that DB self-creates on first run).
COPY code/ ./code/
COPY mockups/ ./mockups/
COPY data/ ./data/

# Cloud Run sends traffic to $PORT (default 8080). One worker: the in-process engine
# cache is per-process, so multiple workers would each load the models and diverge.
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080} --app-dir code --workers 1"]
