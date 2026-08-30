FROM python:3.11-slim

# Prevent buffered logs from hiding errors, keep TF quiet 
ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_ENABLE_ONEDNN_OPTS=0

# opencv-python-headless still needs a couple of system libs at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (better layer caching — code changes won't force a full reinstall)
COPY deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY webapp/ webapp/
COPY src/ src/
COPY db/ db/
COPY models/ models/
COPY retinascan_ai.db .

EXPOSE 8080

# Shell form (not exec array) so ${PORT} actually expands — Cloud Run injects PORT at runtime
CMD streamlit run webapp/newapp.py \
    --server.port=${PORT:-8080} \
    --server.address=0.0.0.0 \
    --server.headless=true