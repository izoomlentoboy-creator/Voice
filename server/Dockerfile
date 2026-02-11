FROM python:3.11-slim

WORKDIR /app

# System deps for librosa/soundfile
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsndfile1 \
        libffi-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r /app/server/requirements.txt

# Copy ML package, CLI, and server
COPY voice_disorder_detection/ /app/voice_disorder_detection/
COPY main.py /app/main.py
COPY server/ /app/server/

# Create persistent directories
RUN mkdir -p /app/models /app/cache /app/feedback /app/logs

# Working directory
WORKDIR /app

ENV PYTHONPATH=/app
ENV TBVOICE_DATABASE_URL=sqlite:///./server/tbvoice.db

# Declare volumes for data persistence
VOLUME ["/app/models", "/app/cache", "/root/.local/share/sbvoicedb"]

EXPOSE 8000

CMD ["uvicorn", "server.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
