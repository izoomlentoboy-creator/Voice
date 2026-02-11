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

# Copy ML package + server
COPY voice_disorder_detection/ /app/voice_disorder_detection/
COPY server/ /app/server/

# Create models directory (trained models will be mounted via volume)
RUN mkdir -p /app/models

# Working directory
WORKDIR /app

ENV PYTHONPATH=/app
ENV TBVOICE_DATABASE_URL=sqlite:///./server/tbvoice.db

EXPOSE 8000

CMD ["uvicorn", "server.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
