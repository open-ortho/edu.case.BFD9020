# syntax=docker/dockerfile:1.6

### Base image with system and Python deps ###
FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libopenblas-dev \
        libblas-dev \
        liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "fastai<2.8" \
    && pip install --no-cache-dir scikit-image \
    && pip install --no-cache-dir imageio \
    && pip install --no-cache-dir fastapi[standard]

### Stage dedicated to heavyweight model artifacts ###
FROM scratch AS model_assets
COPY models/ /models

### Final runtime image ###
FROM base AS runtime
WORKDIR /app

COPY main.py ./
COPY README.md ./
COPY dev_scratchpad.md ./dev_scratchpad.md
COPY BFD9020.html ./
COPY static ./static
COPY --from=model_assets /models ./models

EXPOSE 9020

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9020"]