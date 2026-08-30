FROM python:3.11-slim

ARG VCS_REF=unknown
ARG BUILD_DATE=unknown

LABEL org.opencontainers.image.title="Cats vs Dogs MLOps API" \
      org.opencontainers.image.description="FastAPI service for the evaluated Cats vs Dogs baseline" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.source="https://github.com/praveen-kanwar-bits/cats-dogs-mlops-pipeline"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY requirements-serving.txt ./
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY src ./src
COPY params.yaml ./params.yaml
COPY artifacts/release/cats_dogs_cnn.pt ./artifacts/model/cats_dogs_cnn.pt
COPY artifacts/release/metadata.json ./artifacts/model/metadata.json
COPY artifacts/release/SHA256SUMS /tmp/SHA256SUMS

# A published image must contain the evaluated release checkpoint byte-for-byte.
RUN cd artifacts/model && sha256sum -c /tmp/SHA256SUMS && rm /tmp/SHA256SUMS

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=5 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready')"

CMD ["uvicorn", "cats_dogs_mlops.interfaces.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
