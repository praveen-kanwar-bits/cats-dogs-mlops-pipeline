FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY requirements-serving.txt ./
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements-serving.txt

COPY src ./src
COPY params.yaml ./params.yaml
COPY artifacts/model ./artifacts/model

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD python -c "import requests; requests.get('http://127.0.0.1:8000/health', timeout=3).raise_for_status()"

CMD ["uvicorn", "cats_dogs_mlops.interfaces.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
