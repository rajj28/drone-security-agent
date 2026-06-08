FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.light.txt .
RUN pip install --no-cache-dir -r requirements.light.txt

# Copy app
COPY src/ ./src/

EXPOSE 8000
CMD uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
