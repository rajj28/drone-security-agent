# Stage 1: Use pre-built frontend (already in frontend/dist/)
# Stage 2: Final Python image
FROM python:3.10-slim
WORKDIR /app

# Install curl for healthcheck + ffmpeg for video processing
RUN apt-get update && apt-get install -y curl ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy requirements
# Pin Pinecone 9.x for integrated inference search/upsert_records API
COPY requirements.light.txt .
RUN pip install --no-cache-dir -r requirements.light.txt

# Copy app and built frontend
COPY src/ ./src/
COPY frontend/dist/ ./frontend/dist/

# Cloud Run requires port 8080
ENV PORT=8080
EXPOSE 8080

# Start with explicit host/port for Cloud Run
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
