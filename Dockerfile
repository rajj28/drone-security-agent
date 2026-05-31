FROM python:3.10-slim

# Install ffmpeg and system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the API using shell to expand PORT environment variable
SHELL ["/bin/sh", "-c"]
CMD uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
