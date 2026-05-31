FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    || apt-get install -y \
    gcc \
    python3-dev \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY demo/ ./demo/
COPY .env.example .env

# Create necessary directories
RUN mkdir -p data/frames data/extracted outputs/telemetry outputs/analysis outputs/alerts outputs/index outputs/session

# Expose ports
EXPOSE 8000 8501

# Default command - start the API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
