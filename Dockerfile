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

# Copy and make startup script executable
COPY start_api.sh /app/start_api.sh
RUN chmod +x /app/start_api.sh

# Run the API using startup script
CMD ["/app/start_api.sh"]
