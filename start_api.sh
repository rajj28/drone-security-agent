#!/bin/sh
# API Startup Script - handles PORT env var properly

PORT=${PORT:-8000}
echo "🚀 Starting API on port $PORT"

exec uvicorn src.api:app --host 0.0.0.0 --port $PORT --workers 1
