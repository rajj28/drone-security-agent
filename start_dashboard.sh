#!/bin/sh
# Dashboard Startup Script - handles PORT env var properly

PORT=${PORT:-8501}
echo "📊 Starting Dashboard on port $PORT"

exec streamlit run demo/dashboard.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
