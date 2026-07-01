#!/bin/bash
# Build script for Render deployment

echo "🚀 Starting Render build process..."

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Verify installation
echo "✅ Verifying installations..."
python -c "import fastapi; import torch; print('All core dependencies installed successfully')"

# Create necessary directories
echo "📁 Creating output directories..."
mkdir -p outputs/analysis outputs/alerts outputs/logs data

# Set permissions
chmod -R 755 outputs/ data/

echo "🎉 Build complete! Ready to start services."
