#!/bin/bash

# Kids Facts Video Pipeline - Startup Script

echo "🎬 Kids Facts Video Pipeline"
echo "============================"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check for FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  FFmpeg is not installed. Video processing will fail."
    echo "   Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
fi

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt -q

# Check for FAL_KEY
if [ -z "$FAL_KEY" ]; then
    # Check config.json
    if grep -q "YOUR_FAL_API_KEY_HERE" config.json; then
        echo ""
        echo "⚠️  Fal API key not configured!"
        echo "   Either:"
        echo "   1. Edit config.json and replace YOUR_FAL_API_KEY_HERE"
        echo "   2. Set FAL_KEY environment variable"
        echo ""
    fi
fi

# Create required directories
mkdir -p uploads clips outputs sfx temp

echo ""
echo "🚀 Starting server..."
echo "   Open http://localhost:5000 in your browser"
echo ""

python server.py
