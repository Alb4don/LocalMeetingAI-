#!/bin/bash

echo "======================================="
echo "  LocalMeeting AI - Advanced Edition"
echo "======================================="
echo ""

if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    exit 1
fi

echo "Checking dependencies..."
python3 -c "import fastapi, uvicorn, faster_whisper, transformers, torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: Some dependencies are missing"
    echo "Run: pip install -r requirements.txt"
    echo ""
    exit 1
fi

echo "All dependencies OK!"
echo "Starting server on http://127.0.0.1:8000"
echo ""
echo "Advanced AI Features:"
echo "- Keyword Extraction (TF-IDF)"
echo "- Action Item Detection"
echo "- Emotion Analysis"
echo "- Smart Summarization (BART)"
echo "- Sentiment Trending"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python3 server.py
