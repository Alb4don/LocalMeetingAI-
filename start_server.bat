@echo off
echo =======================================
echo   LocalMeeting AI - Advanced Edition
echo =======================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Checking dependencies...
python -c "import fastapi, uvicorn, faster_whisper, transformers, torch" >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Some dependencies are missing
    echo Run: pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo All dependencies OK!
echo Starting server on http://127.0.0.1:8000
echo.
echo Advanced AI Features:
echo - Keyword Extraction (TF-IDF)
echo - Action Item Detection
echo - Emotion Analysis
echo - Smart Summarization (BART)
echo - Sentiment Trending
echo.
echo Press Ctrl+C to stop
echo.

python server.py
pause
