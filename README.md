# Prerequisites

- Python 3.9 or higher
- Chrome/Chromium browser
- 8GB RAM minimum (16GB recommended)
- Optional: CUDA-compatible GPU for faster processing

# Installation

- Install Python dependencies:

      pip install -r requirements.txt

- Open Chrome and navigate to chrome://extensions/
- Enable Developer mode (top right)
- Click Load unpacked
- Select the extension directory

# Usage

- Start the Server (windows)

        start_server.bat

- Linux

        ./start_server.sh

Or manually:

        python server.py

- The server will start on ***http://127.0.0.1:8000***

- Navigate to a tab with audio (Google Meet, Zoom, YouTube, etc.)
- Click the extension icon
- Click Iniciar Captura
- Watch real-time transcriptions with sentiment analysis
- Click Finalizar to generate a comprehensive AI report
  
![LocalMeet](https://github.com/user-attachments/assets/37617000-7ab6-4374-9219-b3760a22f4f7)

# Future implementations

- [ ] Speaker diarization
- [ ] Multi-language support enhancement
- [ ] Custom vocabulary injection
- [ ] Real-time translation
- [ ] Meeting templates
