let socket = null;
let mediaRecorder = null;
let audioContext = null;
let stream = null;
let reconnectInterval = null;
let isReconnecting = false;
let messageQueue = [];
let isProcessing = false;

function sendToPopup(message) {
    chrome.runtime.sendMessage(message).catch(() => {});
}

function initWebSocket() {
    if (socket && socket.readyState === WebSocket.OPEN) return;
    if (isReconnecting) return;
    
    socket = new WebSocket('ws://127.0.0.1:8000/ws/meeting');
    
    socket.onopen = () => {
        sendToPopup({ target: 'popup', type: 'WS_STATE', state: 'CONNECTED' });
        isReconnecting = false;
        if (reconnectInterval) {
            clearInterval(reconnectInterval);
            reconnectInterval = null;
        }
    };

    socket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            sendToPopup({ target: 'popup', type: 'DATA', payload: data });
        } catch (e) {
            console.error("Message parse error:", e);
        }
    };
    
    socket.onerror = () => {
        sendToPopup({ target: 'popup', type: 'ERROR', message: 'Connection error. Make sure server is running on port 8000.' });
    };
    
    socket.onclose = () => {
        sendToPopup({ target: 'popup', type: 'WS_STATE', state: 'DISCONNECTED' });
        if (!isReconnecting && stream) {
            attemptReconnect();
        }
    };
}

function attemptReconnect() {
    isReconnecting = true;
    reconnectInterval = setInterval(() => {
        if (!socket || socket.readyState === WebSocket.CLOSED) {
            initWebSocket();
        }
    }, 3000);
}

async function startCapture(streamId) {
    initWebSocket();

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                mandatory: {
                    chromeMediaSource: 'tab',
                    chromeMediaSourceId: streamId
                }
            },
            video: false
        });

        audioContext = new AudioContext({ sampleRate: 16000 });
        const source = audioContext.createMediaStreamSource(stream);
        const dest = audioContext.createMediaStreamDestination();
        source.connect(dest);

        mediaRecorder = new MediaRecorder(dest.stream, { 
            mimeType: 'audio/webm;codecs=opus',
            audioBitsPerSecond: 128000
        });
        
        mediaRecorder.ondataavailable = async (e) => {
            if (e.data.size > 0 && socket && socket.readyState === WebSocket.OPEN) {
                try {
                    const buffer = await e.data.arrayBuffer();
                    socket.send(buffer);
                } catch (err) {
                    console.error("Send error:", err);
                }
            }
        };

        mediaRecorder.onerror = () => {
            sendToPopup({ target: 'popup', type: 'ERROR', message: 'Recording error occurred.' });
        };

        mediaRecorder.start(1000);

    } catch (e) {
        console.error("Capture error:", e);
        sendToPopup({ target: 'popup', type: 'ERROR', message: e.message });
    }
}

function stopCapture() {
    if (reconnectInterval) {
        clearInterval(reconnectInterval);
        reconnectInterval = null;
    }
    
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
        stream = null;
    }
    
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
    
    if (socket && socket.readyState === WebSocket.OPEN) {
        try {
            socket.send(JSON.stringify({ command: 'generate_report' }));
            setTimeout(() => {
                if (socket) {
                    socket.close();
                    socket = null;
                }
            }, 5000);
        } catch (e) {
            console.error("Stop command error:", e);
        }
    }
}

chrome.runtime.onMessage.addListener((msg) => {
    if (msg.target === 'offscreen') {
        if (msg.type === 'INIT_STREAM') startCapture(msg.streamId);
        if (msg.type === 'STOP_STREAM') stopCapture();
    }
});