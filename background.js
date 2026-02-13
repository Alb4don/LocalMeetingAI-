let isCapturing = false;
let captureTabId = null;

async function createOffscreen() {
    try {
        if (await chrome.offscreen.hasDocument()) return;
        await chrome.offscreen.createDocument({
            url: 'offscreen.html',
            reasons: ['USER_MEDIA'],
            justification: 'Capturing tab audio for local analysis'
        });
    } catch (err) {
        console.error("Offscreen creation error:", err);
    }
}

async function closeOffscreen() {
    try {
        if (await chrome.offscreen.hasDocument()) {
            await chrome.offscreen.closeDocument();
        }
    } catch (err) {
        console.error("Offscreen close error:", err);
    }
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.target !== 'background') return;

    if (msg.type === 'START') {
        if (isCapturing) {
            sendResponse({ status: 'error', message: 'Already capturing.' });
            return true;
        }

        chrome.tabCapture.getMediaStreamId({ consumerTabId: sender.tab?.id }, async (streamId) => {
            if (chrome.runtime.lastError) {
                console.warn("Capture failed:", chrome.runtime.lastError.message);
                sendResponse({ status: 'error', message: 'Tab is already being captured. Please refresh the page and try again.' });
                return;
            }

            if (streamId) {
                try {
                    await createOffscreen();
                    
                    setTimeout(() => {
                        chrome.runtime.sendMessage({
                            target: 'offscreen',
                            type: 'INIT_STREAM',
                            streamId: streamId
                        }).catch(err => {
                            console.error("Message send error:", err);
                        });
                    }, 100);
                    
                    isCapturing = true;
                    captureTabId = sender.tab?.id;
                    sendResponse({ status: 'started' });
                } catch (err) {
                    console.error("Error starting capture:", err);
                    sendResponse({ status: 'error', message: err.message });
                }
            } else {
                sendResponse({ status: 'error', message: 'No stream ID generated.' });
            }
        });
        return true;
    }

    if (msg.type === 'STOP') {
        if (isCapturing) {
            setTimeout(() => {
                chrome.runtime.sendMessage({ target: 'offscreen', type: 'STOP_STREAM' }).catch(() => {});
            }, 100);
            
            setTimeout(async () => {
                isCapturing = false;
                captureTabId = null;
            }, 1000);
        }
        sendResponse({ status: 'stopped' });
        return true;
    }
    
    if (msg.type === 'RESET') {
        closeOffscreen();
        isCapturing = false;
        captureTabId = null;
        sendResponse({ status: 'reset' });
        return true;
    }

    if (msg.type === 'GET_STATUS') {
        sendResponse({ isCapturing, captureTabId });
        return true;
    }
});

chrome.tabs.onRemoved.addListener((tabId) => {
    if (tabId === captureTabId) {
        chrome.runtime.sendMessage({ target: 'offscreen', type: 'STOP_STREAM' }).catch(() => {});
        isCapturing = false;
        captureTabId = null;
    }
});