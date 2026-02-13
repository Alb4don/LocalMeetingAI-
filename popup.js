const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const status = document.getElementById('status');
const statusText = status.querySelector('.status-text');
const feed = document.getElementById('feed');
const reportDiv = document.getElementById('report');
const errorMessage = document.getElementById('errorMessage');
const statsBar = document.getElementById('statsBar');
const transcriptCount = document.getElementById('transcriptCount');
const sentimentPos = document.getElementById('sentimentPos');
const sentimentNeg = document.getElementById('sentimentNeg');

let stats = {
    total: 0,
    positive: 0,
    negative: 0,
    neutral: 0
};

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    setTimeout(() => {
        errorMessage.style.display = 'none';
    }, 5000);
}

function updateStats() {
    transcriptCount.textContent = stats.total;
    sentimentPos.textContent = stats.positive;
    sentimentNeg.textContent = stats.negative;
}

function resetStats() {
    stats = { total: 0, positive: 0, negative: 0, neutral: 0 };
    updateStats();
}

function formatTime() {
    const now = new Date();
    return now.toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

startBtn.addEventListener('click', () => {
    statusText.textContent = "Iniciando...";
    errorMessage.style.display = 'none';
    
    chrome.runtime.sendMessage({ type: 'START', target: 'background' }, (res) => {
        if (chrome.runtime.lastError) {
            statusText.textContent = "Erro de conexão";
            showError('Erro ao conectar com o servidor. Verifique se o servidor está rodando.');
            return;
        }

        if (!res) {
            statusText.textContent = "Erro";
            showError('Sem resposta do background script');
            return;
        }

        if (res.status === 'started') {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            feed.innerHTML = '';
            reportDiv.style.display = 'none';
            statsBar.style.display = 'flex';
            resetStats();
            statusText.textContent = "Capturando...";
            
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'loading';
            loadingDiv.textContent = 'Aguardando conexão com servidor';
            feed.appendChild(loadingDiv);
        } else if (res.status === 'error') {
            showError(res.message);
            statusText.textContent = "Erro ao iniciar";
        }
    });
});

stopBtn.addEventListener('click', () => {
    chrome.runtime.sendMessage({ type: 'STOP', target: 'background' });
    startBtn.disabled = false;
    stopBtn.disabled = true;
    
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading';
    loadingDiv.innerHTML = '<strong>Gerando relatório via IA local</strong><br>Isso pode levar alguns segundos';
    feed.appendChild(loadingDiv);
    feed.scrollTop = feed.scrollHeight;
});

chrome.runtime.onMessage.addListener((msg) => {
    if (msg.target !== 'popup') return;

    if (msg.type === 'WS_STATE') {
        if (msg.state === 'CONNECTED') {
            statusText.textContent = 'Conectado (Localhost)';
            status.classList.add('connected');
            
            const loadingElements = feed.querySelectorAll('.loading');
            loadingElements.forEach(el => el.remove());
            
            if (feed.children.length === 0) {
                const infoDiv = document.createElement('div');
                infoDiv.className = 'empty-state';
                infoDiv.innerHTML = '<div style="color: #10b981;">✓ Conectado ao servidor</div><div style="font-size: 12px; margin-top: 8px;">Aguardando áudio...</div>';
                feed.appendChild(infoDiv);
            }
        } else {
            statusText.textContent = 'Desconectado';
            status.classList.remove('connected');
            showError('Conexão perdida com o servidor. Tentando reconectar...');
        }
    }

    if (msg.type === 'ERROR') {
        showError(msg.message);
    }

    if (msg.type === 'DATA') {
        const payload = msg.payload;

        if (payload.type === 'realtime') {
            const emptyStates = feed.querySelectorAll('.empty-state, .loading');
            emptyStates.forEach(el => el.remove());
            
            const div = document.createElement('div');
            div.className = 'item';
            
            let sentimentClass = 'neu';
            let sentimentLabel = 'Neutro';
            
            if (payload.sentiment === 'Positivo') {
                sentimentClass = 'pos';
                sentimentLabel = 'Positivo';
                stats.positive++;
            } else if (payload.sentiment === 'Negativo') {
                sentimentClass = 'neg';
                sentimentLabel = 'Negativo';
                stats.negative++;
            } else {
                stats.neutral++;
            }
            
            stats.total++;
            updateStats();
            
            div.className = `item ${sentimentClass}`;
            
            div.innerHTML = `
                <div class="meta">
                    <span class="timestamp">${formatTime()}</span>
                    <span class="sentiment-tag">${sentimentLabel}</span>
                </div>
                <div class="text">${payload.text}</div>
            `;
            
            feed.appendChild(div);
            feed.scrollTop = feed.scrollHeight;
            
            if (feed.children.length > 50) {
                feed.removeChild(feed.firstChild);
            }
        }

        if (payload.type === 'report') {
            feed.style.display = 'none';
            reportDiv.style.display = 'block';
            reportDiv.textContent = payload.content;
            statusText.textContent = 'Relatório Gerado';
            statsBar.style.display = 'none';
        }
    }
});

chrome.runtime.sendMessage({ type: 'GET_STATUS', target: 'background' }, (response) => {
    if (chrome.runtime.lastError) {
        return;
    }
    
    if (response && response.isCapturing) {
        startBtn.disabled = true;
        stopBtn.disabled = false;
        statusText.textContent = 'Capturando...';
    }
});