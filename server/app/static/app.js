// EchoFlow 1.0 - Web Interface JavaScript

let mediaRecorder = null;
let audioChunks = [];
let recordings = {
    A: null,
    I: null,
    U: null
};
let currentVowel = null;
let recordingTimer = null;
let audioContext = null;
let analyser = null;
let animationId = null;

// Initialize audio context
async function initAudio() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(stream);
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        source.connect(analyser);
        return stream;
    } catch (error) {
        showError('Не удалось получить доступ к микрофону. Пожалуйста, разрешите доступ.');
        throw error;
    }
}

// Record vowel
async function recordVowel(vowel) {
    if (currentVowel) return; // Already recording
    
    currentVowel = vowel;
    audioChunks = [];
    
    try {
        const stream = await initAudio();
        
        // Update UI
        const card = document.getElementById(`card${vowel}`);
        const btn = document.getElementById(`btn${vowel}`);
        const status = document.getElementById(`status${vowel}`);
        const visualizer = document.getElementById(`visualizer${vowel}`);
        const timer = document.getElementById(`timer${vowel}`);
        
        card.classList.add('recording');
        status.textContent = 'Запись...';
        status.classList.remove('pending');
        status.classList.add('recording');
        btn.textContent = '⏹️ Остановить';
        btn.classList.add('recording');
        visualizer.classList.add('active');
        timer.classList.add('active');
        
        // Start recording
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            recordings[vowel] = audioBlob;
            
            // Update UI
            card.classList.remove('recording');
            card.classList.add('completed');
            status.textContent = 'Готово ✓';
            status.classList.remove('recording');
            status.classList.add('completed');
            btn.textContent = '🔄 Перезаписать';
            btn.classList.remove('recording');
            visualizer.classList.remove('active');
            timer.classList.remove('active');
            timer.textContent = '5';
            
            // Enable next vowel button
            if (vowel === 'A') {
                document.getElementById('btnI').disabled = false;
            } else if (vowel === 'I') {
                document.getElementById('btnU').disabled = false;
            }
            
            // Enable analyze button if all recorded
            if (recordings.A && recordings.I && recordings.U) {
                document.getElementById('analyzeBtn').disabled = false;
            }
            
            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());
            currentVowel = null;
            
            if (animationId) {
                cancelAnimationFrame(animationId);
            }
        };
        
        mediaRecorder.start();
        
        // Start visualizer
        visualizeAudio(visualizer);
        
        // Start 5-second timer
        let timeLeft = 5;
        timer.textContent = timeLeft;
        
        recordingTimer = setInterval(() => {
            timeLeft--;
            timer.textContent = timeLeft;
            
            if (timeLeft <= 0) {
                clearInterval(recordingTimer);
                mediaRecorder.stop();
            }
        }, 1000);
        
        // Auto-stop after 5 seconds
        setTimeout(() => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
            }
        }, 5000);
        
    } catch (error) {
        console.error('Recording error:', error);
        showError('Ошибка записи. Попробуйте еще раз.');
        currentVowel = null;
    }
}

// Visualize audio
function visualizeAudio(visualizerElement) {
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    // Clear previous bars
    visualizerElement.innerHTML = '';
    
    // Create bars
    const barCount = 40;
    const bars = [];
    for (let i = 0; i < barCount; i++) {
        const bar = document.createElement('div');
        bar.className = 'visualizer-bar';
        bar.style.left = `${(i / barCount) * 100}%`;
        visualizerElement.appendChild(bar);
        bars.push(bar);
    }
    
    function draw() {
        analyser.getByteFrequencyData(dataArray);
        
        for (let i = 0; i < barCount; i++) {
            const index = Math.floor((i / barCount) * bufferLength);
            const value = dataArray[index];
            const height = (value / 255) * 100;
            bars[i].style.height = `${height}%`;
        }
        
        animationId = requestAnimationFrame(draw);
    }
    
    draw();
}

// Analyze voice
async function analyzeVoice() {
    if (!recordings.A || !recordings.I || !recordings.U) {
        showError('Пожалуйста, запишите все три гласных звука.');
        return;
    }
    
    // Show loading
    document.getElementById('recordingSection').style.display = 'none';
    document.getElementById('loading').classList.add('show');
    document.getElementById('error').classList.remove('show');
    
    try {
        // Create form data
        const formData = new FormData();
        formData.append('audio_a', recordings.A, 'a.wav');
        formData.append('audio_i', recordings.I, 'i.wav');
        formData.append('audio_u', recordings.U, 'u.wav');
        formData.append('user_id', getUserId());
        formData.append('app_version', '1.0-web');
        formData.append('device_model', navigator.userAgent);
        
        // Send to API
        const response = await fetch('/api/v1/analyze', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Ошибка анализа');
        }
        
        const result = await response.json();
        
        // Hide loading
        document.getElementById('loading').classList.remove('show');
        
        // Show results
        displayResults(result);
        
    } catch (error) {
        console.error('Analysis error:', error);
        document.getElementById('loading').classList.remove('show');
        document.getElementById('recordingSection').style.display = 'block';
        showError(`Ошибка анализа: ${error.message}`);
    }
}

// Display results
function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    
    const verdict = data.result.verdict;
    const verdictLabel = data.result.verdict_label;
    const confidence = data.result.confidence_percent;
    const details = data.details;
    const recommendation = data.recommendation;
    
    let html = `
        <div class="verdict ${verdict}">
            <div class="verdict-label">${verdictLabel}</div>
            <div class="confidence">${confidence}%</div>
            <div>Уверенность модели</div>
        </div>
        
        <div class="result-card">
            <h3 style="margin-bottom: 20px; color: #333;">Детальный анализ</h3>
    `;
    
    // Add categories
    for (const [key, value] of Object.entries(details)) {
        const score = Math.round(value.score * 100);
        html += `
            <div class="category">
                <div class="category-name">${value.label}</div>
                <div class="category-score">
                    <div class="score-bar">
                        <div class="score-fill" style="width: ${score}%"></div>
                    </div>
                    <div class="score-value">${score}%</div>
                </div>
            </div>
        `;
    }
    
    html += `</div>`;
    
    // Add recommendation
    html += `
        <div class="recommendation">
            <div class="recommendation-title">💡 Рекомендация</div>
            <div class="recommendation-text">${recommendation}</div>
        </div>
    `;
    
    // Add new analysis button
    html += `
        <button class="analyze-button" onclick="resetAnalysis()" style="margin-top: 20px;">
            🔄 Новый анализ
        </button>
    `;
    
    resultsDiv.innerHTML = html;
    resultsDiv.classList.add('show');
}

// Reset analysis
function resetAnalysis() {
    // Reset recordings
    recordings = { A: null, I: null, U: null };
    
    // Reset UI
    ['A', 'I', 'U'].forEach(vowel => {
        const card = document.getElementById(`card${vowel}`);
        const btn = document.getElementById(`btn${vowel}`);
        const status = document.getElementById(`status${vowel}`);
        
        card.classList.remove('recording', 'completed');
        status.textContent = 'Ожидание';
        status.classList.remove('recording', 'completed');
        status.classList.add('pending');
        btn.textContent = `🎙️ Записать звук "${vowel}"`;
        btn.classList.remove('recording');
        
        if (vowel !== 'A') {
            btn.disabled = true;
        } else {
            btn.disabled = false;
        }
    });
    
    document.getElementById('analyzeBtn').disabled = true;
    document.getElementById('results').classList.remove('show');
    document.getElementById('recordingSection').style.display = 'block';
}

// Show error
function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.textContent = message;
    errorDiv.classList.add('show');
    
    setTimeout(() => {
        errorDiv.classList.remove('show');
    }, 5000);
}

// Get or create user ID
function getUserId() {
    let userId = localStorage.getItem('echoflow_user_id');
    if (!userId) {
        userId = 'web-' + Math.random().toString(36).substr(2, 9) + '-' + Date.now();
        localStorage.setItem('echoflow_user_id', userId);
    }
    return userId;
}

// Check browser compatibility
window.addEventListener('load', () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showError('Ваш браузер не поддерживает запись аудио. Используйте современный браузер (Chrome, Firefox, Safari).');
        document.querySelectorAll('.record-button').forEach(btn => btn.disabled = true);
    }
});
