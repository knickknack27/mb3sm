document.addEventListener('DOMContentLoaded', () => {
    const micButton = document.getElementById('micButton');
    const statusDiv = document.getElementById('status');
    const transcriptionDiv = document.getElementById('transcription');
    const assistantResponseDiv = document.getElementById('assistantResponse');
    const conversationLogDiv = document.getElementById('conversationLog');

    let mediaRecorder;
    let audioChunks = [];
    let isSessionActive = false;
    let isRecording = false;
    let audioContext, analyser, sourceNode, silenceTimer, stream;
    const BACKEND_URL = 'http://localhost:5001/api/transcribe-and-chat'; // Ensure this matches your Python backend port

    // --- UI Update Functions ---
    function updateStatus(message) {
        statusDiv.textContent = `Status: ${message}`;
    }

    function displayTranscription(text) {
        transcriptionDiv.innerHTML = `<strong>You:</strong> ${text}`;
    }

    function displayAssistantResponse(text) {
        assistantResponseDiv.innerHTML = `<strong>Assistant:</strong> ${text}`;
    }

    function addToConversationLog(role, text) {
        const entry = document.createElement('div');
        entry.classList.add('log-entry', role);
        entry.innerHTML = `<strong>${role === 'user' ? 'You' : 'Assistant'}:</strong> ${text}`;
        conversationLogDiv.appendChild(entry);
        conversationLogDiv.scrollTop = conversationLogDiv.scrollHeight; // Scroll to bottom
    }

    function clearCurrentInteractionDisplays() {
        transcriptionDiv.innerHTML = '';
        assistantResponseDiv.innerHTML = '';
    }

    // --- Silence Detection ---
    function startSilenceDetection(stream) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        sourceNode = audioContext.createMediaStreamSource(stream);
        analyser = audioContext.createAnalyser();
        sourceNode.connect(analyser);
        analyser.fftSize = 2048;
        const dataArray = new Uint8Array(analyser.fftSize);
        let silenceStart = null;
        const SILENCE_THRESHOLD = 0.03; // Lower threshold for more sensitivity
        const SILENCE_DURATION = 2000; // 1 second of silence to auto-stop

        function checkSilence() {
            analyser.getByteTimeDomainData(dataArray);
            let sum = 0;
            for (let i = 0; i < dataArray.length; i++) {
                let normalized = (dataArray[i] - 128) / 128;
                sum += normalized * normalized;
            }
            let volume = Math.sqrt(sum / dataArray.length);
            if (volume < SILENCE_THRESHOLD) {
                if (!silenceStart) silenceStart = Date.now();
                if (Date.now() - silenceStart > SILENCE_DURATION) {
                    stopRecording();
                    return;
                }
            } else {
                silenceStart = null;
            }
            if (isRecording) {
                silenceTimer = setTimeout(checkSilence, 100);
            }
        }
        checkSilence();
    }

    function stopSilenceDetection() {
        if (silenceTimer) clearTimeout(silenceTimer);
        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }
        analyser = null;
        sourceNode = null;
    }

    // --- Media Recorder Functions ---
    async function startRecordingWithSilenceDetection() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            updateStatus('getUserMedia not supported on your browser!');
            alert('Your browser does not support audio recording.');
            return;
        }
        try {
            stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = async () => {
                stopSilenceDetection();
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                audioChunks = [];
                sendAudioToBackend(audioBlob);
                // Stop all tracks in the stream to turn off the mic light/indicator
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start();
            isRecording = true;
            micButton.textContent = '🎤 Mic Off (Listening...)';
            micButton.classList.add('recording');
            updateStatus('Listening... Speak now.');
            clearCurrentInteractionDisplays();
            startSilenceDetection(stream);
        } catch (err) {
            console.error('Error accessing microphone:', err);
            updateStatus(`Error accessing microphone: ${err.message}`);
            alert(`Error accessing microphone: ${err.message}. Please ensure permission is granted.`);
            isRecording = false;
            micButton.textContent = '🎤 Mic On';
            micButton.classList.remove('recording');
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
            isRecording = false;
            micButton.textContent = '🎤 Mic On';
            micButton.classList.remove('recording');
            updateStatus('Processing audio...');
        }
    }

    // --- Backend Communication ---
    async function sendAudioToBackend(audioBlob) {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');

        updateStatus('Raj is thinking...');
        transcriptionDiv.innerHTML = '';
        assistantResponseDiv.innerHTML = '';

        try {
            const response = await fetch(BACKEND_URL, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: 'Unknown error from backend' , details: response.statusText}));
                throw new Error(`Backend error: ${response.status} ${response.statusText}. Details: ${JSON.stringify(errorData.details || errorData.error)}`);
            }

            const result = await response.json();

            displayTranscription(result.userTranscript);
            displayAssistantResponse(result.assistantReply);
            addToConversationLog('user', result.translatedTranscript);
            addToConversationLog('assistant', result.assistantReply);
            updateStatus('Playing assistant reply...');

            // Play Sarvam TTS audio if available as base64
            if (result.audioBase64) {
                let audio = new Audio('data:audio/wav;base64,' + result.audioBase64);
                updateStatus('Playing assistant reply...');
                micButton.textContent = '🔈 Assistant Speaking';
                micButton.classList.remove('recording');
                micButton.disabled = true; // Disable mic button while speaking
                audio.onended = () => {
                    micButton.disabled = false; // Re-enable mic button after speaking
                    if (isSessionActive) {
                        micButton.textContent = '🎤 Mic Off (Listening...)';
                        micButton.classList.add('recording');
                        updateStatus('Listening... Speak now.');
                        startRecordingWithSilenceDetection();
                    }
                };
                audio.play();
            } else {
                // If no audio, auto-restart listening
                if (isSessionActive) {
                    setTimeout(() => startRecordingWithSilenceDetection(), 1000);
                }
            }
        } catch (err) {
            console.error('Error sending audio to backend:', err);
            updateStatus(`Error: ${err.message}`);
            assistantResponseDiv.innerHTML = `<strong style="color:red;">Error:</strong> ${err.message}`;
            // Auto-restart listening if session is active
            if (isSessionActive) {
                setTimeout(() => startRecordingWithSilenceDetection(), 2000);
            }
        }
    }

    // --- Event Listeners ---
    micButton.addEventListener('click', () => {
        if (isSessionActive) {
            // Stop the session
            isSessionActive = false;
            if (isRecording) stopRecording();
            micButton.textContent = '🎤 Mic On';
            micButton.classList.remove('recording');
            updateStatus('Session stopped. Click Mic On to start again.');
        } else {
            // Start the session
            isSessionActive = true;
            startRecordingWithSilenceDetection();
            updateStatus('Listening... Speak now.');
        }
    });

    updateStatus('Ready. Click Mic On to start.');
}); 
