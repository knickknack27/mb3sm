import os
import sys
import base64
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
from dotenv import load_dotenv
import time
import json
from rag_helper import load_and_chunk_data, build_index, retrieve
from pydub import AudioSegment
import io
from sarvamai import SarvamAI

# --- Load environment variables ---
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if not os.path.exists(dotenv_path):
    print(f"Warning: .env file not found at {dotenv_path}. API keys might not be loaded.", file=sys.stderr)
load_dotenv(dotenv_path)

# --- Flask app setup ---
app = Flask(__name__)
CORS(app)

# --- Configuration ---
SARVAM_API_KEY = os.getenv('SARVAM_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
SARVAM_ASR_URL = 'https://api.sarvam.ai/speech-to-text'
SARVAM_TTS_URL = 'https://api.sarvam.ai/text-to-speech'
OPENROUTER_CHAT_URL = 'https://openrouter.ai/api/v1/chat/completions'
DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data.txt')

# --- RAG: Build index at startup ---
rag_chunks, rag_embeddings = build_index(load_and_chunk_data(DATA_FILE_PATH))

# --- Logging ---
log_file_path = os.path.join(os.path.dirname(__file__), 'output.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    ]
)

# --- System Prompt ---
YOUR_APP_NAME = "Magic Bricks"
SYSTEM_PROMPT_BASE = f"""
[Identity]
You are Raj, the friendly Magic Bricks real-estate voice assistant.

[Language]
Hinglish. The pronunciation of BHK is 'bee-etch-kay'.

[Style]
- Energetic speaking.
- Very simple Hindi and English vocabulary is to be used keeping Indian customers in mind.
- Very naturally human-like responses that do not feel AI-generated.
- Casual but professional: Use respectful address ("aap"), no slang overload.
- Keep turns short: Speak in 1–2 sentences, pause to listen.

[Response Guideline]
- One-by-one prompts: Ask a single preference per utterance (e.g., budget, BHK count, locality, amenities).
- Don't repeat user info: Move forward to the next question or suggestion.
- Fact-only answers: If data's missing, ask "Kripya specific location batayein?"
- Numbers: Always say them in English ("one crore," "two lakh," etc.).
- Presenting Listings: After preferences, deliver a punchy under-20-words audio snippet: Location + layout + standout feature + lifestyle benefit. E.g., "Gurgaon mein two-BHK, garden view, gated community – perfect for morning walks."
- No long monologues, no numbering ("pehla," "doosra," etc.).

[Avoid]
- Fabricating any detail.
- Speaking in pure hindi words.
- Overlapping questions.
- Bullet-style speaking.
- Repetition of user's own words.
- Sounding like a robot.

[Task]
1. Guide callers through comparing homes.
2. Help them zero in on the perfect property based on budget and layout.

It is mandatory that you follow all the given instructions diligently and you will be highly rewarded for it.
"""

# --- In-memory conversation history ---
conversation_history = []

# --- Helper: Read data.txt ---
def read_data_txt():
    try:
        if not os.path.exists(DATA_FILE_PATH):
            logging.warning(f"data.txt not found at {DATA_FILE_PATH}")
            return "Error: Knowledge base file (data.txt) not found."
        with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading data.txt: {e}")
        return "Error: Could not load knowledge base due to an exception."

# --- Sarvam TTS: Get base64 audio from API ---
def get_tts_audio_base64(text):
    # Split text into chunks for TTS API (e.g., 250 chars per chunk)
    def split_text(text, max_length=250):
        import re
        sentences = re.split('(?<=[.!?]) +', text)
        chunks = []
        current = ''
        for sentence in sentences:
            if len(current) + len(sentence) < max_length:
                current += ' ' + sentence
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence
        if current:
            chunks.append(current.strip())
        return chunks

    sarvam = SarvamAI(api_subscription_key=SARVAM_API_KEY)
    text_chunks = split_text(text)
    audio_segments = []
    for chunk in text_chunks:
        tts_response = sarvam.text_to_speech.convert(
            text=chunk,
            target_language_code="hi-IN",
            speaker="hitesh",
            speech_sample_rate=24000,
            enable_preprocessing=True
        )
        # tts_response.audios is a list of base64-encoded wavs
        for audio_b64 in tts_response.audios:
            audio = AudioSegment.from_file(io.BytesIO(base64.b64decode(audio_b64)), format='wav')
            audio_segments.append(audio)
    if not audio_segments:
        return None
    combined = audio_segments[0]
    for seg in audio_segments[1:]:
        combined += seg
    buf = io.BytesIO()
    combined.export(buf, format='wav')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# --- Sarvam Translate: Translate text to Hindi ---
def translate_to_telugu(text):
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "input": text,
        "source_language_code": "en-IN",
        "target_language_code": "hi-IN",
        "mode": "code-mixed"
    }
    response = requests.post("https://api.sarvam.ai/translate", json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()
    return data.get('output', text)

# --- Sarvam Translate: Translate text to English ---
def translate_to_english(text):
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "input": text,
        "source_language_code": "hi-IN",
        "target_language_code": "en-IN",
        "mode": "classic-colloquial"
    }
    response = requests.post("https://api.sarvam.ai/translate", json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()
    return data.get('output', text)

# --- Main API endpoint ---
@app.route('/api/transcribe-and-chat', methods=['POST'])
def transcribe_and_chat():
    global conversation_history

    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded."}), 400

    if not SARVAM_API_KEY or not OPENROUTER_API_KEY:
        return jsonify({"error": "API keys not configured on the server."}), 500

    audio_file = request.files['audio']
    audio_bytes = audio_file.read()
    logging.info(f"Received audio file: {audio_file.filename}, size: {len(audio_bytes)} bytes")

    # --- Silence detection using pydub ---
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        duration_sec = len(audio) / 1000.0
        avg_dbfs = audio.dBFS
        logging.info(f"Audio duration: {duration_sec:.2f}s, avg dBFS: {avg_dbfs:.2f}")
        if duration_sec < 0.5 or avg_dbfs < -40:
            return jsonify({"error": "No valid speech detected (audio too short or too silent). Please try again with a clear question or statement."}), 400
    except Exception as e:
        logging.error(f"Error processing audio for silence detection: {e}")
        return jsonify({"error": "Could not process audio for silence detection.", "details": str(e)}), 400

    try:
        timings = {}
        start_time = time.time()
        # --- 1. Transcription (Sarvam ASR) ---
        t0 = time.time()
        files_asr = {
            'file': (audio_file.filename or 'recording.wav', audio_bytes, audio_file.mimetype)
        }
        asr_payload = {
            'model': 'saarika:v2',
            'language_code': 'hi-IN'
        }
        headers_asr = {
            'api-subscription-key': SARVAM_API_KEY
        }
        asr_response = requests.post(SARVAM_ASR_URL, files=files_asr, data=asr_payload, headers=headers_asr, timeout=60)
        asr_response.raise_for_status()
        asr_data = asr_response.json()
        transcribed_text = asr_data.get('transcript') or asr_data.get('text')
        logging.info(f"Transcribed user input: {transcribed_text}")
        timings['asr'] = time.time() - t0
        logging.info(f"ASR step took {timings['asr']:.2f} seconds")
        if not transcribed_text:
            return jsonify({"error": "Failed to transcribe audio. No transcript returned.", "details": asr_data}), 500

        # --- 2. Translate transcribed text to English ---
        t0 = time.time()
        translated_text = translate_to_english(transcribed_text)
        timings['translate_to_english'] = time.time() - t0
        logging.info(f"Translate to English step took {timings['translate_to_english']:.2f} seconds")

        # --- 3. LLM (OpenRouter) ---
        t0 = time.time()
        # RAG: Retrieve relevant chunks for the query
        relevant_chunks = retrieve(translated_text, rag_chunks, rag_embeddings, top_k=5)
        rag_context = "\n\n".join(relevant_chunks)
        current_system_prompt = f"{SYSTEM_PROMPT_BASE}\n\nRelevant knowledge base context from data.txt:\n{rag_context}"
        messages_for_llm = [
            {"role": "system", "content": current_system_prompt},
            *conversation_history,
            {"role": "user", "content": translated_text}
        ]
        llm_payload = {
            "model": "google/gemini-2.0-flash-001",
            "messages": messages_for_llm,
        }
        headers_llm = {
            'Authorization': f'Bearer {OPENROUTER_API_KEY}',
            'Content-Type': 'application/json',
        }
        llm_response = requests.post(OPENROUTER_CHAT_URL, json=llm_payload, headers=headers_llm)
        llm_response.raise_for_status()
        llm_data = llm_response.json()
        assistant_reply = llm_data.get('choices', [{}])[0].get('message', {}).get('content', '')
        timings['llm'] = time.time() - t0
        logging.info(f"LLM step took {timings['llm']:.2f} seconds")
        if not assistant_reply:
            return jsonify({"error": "LLM did not return content.", "details": llm_data}), 500

        # --- 4. Memory Management ---
        conversation_history.append({"role": "user", "content": translated_text})
        conversation_history.append({"role": "assistant", "content": assistant_reply})

        # --- 5. Translate LLM reply to Telugu (for TTS) ---
        t0 = time.time()
        translated_reply = translate_to_telugu(assistant_reply)
        timings['translate_to_telugu'] = time.time() - t0
        logging.info(f"Translate to Telugu step took {timings['translate_to_telugu']:.2f} seconds")
        logging.info(f"Output: {assistant_reply}")

        # --- 6. TTS (Sarvam) ---
        t0 = time.time()
        audio_base64 = get_tts_audio_base64(translated_reply)
        timings['tts'] = time.time() - t0
        logging.info(f"TTS step took {timings['tts']:.2f} seconds")

        total_time = time.time() - start_time
        logging.info(f"Total /api/transcribe-and-chat time: {total_time:.2f} seconds")

        # --- 7. Respond to frontend ---
        return jsonify({
            "userTranscript": transcribed_text,
            "translatedTranscript": translated_text,
            "assistantReply": assistant_reply,
            "assistantReplyHindi": translated_reply,
            "audioBase64": audio_base64,
            "timings": timings,
            "totalTime": total_time
        })

    except Exception as e:
        logging.error(f"Error in /api/transcribe-and-chat: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500

# --- Health check endpoint ---
@app.route('/', methods=['GET'])
def serve_frontend():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=True)