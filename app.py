import time

import torch
import torchaudio
import numpy as np
import io, os
from flask import Flask, render_template
from flask_socketio import SocketIO
from collections import deque
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

from helper import predict_gender, load_model

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet", logger=True, engineio_logger=True,
                    ping_interval=25, ping_timeout=120)

# Load the gender classification model
model, feature_extractor = load_model()

# Store last 5 audio chunks for improved accuracy
audio_buffer = deque(maxlen=5)


def analyze_audio(audio_data, sample_rate=16000):
    """ Process audio and predict gender using the last 5 collected samples """
    # Convert raw bytes to numpy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

    # Add the new sample to the buffer
    audio_buffer.append(audio_array)

    # Combine the last 5 collected samples
    if len(audio_buffer) == 5:
        combined_audio = np.concatenate(list(audio_buffer))
        print(len(combined_audio))
    else:
        combined_audio = np.concatenate(list(audio_buffer))  # Use available samples

    # Normalize
    waveform = torch.tensor(combined_audio, dtype=torch.float32)
    waveform /= torch.max(torch.abs(waveform)) if torch.max(torch.abs(waveform)) > 0 else 1

    # Predict gender
    t = time.time()
    gender = predict_gender(waveform, model, feature_extractor)
    print(gender, time.time() - t)
    return f"Predicted Gender: {gender}"


@socketio.on("audio")
def handle_audio(data):
    """ Receive audio from the browser, process it, and return results """
    sample_rate = 16000  # Ensure fixed sample rate
    result = analyze_audio(data, sample_rate)
    socketio.emit("analysis", result)  # Send result back


@app.route('/')
def index():
    return render_template("index.html")


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # ✅ Update port from environment variable
    socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)
