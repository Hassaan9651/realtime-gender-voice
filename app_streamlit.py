# import streamlit as st
# from stream_helper import *
# import time
# import numpy as np
#
# st.title("Gender by Voice")
#
#
# model, feature_extractor = load_model()
#
# def start_audio_stream():
#     p = pyaudio.PyAudio()
#     stream = p.open(
#         format=FORMAT,
#         channels=CHANNELS,
#         rate=RATE,
#         input=True,
#         frames_per_buffer=CHUNK
#     )
#     return p, stream
#
#
# if 'recording' not in st.session_state:
#     st.session_state.recording = False
# if 'gender_label' not in st.session_state:
#     st.session_state.gender_label = "Unknown"
#
#
# def toggle_recording():
#     st.session_state.recording = not st.session_state.recording
#     if st.session_state.recording:
#         p, stream = start_audio_stream()
#         st.session_state.p = p
#         st.session_state.stream = stream
#     else:
#         if 'stream' in st.session_state:
#             st.session_state.stream.stop_stream()
#             st.session_state.stream.close()
#             st.session_state.p.terminate()
#         # st.session_state.gender_label = "Unknown"
#     st.rerun()
#
# reset = 0
# def process_audio():
#     if 'stream' in st.session_state and st.session_state.recording:
#         frames = []
#         duration = 0.3
#         for _ in range(0, int(RATE / CHUNK * duration)):
#             data = st.session_state.stream.read(CHUNK, exception_on_overflow=False)
#             frames.append(data)
#
#         waveform_np = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32)
#         if waveform_np.size != 0:
#             waveform = torch.tensor(waveform_np)
#             waveform /= torch.max(torch.abs(waveform)) if torch.max(torch.abs(waveform)) > 0 else 1
#             # t = time.time()
#             st.session_state.gender_label = predict_gender(waveform, model, feature_extractor)
#             # print(time.time() - t)
#             color1 = "red" if st.session_state.gender_label=="Male" else ("yellow" if st.session_state.gender_label=="None" else "green")
#             st.markdown(f"<h1 style='text-align: center; color: {color1}; font-weight: bold;'>{st.session_state.gender_label}</h1>", unsafe_allow_html=True)
#         # time.sleep(0.1)
#         st.rerun()
#
#
# mic_icon = "🎤 Start Recording" if not st.session_state.recording else "⏹ Stop Recording"
# if st.button(mic_icon, key="mic_button"):
#     toggle_recording()
#
# if st.session_state.recording:
#     process_audio()


import streamlit as st
import numpy as np
import torch
from helper import load_model, predict_gender  # Assuming these exist
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
# import av
# import threading

import asyncio

try:
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        asyncio.set_event_loop(asyncio.new_event_loop())
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


try:
    dir(torch.classes)  # Force Streamlit to ignore it
except RuntimeError:
    torch.classes = None  # Prevents Streamlit from accessing it


# Load the ML model & feature extractor
model, feature_extractor = load_model()

# Streamlit UI
st.title("🎙️ Gender by Voice (WebRTC)")

# lock = threading.Lock()
if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer = []


def audio_callback(frame):
    """Processes audio stream from WebRTC."""
    audio_data = np.array(frame.to_ndarray()).astype(np.float32)
    print(len(audio_data))

    # Collect 0.3s chunks before processing
    sample_rate = frame.sample_rate  # Get sample rate from WebRTC
    chunk_size = int(sample_rate * 0.3)  # Convert 0.3 sec to samples

    st.session_state.audio_buffer.extend(audio_data)

    if len(st.session_state.audio_buffer) >= chunk_size:
        # Extract a 0.3s chunk
        chunk = np.array(st.session_state.audio_buffer[:chunk_size])
        st.session_state.audio_buffer = st.session_state.audio_buffer[chunk_size:]  # Properly update buffer

        # Normalize and predict gender
        if chunk.size > 0:
            waveform = torch.tensor(chunk, dtype=torch.float32)
            waveform /= torch.max(torch.abs(waveform)) if torch.max(torch.abs(waveform)) > 0 else 1

            st.session_state.gender_label = predict_gender(waveform, model, feature_extractor)

        # st.rerun()
    return frame  # Must return a frame for WebRTC to continue streaming
# WebRTC Streaming
ctx = webrtc_streamer(
    key="audio_stream",
    mode=WebRtcMode.SENDONLY,  # Only send audio, no video needed
    # audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},  # Audio only
    audio_frame_callback= audio_callback,
    rtc_configuration={  # Add this config
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": "turn:relay.metered.ca:80", "username": "open", "credential": "open"}]
    }
)

# Initialize session state for gender label
if "gender_label" not in st.session_state:
    st.session_state.gender_label = "Unknown"

# Display gender prediction dynamically
color = "red" if st.session_state.gender_label == "Male" else (
    "yellow" if st.session_state.gender_label == "None" else "green")
st.markdown(
    f"<h1 style='text-align: center; color: {color}; font-weight: bold;'>{st.session_state.gender_label}</h1>",
    unsafe_allow_html=True
)

if ctx and ctx.state.playing:
    st.write("WebRTC is running.")
    print("WebRTC is running.")  # Print to logs
else:
    print("WebRTC not running.")  # Print if it never starts