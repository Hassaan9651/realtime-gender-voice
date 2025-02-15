import streamlit as st
from stream_helper import *
import time
import numpy as np

st.title("Gender by Voice")

model, feature_extractor = load_model()

def start_audio_stream():
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    return p, stream


if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'gender_label' not in st.session_state:
    st.session_state.gender_label = "Unknown"


def toggle_recording():
    st.session_state.recording = not st.session_state.recording
    if st.session_state.recording:
        p, stream = start_audio_stream()
        st.session_state.p = p
        st.session_state.stream = stream
    else:
        if 'stream' in st.session_state:
            st.session_state.stream.stop_stream()
            st.session_state.stream.close()
            st.session_state.p.terminate()
        # st.session_state.gender_label = "Unknown"
    st.rerun()

reset = 0
def process_audio():
    if 'stream' in st.session_state and st.session_state.recording:
        frames = []
        duration = 0.3
        for _ in range(0, int(RATE / CHUNK * duration)):
            data = st.session_state.stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

        waveform_np = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32)
        if waveform_np.size != 0:
            waveform = torch.tensor(waveform_np)
            waveform /= torch.max(torch.abs(waveform)) if torch.max(torch.abs(waveform)) > 0 else 1
            t = time.time()
            st.session_state.gender_label = predict_gender(waveform, model, feature_extractor)
            print(time.time() - t)
            color1 = "red" if st.session_state.gender_label=="Male" else ("yellow" if st.session_state.gender_label=="None" else "green")
            st.markdown(f"<h1 style='text-align: center; color: {color1}; font-weight: bold;'>{st.session_state.gender_label}</h1>", unsafe_allow_html=True)
        # time.sleep(0.1)
        st.rerun()


mic_icon = "🎤 Start Recording" if not st.session_state.recording else "⏹ Stop Recording"
if st.button(mic_icon, key="mic_button"):
    toggle_recording()

if st.session_state.recording:
    process_audio()