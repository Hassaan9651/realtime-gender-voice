import streamlit as st
import numpy as np
import av
from streamlit_webrtc import WebRtcMode, webrtc_streamer

import asyncio

try:
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        asyncio.set_event_loop(asyncio.new_event_loop())
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# Audio processing function
def process_audio(frame: av.AudioFrame) -> av.AudioFrame:
    samples = frame.to_ndarray()
    print(len(samples))
    samples = samples * 2  # Example: Amplifying the volume
    samples = np.clip(samples, -32768, 32767).astype(np.int16)
    new_frame = av.AudioFrame.from_ndarray(samples, layout=frame.layout)
    new_frame.sample_rate = frame.sample_rate
    return new_frame


st.title("Real-time Audio Processing")
st.write("This app captures audio from your microphone, processes it, and plays it back in real-time.")

webrtc_ctx = webrtc_streamer(
    key="audio-processor",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=process_audio,
    media_stream_constraints={"audio": True, "video": False},
    rtc_configuration={  # Add this config
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]},
                       {"urls": "turn:relay.metered.ca:80", "username": "open", "credential": "open"}]
    }
)

if webrtc_ctx and webrtc_ctx.state.playing:
    st.write("WebRTC is running.")
    print("WebRTC is running.")  # Print to logs
else:
    print("WebRTC not running.")  # Print if it never starts