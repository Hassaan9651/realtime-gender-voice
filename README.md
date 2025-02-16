# Realtime-Gender-Voice-Detection
This project aims to design and develop a machine learning (ML) system that processes a real-time audio stream from a microphone and determines the gender of the speaker. The system provides continuous inference, allowing it to detect gender changes dynamically. The output is a live text stream indicating whether the running/speaking voice is classified as **Male** or **Female**.

## Approach to the problem
- The microphone input is captured using **Flask-SocketIO**, enabling real-time audio streaming from a web browser.
- The frontend, built with **HTML & JavaScript**, uses `navigator.mediaDevices.getUserMedia` to capture live microphone input.
- The captured audio is streamed to a Flask backend via WebSockets, ensuring low-latency real-time processing.
- A pre-trained machine learning model with 316M params, `alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech`, from Hugging Face, is utilized for gender classification.
- The model is binary so in case of noise or blank it'll still go with a Gender
- This model is based on the Facebook's **Wav2Vec 2.0** architecture, which is well-suited for processing raw audio waveforms and extracting meaningful speech features.
- The audio chunks are passed through the model, which classifies each segment as either **Male** or **Female**.
- The system is designed as a continuous streaming pipeline where each audio packet sized 4096 undergoes real-time inference.
- The classification results are continuously updated and sent back to the frontend using WebSockets for live display.
- The inference output is formatted as a live text stream on a dynamic web UI.
- The application is deployed on **Google Cloud Run**, providing scalable real-time inference.

## Challenges Faced
Predicting gender in real-time while keeping the system responsive maintaining accuracy.
   **Solution:**
   - Optimized the chunk size to balance **latency and accuracy**.
   - Eliminated redundant processing by directly passing audio to the model without intermediate storage.
   - Implemented a queuing system so that each time the model has enough context to make a sound prediction.
   - Ensured **non-blocking processing** by handling audio capture and model inference in parallel.
   - Reduced **preprocessing overhead** by skipping unnecessary transformations and keeping feature extraction efficient.
   - Designed a **lightweight Flask-SocketIO server** to handle real-time WebSocket communication efficiently.

**Other Challenges**
- Google Cloud Run does not natively support WebSockets well, leading to connection drops.
   **Solution:**
   - Enabled **long polling fallback** on both frontend and backend to ensure uninterrupted streaming.
   - Increased `ping_interval` and `ping_timeout` in `Flask-SocketIO` to prevent premature disconnections.
   - Used **Gunicorn with eventlet** to ensure compatibility with WebSockets.
- Preventing crashes when stopping and restarting the microphone stream.
   **Solution:**
   - Ensured proper cleanup of WebSocket connections upon stream termination.
   - Implemented **reconnection logic** on the frontend to automatically reconnect if the WebSocket disconnects.
- Keeping the UI responsive while processing audio data in the background.
   **Solution:**
   - Used **asynchronous WebSocket handling** on both frontend and backend.
   - Reduced blocking operations by processing audio on a separate thread in Flask.
   - Used **shorter inference windows** to avoid speech misclassification due to noise.
