# realtime-gender-voice-detection
A streamlit based realtime gender identification from audio web app powered by transformers and pyaudio


## Challenges Faced


Predicting gender in real-time while keeping the system responsive.
   **Solution:**
   - Used a **circular buffer approach** to continuously record short audio segments without gaps.
   - Optimized the **0.3s** to ensure minimal delay while preserving accuracy.
   - Eliminated redundant processing by directly passing audio to the model without intermediate storage.
   - Ensured **non-blocking recording** by allowing parallel audio capture and model inference by using threading.
   - Reduced **preprocessing overhead** by skipping unnecessary transformations and keeping feature extraction efficient.
   - Implemented a **short recording duration** (0.3s) to keep predictions frequent and responsive.

**Other Challenges**
- Streamlit's default execution model does not support continuous loops, making it difficult to process real-time microphone input efficiently.
- Preventing crashes when stopping and restarting the microphone stream.
- Keeping the UI responsive while processing audio data in the background.
- Handling noisy input and preventing incorrect predictions.

