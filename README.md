# Realtime-Gender-Voice-Detection
This project aims to design and develop a machine learning (ML) system that processes a real-time audio stream from a microphone and determines the gender of the speaker. The system provides continuous inference, allowing it to detect gender changes dynamically. The output is a live text stream indicating whether the running/speaking voice is classified as **Male** or **Female**.

## Approach to the problem
- The microphone input is captured using the `pyaudio` library, which enables real-time audio streaming.
- The audio stream is processed in small chunks to allow efficient inference while maintaining low latency.
- A pre-trained machine learning model with 96M params, `7wolf/wav2vec2-base-gender-classification`, from Hugging Face, is utilized for gender classification.
- This model is based on the **Wav2Vec 2.0** architecture, which is well-suited for processing raw audio waveforms and extracting meaningful speech features.
- The audio chunks are passed through the model, which classifies each segment as either **Male** or **Female**.
- The system is designed as a continuous streaming pipeline where each audio chunk undergoes real-time inference.
- The classification results are continuously updated to reflect changes in the detected speaker's gender.
- The inference output is formatted as a live text stream.
- A minimalistic user interface is developed using **Streamlit**, providing a web-based interface for real-time visualization of the detected gender.
- The web app displays the speaker's gender and updates dynamically as the speaker changes.
- Real-time gender classification with minimal latency.


## Challenges Faced

Predicting gender in real-time while keeping the system responsive.
   **Solution:**
   - Used a **circular buffer approach** to continuously record short audio segments without gaps.
   - Optimized the **0.3s** to ensure minimal delay while preserving accuracy.
   - Eliminated redundant processing by directly passing audio to the model without intermediate storage.
   - Ensured **non-blocking recording** by allowing audio capture and model inference with minimal delay.
   - Reduced **preprocessing overhead** by skipping unnecessary transformations and keeping feature extraction efficient.
   - Implemented a **short recording duration** (0.3s) to keep predictions frequent and responsive.

**Other Challenges**
- Streamlit's default execution model does not support continuous loops, making it difficult to process real-time microphone input efficiently.
- Preventing crashes when stopping and restarting the microphone stream.
- Keeping the UI responsive while processing audio data in the background.
- Handling noisy input and preventing incorrect predictions.