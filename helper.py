import sys
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# Constants
TARGET_SAMPLING_RATE = 16000


def load_model():
    """Load the model and feature extractor."""
    model_name = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    return model, feature_extractor


def predict_gender(waveform, model, feature_extractor):
    """Predict gender from an audio waveform."""
    inputs = feature_extractor(
        waveform,
        sampling_rate=TARGET_SAMPLING_RATE,
        return_tensors="pt",
        padding=True,
        do_normalize=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # print(logits)
        predicted_class_id = logits.argmax(dim=-1).item()
        # if abs(logits[0][0] - logits[0][1]) <= 3.5:
        #     return "None"
    return "Male" if predicted_class_id else "Female"