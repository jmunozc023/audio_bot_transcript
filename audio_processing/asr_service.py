from typing import List
from faster_whisper import WhisperModel
from audio_processing.config import Settings

def create_asr_model(settings: Settings) -> WhisperModel:
    print("Loading ASR model...")
    return WhisperModel(
        settings.asr_model_name,
        device=settings.asr_device,
        compute_type=settings.asr_compute_type,
    )

def transcribe_segments(model: WhisperModel, audio_file: str) -> List:
    print("Transcribing audio...")
    segments, _ = model.transcribe(audio_file)
    return list(segments)