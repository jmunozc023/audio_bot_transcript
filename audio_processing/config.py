import os
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass(frozen=True)

class Settings:
    audio_file: str = "data/llamada.mp3"
    output_file: str = "data/call.txt"
    sample_rate: int = 16000
    min_chunk_seconds: float = 0.5
    n_speakers: int = 2
    asr_model_name: str = "base"
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    llm_model: str = "gpt-4o-mini"

def load_settings() -> Settings:
    load_dotenv()
    return Settings()
def validate_runtime(settings: Settings) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OpenAI API Key in environment variables.")
    if not os.path.exists(settings.audio_file):
        raise FileNotFoundError(f"Audio file not found: {settings.audio_file}")