import os
from .config import load_settings, validate_runtime
from .asr_service import create_asr_model, transcribe_segments
from .diarization import extract_embeddings_and_texts, cluster_speakers, build_speaker_transcript
from .role_labeler import create_openai_client, relabel_speakers_with_llm

def write_output(output_file: str, content: str) -> None:
    directory = os.path.dirname(output_file)
    if directory: 
        os.makedirs(directory, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)
        print("call.txt generated")

def run_pipeline() -> None:
    settings = load_settings()
    validate_runtime(settings)
    asr_model = create_asr_model(settings)
    openai_client = create_openai_client()
    segments = transcribe_segments(asr_model, settings.audio_file)
    embeddings, texts = extract_embeddings_and_texts(settings.audio_file, settings.sample_rate, segments, settings.min_chunk_seconds,)
    labels = cluster_speakers(embeddings, settings.n_speakers)
    transcript = build_speaker_transcript(labels, texts)
    final_transcript = relabel_speakers_with_llm(openai_client, settings.llm_model, transcript,)
    write_output(settings.output_file, final_transcript)
    