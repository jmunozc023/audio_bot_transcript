import os
import io
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon", quiet=True)

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
sid = SentimentIntensityAnalyzer()

AUDIO_FILE = "data/llamada.mp3"
OUTPUT_FILE = "data/call.txt"

_progress_cb = None


def set_progress_callback(cb):
    global _progress_cb
    _progress_cb = cb


def report(msg: str, pct: int = None):
    print(msg)
    if _progress_cb:
        _progress_cb(msg, pct)


def transcribe_and_diarize() -> list[dict]:
    """
    Transcribe el audio e identifica hablantes en un solo paso
    usando gpt-4o-transcribe-diarize.

    Retorna lista de segmentos con: start, end, speaker, text
    """
    report("Transcribiendo y diarizando con gpt-4o-transcribe-diarize...")
    t0 = time.time()

    with open(AUDIO_FILE, "rb") as f:
        audio_bytes = f.read()

    buf = io.BytesIO(audio_bytes)
    buf.name = os.path.basename(AUDIO_FILE)

    response = client.audio.transcriptions.create(
        model="gpt-4o-transcribe-diarize",
        file=buf,
        response_format="verbose_json",
    )

    elapsed = round(time.time() - t0, 1)

    # El modelo devuelve segmentos con speaker label incluido
    segs = []
    for seg in (response.segments or []):
        segs.append({
            "start":   round(seg.start, 2),
            "end":     round(seg.end,   2),
            "speaker": getattr(seg, "speaker", "Speaker_0"),  # ej: "speaker_0", "speaker_1"
            "text":    seg.text.strip(),
        })

    # Normalizar etiquetas de hablantes a "Agent" / "Customer" / "Supervisor"
    segs = normalize_speaker_labels(segs)

    speakers = len({s["speaker"] for s in segs})
    report(f"Transcripción: {len(segs)} segmentos, {speakers} hablantes en {elapsed}s", 60)

    return segs


def normalize_speaker_labels(segments: list[dict]) -> list[dict]:
    """
    El modelo devuelve etiquetas genéricas como 'speaker_0', 'speaker_1'.
    Las mapeamos a 'Agent', 'Customer', 'Supervisor', etc.
    El primer hablante se asume Agente (quien inicia la llamada),
    el resto se asignan en orden de aparición.
    """
    ROLE_NAMES = ["Agent", "Customer", "Supervisor", "Other"]

    seen = {}  # speaker_id -> nombre de rol
    for seg in segments:
        raw = seg["speaker"]
        if raw not in seen:
            idx = len(seen)
            seen[raw] = ROLE_NAMES[idx] if idx < len(ROLE_NAMES) else f"Speaker_{idx}"
        seg["speaker"] = seen[raw]

    return segments


def add_sentiment(segments: list[dict]) -> list[dict]:
    """
    Agrega análisis de sentimiento a cada segmento usando GPT-4o-mini,
    validado por VADER como guardrail.
    """
    report("Analizando sentimiento con GPT-4o-mini...", 70)
    t0 = time.time()

    # Construir texto para GPT
    lines = [
        f"[{s['start']}s - {s['end']}s] {s['speaker']}: {s['text']}"
        for s in segments
    ]
    transcript_text = "\n".join(lines)

    SENTIMENT_PROMPT = """
Eres un analista experto en call centers.
Recibirás una transcripción con hablantes ya identificados.

Tu única tarea es agregar análisis de sentimiento a cada segmento.

Devuelve ÚNICAMENTE un JSON válido, sin markdown, con esta estructura:
{{
  "segments": [
    {{
      "start": 0.0,
      "end": 4.2,
      "sentiment": "positive",
      "sentiment_score": 0.62
    }}
  ],
  "speaker_summary": {{
    "Agent": {{
      "avg_sentiment": 0.31,
      "overall": "positive",
      "talk_time_seconds": 142.5,
      "segment_count": 12
    }},
    "Customer": {{
      "avg_sentiment": -0.18,
      "overall": "negative",
      "talk_time_seconds": 98.3,
      "segment_count": 9
    }}
  }},
  "detected_speakers": 2
}}

Transcripción:
{transcript}
""".format(transcript=transcript_text)

    max_tokens = min(len(segments) * 100 + 512, 8192)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a call center sentiment analyst. Respond with valid JSON only, no markdown."
            },
            {
                "role": "user",
                "content": SENTIMENT_PROMPT
            }
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=max_tokens,
    )

    if response.choices[0].finish_reason == "length":
        raise ValueError(
            f"GPT-4o-mini cortó la respuesta ({len(segments)} segmentos). "
            "Considera usar un audio más corto."
        )

    sentiment_result = json.loads(response.choices[0].message.content.strip())

    # Combinar sentiment de GPT con los segmentos originales (por índice)
    sentiment_segs = sentiment_result.get("segments", [])
    for i, seg in enumerate(segments):
        if i < len(sentiment_segs):
            seg["sentiment"]       = sentiment_segs[i].get("sentiment", "neutral")
            seg["sentiment_score"] = sentiment_segs[i].get("sentiment_score", 0.0)
        else:
            seg["sentiment"]       = "neutral"
            seg["sentiment_score"] = 0.0

    elapsed = round(time.time() - t0, 1)
    report(f"Sentimiento: {len(segments)} segmentos analizados en {elapsed}s", 85)

    return segments, sentiment_result.get("speaker_summary", {}), sentiment_result.get("detected_speakers", len({s["speaker"] for s in segments}))


def validate_sentiment(segments: list[dict]) -> list[dict]:
    """
    Valida los scores de GPT con VADER.
    Si la diferencia es > 0.5, usa el score de VADER.
    """
    for seg in segments:
        vader_score = sid.polarity_scores(seg["text"])["compound"]
        gpt_score   = seg.get("sentiment_score", 0)
        if abs(vader_score - gpt_score) > 0.5:
            seg["sentiment_score"] = round(vader_score, 3)
            seg["sentiment"] = (
                "positive" if vader_score >= 0.05 else
                "negative" if vader_score <= -0.05 else
                "neutral"
            )
    return segments


def save_transcript(segments: list[dict], speaker_summary: dict, detected_speakers: int):
    lines = ["=== Call Summary ==="]

    for speaker, data in speaker_summary.items():
        lines.append(
            f"{speaker}: sentiment={data['overall']}"
            f"(avg={data['avg_sentiment']}), "
            f"talk_time={data['talk_time_seconds']}s, "
            f"segments={data['segment_count']}"
        )
    lines.append(f"Total speakers detectados: {detected_speakers}")
    lines.append("\n=== TRANSCRIPT ===")

    for seg in segments:
        lines.append(
            f"{seg['speaker']}: {seg['text']} "
            f"[{seg['start']}s - {seg['end']}s] "
            f"[{seg['sentiment']} {seg['sentiment_score']}]"
        )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    report(f"{OUTPUT_FILE} generado - {len(segments)} segmentos", 100)


def transcribe_audio():
    t0 = time.time()

    # Paso 1: transcripción + diarización nativa (1 llamada de API)
    segments = transcribe_and_diarize()

    # Paso 2: análisis de sentimiento con GPT-4o-mini
    segments, speaker_summary, detected_speakers = add_sentiment(segments)

    # Paso 3: validar sentimiento con VADER
    segments = validate_sentiment(segments)

    # Paso 4: guardar resultado
    save_transcript(segments, speaker_summary, detected_speakers)

    report(f"Tiempo total: {round(time.time() - t0, 1)}s", 100)


if __name__ == "__main__":
    transcribe_audio()