import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

try:
  nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
  nltk.download("vader_lexicon", quiet=True)


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
sid = SentimentIntensityAnalyzer()

AUDIO_FILE = "data/llamada.mp3"
OUTPUT_FILE = "data/call.txt"

def transcribe_audio_whisper() -> str:
    print("Transcribiendo audio con Whisper...")

    with open(AUDIO_FILE, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    
    segments = response.segments or []

    lines = []
    for seg in segments:
        start = round(seg.start, 2)
        end = round(seg.end, 2)
        text = seg.text.strip()
        lines.append(f"[{start}s - {end}s] {text}")
    raw_transcript = "\n".join(lines)
    print(f"Whisper: {len(segments)} segmentos transcritos")
    return raw_transcript

ANALYSIS_PROMPT = """
Eres un analista experto en conversaciones de call center.

Recibirás una transcripción sin procesar con marcas de tiempo generadas por Whisper ASR.
La transcripción no tiene etiquetas de hablantes aún — tu trabajo es agregarlas.

Realiza estas tres tareas:

1. DIARIZACIÓN

    - Identifica a los distintos hablantes basándote en señales conversacionales: patrones de turnos, flujo de preguntas y respuestas, propiedad del tema, estilo de habla y brechas de tiempo que indiquen cambios de hablante.
    - Asigna etiquetas consistentes: Agente, Cliente. Si se detectan más de 2 hablantes, usa Agente, Cliente, Supervisor, etc.
    - Fusiona líneas consecutivas del mismo hablante si la brecha es < 1.5 s.

2. IDENTIFICACIÓN DE ROLES

    - Agente: usa lenguaje formal/de servicio, hace preguntas de aclaración, ofrece soluciones, hace referencia a datos de cuenta/sistema.
    - Cliente: describe problemas, expresa necesidades o emociones, pide ayuda o información.

3. SENTIMIENTO por segmento

    - Etiqueta cada segmento como: "positivo", "negativo" o "neutral".
    - Agrega un puntaje compuesto entre -1.0 y 1.0.

Devuelve ÚNICAMENTE un objeto JSON válido — sin markdown, sin comentarios

{{
  "segments": [
    {{
      "start": 0.0,
      "end": 4.2,
      "speaker": "Agent",
      "text": "...",
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

Raw transcript:
{transcript}
"""

def analysis_with_gpt4o(raw_transcript: str) -> dict:
    print("Analizando con GPT-4o (diarización + rol + sentimiento)...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", 
             "content": 
                ("You are a call center conversation analyst. "
                "You always respond with a valid JSON only, no markdown."
                )
            },
            {"role": "user",
              "content": ANALYSIS_PROMPT.format(transcript=raw_transcript)
            }
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw_json = response.choices[0].message.content.strip()
    result = json.loads(raw_json)

    print(
        f"GPT-4o: {result.get('detected_speakers', '?')} speakers detectados, "
        f"{len(result.get('segments', []))} segmentos."
    )
    return result

def validate_sentiment(segments: list[dict]) -> list[dict]:

    for seg in segments:
        vader_score = sid.polarity_scores(seg["text"])["compound"]
        gpt_score = seg.get("sentiment_score", 0)
        if abs(vader_score - gpt_score) > 0.5:
            seg["sentiment_score"] = round(vader_score, 3)
            seg["sentiment"] = (
                "positive" if vader_score >= 0.05 else
                "negative" if vader_score <= -0.05 else
                "neutral"
            )
    return segments

def save_transcript(result: dict):

    segments = result.get("segments", [])
    speaker_summary = result.get("speaker_summary", {})
    lines = []

    lines.append("=== Call Summary ===")
    for speaker, data in speaker_summary.items():
        lines.append(
            f"{speaker}: sentiment={data['overall']}"
            f"(avg={data['avg_sentiment']}), "
            f"talk_time={data['talk_time_seconds']}s, "
            f"segments={data['segment_count']}"
        )
    lines.append(f"Total speakers detectados: {result.get('detected_speakers', '?')}")
    lines.append("\n=== TRANSCRIPT ===")
    for seg in segments:
        lines.append(
            f"{seg['speaker']}: {seg['text']} "
            f"[{seg['start']}s - {seg['end']}s] "
            f"[{seg['sentiment']} {seg['sentiment_score']}]"
        )
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"{OUTPUT_FILE} generado - {len(segments)} segmentos")

def transcribe_audio():
    raw_transcript = transcribe_audio_whisper()
    result = analysis_with_gpt4o(raw_transcript)
    result["segments"] = validate_sentiment(result.get("segments", []))
    save_transcript(result)


if __name__ == "__main__":
    transcribe_audio()