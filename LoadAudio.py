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

def transcribe_audio_whisper() -> list[dict]:


    report("Transcribiendo audio con Whisper...")
    t0 = time.time()

    with open(AUDIO_FILE, "rb") as f:
        audio_bytes = f.read()

    buf = io.BytesIO(audio_bytes)
    buf.name = os.path.basename(AUDIO_FILE)

    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=buf,
        response_format="verbose_json",
        timestamp_granularities=["segment"],
    )

    elapsed = round(time.time() - t0, 1)
    segs = [
        {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip()
        }
        for seg in (response.segments or [])
    ]
    report(f"Whisper: {len(segs)} segmentos en {elapsed}s", 40)

    return segs



def build_transcript_for_gpt(segments: list[dict]) -> str:

    lines = []
    for seg in segments:
        start = round(seg["start"], 2)
        end   = round(seg["end"],   2)
        lines.append(f"[{start}s - {end}s] {seg['text'].strip()}")
    return "\n".join(lines)

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

def analysis_with_gpt4o(segments: list[dict]) -> dict:
    report("Analizando transcripción con GPT-4o...", 50)
    t0 = time.time()
    max_tokens = min(len(segments) * 200 + 1024, 16384)
    transcript = build_transcript_for_gpt(segments)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", 
             "content": (
                    "You are a call center conversation analyst. "
                    "You always respond with valid JSON only, no markdown."
                )
            },
            {"role": "user",
              "content": ANALYSIS_PROMPT.format(transcript=transcript)
            }
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=max_tokens,
    )

    finish_reason = response.choices[0].finish_reason
    if finish_reason == "length":
        raise ValueError(
            f"GPT-4o cortó la respuesta ({len(segments)} segmentos, "
            f"max_tokens={max_tokens}). Considera usar un audio más corto."
        )
    
    elapsed = round(time.time() - t0, 1)
    result = json.loads(response.choices[0].message.content.strip())


    report(
        f"GPT-4o: {result.get('detected_speakers', '?')} speakers detectados, "
        f"{len(result.get('segments', []))} segmentos en {elapsed}s",
        85
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
    lines = ["=== Call Summary ==="]

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
    report(f"{OUTPUT_FILE} generado - {len(segments)} segmentos", 100)

def transcribe_audio():
    t0 = time.time()
    segs= transcribe_audio_whisper()
    result = analysis_with_gpt4o(segs)
    result["segments"] = validate_sentiment(result["segments"])
    save_transcript(result)
    report(f"Tiempo total: {round(time.time() - t0, 1)}s", 100)


if __name__ == "__main__":
    transcribe_audio()