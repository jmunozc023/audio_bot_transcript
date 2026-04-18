import os
import io
import json
import time
from openai import OpenAI, APITimeoutError, APIConnectionError
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
GPT_CHUNK_SIZE = int(os.getenv("GPT_CHUNK_SIZE", "80"))
GPT_TIMEOUT_SECONDS = float(os.getenv("GPT_TIMEOUT_SECONDS", "240"))
GPT_REQUEST_RETRIES = int(os.getenv("GPT_REQUEST_RETRIES", "2"))
GPT_RETRY_BACKOFF_SECONDS = float(os.getenv("GPT_RETRY_BACKOFF_SECONDS", "2"))

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

def _chunk_segments(segments: list[dict], chunk_size: int) -> list[list[dict]]:
    if chunk_size <= 0:
        return [segments]
    return [segments[i:i + chunk_size] for i in range(0, len(segments), chunk_size)]

def _score_to_label(score: float) -> str:
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"

def _build_speaker_summary_from_segments(segments: list[dict]) -> dict:
    buckets = {}
    for seg in segments:
        speaker = seg.get("speaker", "Unknown")
        duration = max(float(seg.get("end", 0)) - float(seg.get("start", 0)), 0.0)
        score = float(seg.get("sentiment_score", 0))

        data = buckets.setdefault(
            speaker,
            {"talk_time_seconds": 0.0, "segment_count": 0, "scores": []},
        )
        data["talk_time_seconds"] += duration
        data["segment_count"] += 1
        data["scores"].append(score)

    summary = {}
    for speaker, data in buckets.items():
        avg = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0.0
        summary[speaker] = {
            "avg_sentiment": round(avg, 2),
            "overall": _score_to_label(avg),
            "talk_time_seconds": round(data["talk_time_seconds"], 1),
            "segment_count": data["segment_count"],
        }
    return summary

def _normalize_segments(segments: list[dict]) -> list[dict]:
    normalized = []
    for seg in segments:
        start = float(seg.get("start", 0))
        end = float(seg.get("end", start))
        if end < start:
            end = start

        score = float(seg.get("sentiment_score", 0))
        sentiment = seg.get("sentiment")
        if sentiment not in {"positive", "negative", "neutral"}:
            sentiment = _score_to_label(score)

        normalized.append(
            {
                "start": start,
                "end": end,
                "speaker": str(seg.get("speaker", "Unknown")),
                "text": str(seg.get("text", "")).strip(),
                "sentiment": sentiment,
                "sentiment_score": round(score, 3),
            }
        )
    return normalized

def _merge_chunk_results(results: list[dict]) -> dict:
    merged_segments = []
    for item in results:
        merged_segments.extend(_normalize_segments(item.get("segments", [])))

    merged_segments.sort(key=lambda s: (s["start"], s["end"]))
    speaker_summary = _build_speaker_summary_from_segments(merged_segments)

    return {
        "segments": merged_segments,
        "speaker_summary": speaker_summary,
        "detected_speakers": len(speaker_summary),
    }

def _request_gpt_chunk(transcript: str, max_tokens: int, chunk_idx: int, total_chunks: int):
    attempts = GPT_REQUEST_RETRIES + 1
    last_exc = None

    for attempt in range(1, attempts + 1):
        try:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a call center conversation analyst. "
                            "You always respond with valid JSON only, no markdown."
                        ),
                    },
                    {
                        "role": "user",
                        "content": ANALYSIS_PROMPT.format(transcript=transcript),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=max_tokens,
                timeout=GPT_TIMEOUT_SECONDS,
            )
        except (APITimeoutError, APIConnectionError) as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            wait_s = round(GPT_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)), 1)
            report(
                f"Reintento GPT chunk {chunk_idx}/{total_chunks} "
                f"({attempt}/{attempts - 1}) en {wait_s}s por timeout/conexion..."
            )
            time.sleep(wait_s)

    raise RuntimeError(
        f"GPT chunk {chunk_idx}/{total_chunks} falló tras {attempts} intentos "
        f"(timeout={GPT_TIMEOUT_SECONDS}s)."
    ) from last_exc

def analysis_with_gpt4o(segments: list[dict]) -> dict:
    report("Analizando transcripción con GPT-4o...", 50)
    t0 = time.time()
    chunks = _chunk_segments(segments, GPT_CHUNK_SIZE)
    chunk_results = []

    for i, chunk in enumerate(chunks, start=1):
        report(f"GPT chunk {i}/{len(chunks)} ({len(chunk)} segmentos)...")
        max_tokens = min(len(chunk) * 200 + 1024, 16384)
        transcript = build_transcript_for_gpt(chunk)

        response = _request_gpt_chunk(transcript, max_tokens, i, len(chunks))

        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length":
            raise ValueError(
                f"GPT-4o cortó la respuesta en chunk {i}/{len(chunks)} "
                f"({len(chunk)} segmentos, max_tokens={max_tokens})."
            )

        chunk_results.append(json.loads(response.choices[0].message.content.strip()))

    result = _merge_chunk_results(chunk_results)
    elapsed = round(time.time() - t0, 1)

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

def transcribe_audio() -> dict:
    # t0 = time.time()
    # segs= transcribe_audio_whisper()
    # result = analysis_with_gpt4o(segs)
    # result["segments"] = validate_sentiment(result["segments"])
    # save_transcript(result)
    # report(f"Tiempo total: {round(time.time() - t0, 1)}s", 100)

    import time
    t0 = time.time()

    t_whisper = time.time()
    segs = transcribe_audio_whisper()
    whisper_elapsed = round(time.time() - t_whisper, 2)

    t_gpt = time.time()
    result = analysis_with_gpt4o(segs)
    gpt_elapsed = round(time.time() - t_gpt, 2)

    result["segments"] = validate_sentiment(result["segments"])

    result["_latency"] = {
        "whisper_seconds": whisper_elapsed,
        "gpt_seconds":     gpt_elapsed,
        "total_seconds":   round(time.time() - t0, 2),
    }
    result["_raw_segments"] = segs   # segmentos originales de Whisper (sin diarizar)
    save_transcript(result)

    report(f"Tiempo total: {result['_latency']['total_seconds']}s", 100)
    return result


if __name__ == "__main__":
    transcribe_audio()