# eval/run_eval.py
import json
import sys
import os
from eval.eval_wer import compute_wer_metrics
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

sys.path.insert(0, ROOT)

from LoadAudio import transcribe_audio
from eval.metrics import (
    compute_sentiment_metrics,
    compute_speaker_metrics,
    compute_transcription_metrics,
)

def run(output_path="eval/eval_report.json"):
    audio_path = os.path.join(ROOT, "data", "llamada.mp3")

    if not os.path.exists(audio_path):
        print(f"Error: No se encontró el archivo de audio en {audio_path}")
        return
    mod_time = os.path.getmtime(audio_path)
    import datetime
    print(f"Archivo de audio: {audio_path}")
    print(f"Última modificación: {datetime.datetime.fromtimestamp(mod_time)}")
    print(f"Tamaño: {os.path.getsize(audio_path) / 1024 / 1024:.2f} MB\n")
    print("---")

    print("Ejecutando pipeline completo para evaluación...")
    result = transcribe_audio()

    report = {
        "transcription": compute_transcription_metrics(result["_raw_segments"]),
        "sentiment":     compute_sentiment_metrics(result["segments"]),
        "speakers":      compute_speaker_metrics(result["speaker_summary"]),
        "latency":       result["_latency"],
        "wer":           compute_wer_metrics("eval/reference_transcript.txt", result["segments"]),
    }


    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReporte guardado en {output_path}")

if __name__ == "__main__":
    run()