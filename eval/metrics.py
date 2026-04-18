import json
from sklearn.metrics import cohen_kappa_score, f1_score, classification_report
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download("vader_lexicon", quiet=True)

sid = SentimentIntensityAnalyzer()

def _vader_label(text: str) -> str:
    s = sid.polarity_scores(text)["compound"]
    if s >= 0.05:
        return "positive"
    elif s <= -0.05:
        return "negative"
    else:
        return "neutral"

def compute_sentiment_metrics(segments: list[dict]) -> dict:
    """
    Compara etiquetas VADER vs GPT usando los segmentos
    que ya devuelve analysis_with_gpt4o().
    No necesita nada externo: ambos scores viven en cada segmento.
    """
    gpt_labels = [s["sentiment"] for s in segments]
    vader_labels = [_vader_label(s["text"]) for s in segments]

    gpt_scores = [s["sentiment_score"] for s in segments]
    vader_scores = [sid.polarity_scores(s["text"])["compound"] for s in segments]
    mae = sum(abs(g - v) for g, v in zip(gpt_scores, vader_scores)) / len(segments)

    overrides = sum(
        1 for g, v in zip(gpt_scores, vader_scores)
        if abs(g-v) > 0.5
    )

    labels = ["positive", "negative", "neutral"]
    kappa = cohen_kappa_score(vader_labels, gpt_labels, labels=labels)
    report = classification_report(
        vader_labels, gpt_labels, labels=labels, output_dict=True, zero_division=0
    )

    return {
        "cohen_kappa": round(kappa, 4),
        "kappa_interpretation": _interpret_kappa(kappa),
        "mae_scores": round(mae, 4),
        "override_rate": round(overrides / len(segments), 4),
        "overrides_n": overrides,
        "total_segments": len(segments),
        "f1_macro": round(report["macro avg"]["f1-score"], 4),
        "f1_per_class":{
            k: round(v["f1-score"], 4)
            for k, v in report.items()
            if k in labels
        },
        "classification_report": report,
    }
def compute_speaker_metrics(speaker_summary: dict) -> dict:
    """
    Métricas de balance de hablantes a partir del
    speaker_summary que GPT-4o-mini ya produce.
    """
    times = {sp: d["talk_time_seconds"] for sp, d in speaker_summary.items()}
    total = sum(times.values()) or 1
    return {
        "talk_time_seconds": times,
        "talk_time_ratio":   {sp: round(t / total, 4) for sp, t in times.items()},
        "avg_sentiment":     {sp: d["avg_sentiment"] for sp, d in speaker_summary.items()},
        "overall_sentiment": {sp: d["overall"] for sp, d in speaker_summary.items()},
    }

def compute_transcription_metrics(segments: list[dict]) -> dict:
    """
    Métricas de cobertura a partir de los segmentos de Whisper.
    """
    if not segments:
        return {}
    total_audio   = segments[-1]["end"]
    covered       = sum(s["end"] - s["start"] for s in segments)
    minutes       = total_audio / 60 or 1
    return {
        "total_audio_seconds":  round(total_audio, 2),
        "covered_seconds":      round(covered, 2),
        "coverage_pct":         round(covered / total_audio * 100, 2),
        "total_segments":       len(segments),
        "segments_per_minute":  round(len(segments) / minutes, 2),
        "avg_segment_duration": round(covered / len(segments), 2),
    }

def _interpret_kappa(k: float) -> str:
    if k < 0.20: return "poor"
    if k < 0.40: return "fair"
    if k < 0.60: return "moderate"
    if k < 0.80: return "good"
    return "excellent"