# eval/eval_wer.py
import re
from jiwer import wer, cer, process_words

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def read_reference_transcript(path: str) -> str:
    """Lee una referencia en dos formatos:
    1) Diarizada: 'Hablante: texto [Xs - Ys] [sentiment score]'
    2) Texto plano: cualquier contenido sin etiquetas de hablante.
    """
    diarized_lines = []
    plain_lines = []

    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            plain_lines.append(line)

            if line.startswith("===") or "sentiment=" in line:
                continue

            match = re.match(r"^\w+:\s+(.+?)\s+\[\d", line)
            if match:
                diarized_lines.append(match.group(1))

    if diarized_lines:
        return " ".join(diarized_lines)

    return " ".join(plain_lines)

def compute_wer_metrics(reference_path: str, segments: list[dict]) -> dict:
    reference_raw = read_reference_transcript(reference_path)
    reference = normalize(reference_raw)
    hypothesis = normalize(" ".join(s["text"] for s in segments))

    if not reference:
        raise ValueError(
            "La referencia quedó vacía después del parseo/normalización. "
            "Verifica que el archivo de referencia no esté vacío y que su formato sea válido."
        )

    m = process_words(reference, hypothesis)
    return {
        "wer":             round(m.wer, 4),
        "cer":             round(cer(reference, hypothesis), 4),
        "substitutions":   m.substitutions,
        "deletions":       m.deletions,
        "insertions":      m.insertions,
        "reference_words": len(reference.split()),
    }