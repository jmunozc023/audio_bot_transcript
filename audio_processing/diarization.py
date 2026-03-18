from typing import Iterable, List, Sequence, Tuple
import librosa
import numpy as np
from sklearn.cluster import KMeans

def extract_embeddings_and_texts(audio_file: str, sample_rate: int, segments: Sequence, min_chunk_seconds: float,) -> Tuple[List[List[float]], List[str]]:
    audio, sr = librosa.load(audio_file, sr=sample_rate)

    embeddings: List[List[float]] = []
    texts: List[str] = []
    min_samples = int(sr*min_chunk_seconds)

    for seg in segments:
        start = int(seg.start * sr)
        end = int(seg.end * sr)
        chunk = audio[start:end]

        if len(chunk) < min_samples:
            continue

        mean = float(np.mean(chunk))
        std = float(np.std(chunk))
        energy = float(np.mean(np.square(chunk)))
        embeddings.append([mean, std, energy])
        texts.append(seg.text.strip())
    return embeddings, texts

def cluster_speakers(embeddings: Sequence[Sequence[float]], n_speakers: int) -> np.ndarray:
    if not embeddings:
        raise ValueError("No valid audio chunks found for clustering.")
    if len(embeddings) < n_speakers:
        labels = np.zeros(len(embeddings), dtype=int)
        for i in range(len(embeddings)):
            labels[i] = i % n_speakers
        return labels
    print("Clustering speakers...")
    model = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
    return model.fit_predict(embeddings)

def build_speaker_transcript(labels: Iterable[int], texts: Iterable[str]) -> str:
    lines = []
    for label, text in zip(labels, texts):
        lines.append(f"Speaker_{label}: {text}")
    return "\n".join(lines)

