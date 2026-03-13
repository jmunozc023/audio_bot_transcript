import os
import numpy as np
import librosa
from faster_whisper import WhisperModel
from sklearn.cluster import KMeans
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

AUDIO_FILE = "data/llamada.mp3"
OUTPUT_FILE = "data/call.txt"

print("Loading ASR model...")
model = WhisperModel("base", device="cpu", compute_type="int8")


def transcribe_audio():

    print("Transcribing audio...")

    segments, _ = model.transcribe(AUDIO_FILE)

    audio, sr = librosa.load(AUDIO_FILE, sr=16000)

    embeddings = []
    texts = []

    for seg in segments:

        start = int(seg.start * sr)
        end = int(seg.end * sr)

        chunk = audio[start:end]

        if len(chunk) < sr * 0.5:
            continue

        # simple voice embedding
        embedding = np.mean(chunk)
        embeddings.append([embedding])

        texts.append(seg.text.strip())

    print("Clustering speakers...")

    kmeans = KMeans(n_clusters=2)
    labels = kmeans.fit_predict(embeddings)

    transcript_lines = []

    for label, text in zip(labels, texts):

        speaker = f"Speaker_{label}"
        transcript_lines.append(f"{speaker}: {text}")

    transcript = "\n".join(transcript_lines)

    print("Identifying Agent vs Customer...")

    prompt = f"""
    The following call transcript contains two speakers: Speaker_0 and Speaker_1.

    Identify which one is the Agent and which one is the Customer.

    Rewrite the transcript replacing Speaker_0 and Speaker_1.

    Transcript:
    {transcript}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You analyze call center conversations."},
            {"role": "user", "content": prompt}
        ]
    )

    final_transcript = response.choices[0].message.content

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(final_transcript)

    print("call.txt generated")


if __name__ == "__main__":
    transcribe_audio()