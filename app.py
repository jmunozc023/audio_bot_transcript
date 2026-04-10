import subprocess
import os
import sys

from flask import Flask, request, jsonify, render_template
from rag.query_engine import ask_question

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPT_FILE = os.path.join("data", "call.txt")
VENV_PYTHON = os.path.join(BASE_DIR, "venv310", "Scripts", "python.exe")
PYTHON_EXECUTABLE = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable

index = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["GET"])
def ask():
    question = request.args.get("q")

    if not question:
        return jsonify({"error": "No se proporcionó una pregunta."}), 400
    answer = ask_question(question)
    return jsonify({"response": answer})


@app.route("/transcript", methods=["GET"])
def transcript():
    if not os.path.exists(TRANSCRIPT_FILE):
        return jsonify({"error": "No se encontro el transcript."}), 404

    with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as file:
        content = file.read()

    return jsonify({"transcript": content})

@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Dispara LoadAudio.py como subproceso.
    El RAG rebuildeará su índice automáticamente en la próxima consulta
    porque query_engine.py ya tiene lógica lazy (index = None al inicio).
    """
    try:
        result = subprocess.run(
            [PYTHON_EXECUTABLE, os.path.join(BASE_DIR, "LoadAudio.py")],
            capture_output=True,
            text=True,
            cwd=BASE_DIR,
            timeout=600      # 10 min máximo para audios largos
        )
        if result.returncode != 0:
            return jsonify({"error": result.stderr}), 500

        # Forzar rebuild del índice RAG en la próxima consulta
        import rag.query_engine as qe
        qe.index = None

        return jsonify({"status": "ok", "log": result.stdout})

    except subprocess.TimeoutExpired:
        return jsonify({"error": "Transcription timeout (>10 min)"}), 504

if __name__ == "__main__":
    app.run(port=8090)

