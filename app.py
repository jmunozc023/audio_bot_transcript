import os

from flask import Flask, request, jsonify, render_template
from rag.query_engine import ask_question
from LoadAudio import transcribe_audio

app = Flask(__name__)

TRANSCRIPT_FILE = os.path.join("data", "call.txt")

os.makedirs("data", exist_ok=True)

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
    if "audio" not in request.files:
        return jsonify({"error": "No se adjuntó ningún archivo de audio."}), 400
    file = request.files["audio"]
    filename = file.filename or ""

    if filename == "":
        return jsonify({"error": "Nombre de archivo vacío."}), 400
    
    allowed = {"mp3"}
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Formato .{ext} no soportado."}), 415
    
    audio_path = os.path.join("data", f"llamada.{ext}")
    file.save(audio_path)

    import LoadAudio
    LoadAudio.AUDIO_FILE = audio_path
    LoadAudio.OUTPUT_FILE = os.path.join("data", "call.txt")

    try:
        transcribe_audio()
        import rag.query_engine as qe
        qe.index = None
        return jsonify({"status": "ok"})
    
    except Exception as e:
        app.logger.exception("Error during transcription")
        return jsonify({"error": "Ocurrió un error durante la transcripción."}), 500

if __name__ == "__main__":
    app.run(port=8090)
