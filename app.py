import os

from flask import Flask, request, jsonify, render_template
from rag.query_engine import ask_question

app = Flask(__name__)
TRANSCRIPT_FILE = os.path.join("data", "call.txt")

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


if __name__ == "__main__":
    app.run(port=8090)

