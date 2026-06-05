#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from predict import DEFAULT_CHECKPOINT, predict_audio

app = Flask(__name__)
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    if not DEFAULT_CHECKPOINT.exists():
        return jsonify({"error": "Model checkpoint not found. Train the model first."}), 503

    uploaded = request.files.get("audio")
    if uploaded is None or uploaded.filename == "":
        return jsonify({"error": "No audio file provided."}), 400

    if not allowed_file(uploaded.filename):
        return jsonify({"error": "Unsupported file type."}), 400

    suffix = Path(uploaded.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        uploaded.save(tmp.name)
        temp_path = Path(tmp.name)

    try:
        result = predict_audio(temp_path)
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500
    finally:
        temp_path.unlink(missing_ok=True)

    chart_labels = [entry["species"] for entry in result["predictions"]]
    chart_values = [entry["probability"] for entry in result["predictions"]]
    if result["other"] > 0:
        chart_labels.append("Other")
        chart_values.append(result["other"])

    return jsonify(
        {
            "best_match": result["best_match"],
            "predictions": result["predictions"],
            "other": result["other"],
            "chart": {
                "labels": chart_labels,
                "values": chart_values,
            },
        }
    )


def main() -> int:
    if not DEFAULT_CHECKPOINT.exists():
        print(f"Warning: checkpoint not found at {DEFAULT_CHECKPOINT}")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(host="127.0.0.1", port=5000, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
