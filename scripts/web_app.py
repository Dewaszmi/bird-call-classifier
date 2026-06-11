#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from predict import (
    DEFAULT_CHECKPOINT,
    DEFAULT_MIN_MARGIN,
    DEFAULT_MIN_TOP_PROBABILITY,
    default_device,
    predict_audio,
)

app = Flask(
    __name__,
    template_folder=str(ROOT / "bird-call-classifier" / "templates"),
)
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.get("/")
def index():
    return render_template("index.html")


def get_checkpoint() -> Path:
    return Path(app.config.get("CHECKPOINT", DEFAULT_CHECKPOINT))


@app.post("/predict")
def predict():
    checkpoint = get_checkpoint()
    if not checkpoint.exists():
        return jsonify(
            {"error": f"Model checkpoint not found: {checkpoint}. Train the model first."}
        ), 503

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
        result = predict_audio(
            temp_path,
            checkpoint,
            app.config.get("DEVICE"),
            min_margin=app.config.get("MIN_MARGIN", DEFAULT_MIN_MARGIN),
            min_top_probability=app.config.get(
                "MIN_TOP_PROBABILITY", DEFAULT_MIN_TOP_PROBABILITY
            ),
        )
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
            "identified": result["identified"],
            "confidence": result["confidence"],
            "thresholds": {
                "min_margin": app.config.get("MIN_MARGIN", DEFAULT_MIN_MARGIN),
                "min_top_probability": app.config.get(
                    "MIN_TOP_PROBABILITY", DEFAULT_MIN_TOP_PROBABILITY
                ),
            },
            "predictions": result["predictions"],
            "other": result["other"],
            "chart": {
                "labels": chart_labels,
                "values": chart_values,
            },
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Web UI for bird species prediction from audio."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--device",
        default=default_device(),
        help="Inference device",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=DEFAULT_MIN_MARGIN,
        help=(
            "Minimum gap between the top two class probabilities required to "
            f"accept a prediction (default: {DEFAULT_MIN_MARGIN})"
        ),
    )
    parser.add_argument(
        "--min-probability",
        type=float,
        default=DEFAULT_MIN_TOP_PROBABILITY,
        help=(
            "Minimum top-class probability required to accept a prediction "
            f"(default: {DEFAULT_MIN_TOP_PROBABILITY})"
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the web server to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind the web server to (default: 5000)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.checkpoint.exists():
        print(
            f"Error: Checkpoint not found: {args.checkpoint}. Please train the model first."
        )
        return 1

    app.config["CHECKPOINT"] = str(args.checkpoint)
    app.config["DEVICE"] = args.device
    app.config["MIN_MARGIN"] = args.min_margin
    app.config["MIN_TOP_PROBABILITY"] = args.min_probability

    print(f"Using device: {args.device}")
    print(f"Loading checkpoint from {args.checkpoint}")
    print(f"Open http://{args.host}:{args.port} in your browser")
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
