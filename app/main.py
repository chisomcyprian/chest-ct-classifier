import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ALLOWED = {".png", ".jpg", ".jpeg"}
DISCLAIMER = "Research demo only. Not a medical device and not for clinical decisions."


def create_app(predictor=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB uploads

    if predictor is None:
        from ctclf.inference import Predictor
        weights = os.environ.get("WEIGHTS", str(ROOT / "checkpoints" / "best_exp2.pt"))
        predictor = Predictor(weights)

    @app.get("/")
    def index():
        return render_template("index.html", disclaimer=DISCLAIMER)

    @app.get("/health")
    def health():
        return jsonify(status="ok", classes=predictor.classes)

    @app.post("/predict")
    def predict():
        file = request.files.get("file")
        if file is None or file.filename == "":
            return jsonify(error="Upload an image in the 'file' field."), 400
        if Path(file.filename).suffix.lower() not in ALLOWED:
            return jsonify(error="Only PNG or JPG images are supported."), 400
        explain = request.args.get("explain", "false").lower() == "true"
        try:
            result = predictor.predict(file.stream, explain=explain)
        except Exception:
            return jsonify(error="Could not read that image."), 422
        result["disclaimer"] = DISCLAIMER
        return jsonify(result)

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="File too large (max 10 MB)."), 413

    return app