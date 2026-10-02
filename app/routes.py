"""HTTP endpoints and request validation only."""

from flask import Blueprint, current_app, jsonify, request

main = Blueprint("main", __name__)


@main.get("/")
def home():
    return "AI Text Classifier API is running!"


@main.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "'text' must be a non-empty string."}), 400
    if len(text) > current_app.config["MAX_TEXT_LENGTH"]:
        limit = current_app.config["MAX_TEXT_LENGTH"]
        return jsonify({"error": f"'text' must not exceed {limit} characters."}), 400

    prediction = current_app.extensions["predictor"].predict(text)
    return jsonify({"input": text, "prediction": prediction})


@main.app_errorhandler(413)
def request_too_large(error):
    return jsonify({"error": "Request body is too large."}), 413
