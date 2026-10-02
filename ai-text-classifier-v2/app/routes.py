from flask import Blueprint, request, jsonify
from app.services.model_service import predict
from ml.evaluate import evaluate

api = Blueprint("api", __name__)

@api.route("/", methods=["GET"])
def home():
    return {"message": "AI Text Classifier API running 🚀"}

@api.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}
@api.route("/metrics", methods=["GET"])

def metrics():
    result = evaluate(return_dict=True)
    return result

@api.route("/predict", methods=["POST"])
def classify():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Text is required"}), 400

    text = data["text"]

    prediction, confidence = predict(text)

    return jsonify({
        "input": text,
        "prediction": prediction,
        "confidence": round(confidence, 4)
    })