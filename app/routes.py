from flask import Blueprint, request, jsonify
from app.model import predict_text

main = Blueprint("main", __name__)

@main.route("/")
def home():
    return "AI Text Classifier API is running!"

@main.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    prediction = predict_text(text)

    return jsonify({
        "input": text,
        "prediction": prediction
    })