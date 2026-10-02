import joblib
from config import MODEL_PATH, VECTORIZER_PATH

model = None
vectorizer = None

def load_model():
    global model, vectorizer

    if model is None or vectorizer is None:
        print("Loading model...")

        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)

        print("Model loaded successfully 🚀")


def preprocess(text):
    return text.lower().strip()
def predict(text, threshold=0.7):
    load_model()

    clean_text = preprocess(text)
    vec = vectorizer.transform([clean_text])

    probs = model.predict_proba(vec)[0]
    confidence = max(probs)
    prediction = model.classes_[probs.argmax()]

    if confidence < threshold:
        prediction = "uncertain"

    return prediction, float(confidence)