import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from data.dataset import data

MODEL_PATH = "app/models/model.pkl"
VECTORIZER_PATH = "app/models/vectorizer.pkl"


def train_and_save():
    df = pd.DataFrame(data)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df["text"])

    model = MultinomialNB()
    model.fit(X, df["label"])

    os.makedirs("app/models", exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)


def load_model():
    if not os.path.exists(MODEL_PATH):
        train_and_save()

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


model, vectorizer = load_model()


def predict_text(text):
    transformed = vectorizer.transform([text])
    prediction = model.predict(transformed)[0]
    return prediction