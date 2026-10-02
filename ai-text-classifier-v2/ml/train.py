import os
import joblib
import logging

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from ml.utils import load_and_clean_data
from config import DATA_PATH, MODEL_PATH, VECTORIZER_PATH

logging.basicConfig(level=logging.INFO)

def train():
    logging.info("Training started...")

    df = load_and_clean_data(DATA_PATH)

    if df.empty:
        raise Exception("Dataset is empty after cleaning.")

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),   #  BIG upgrade (unigram + bigram)
    max_df=0.9,
    min_df=2 )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # Model (UPGRADED)
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"   #  handles spam imbalance
    )

    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, predictions)

    print("\n==============================")
    print(f"MODEL ACCURACY: {accuracy * 100:.2f}%")
    print("==============================\n")

    print(classification_report(y_test, predictions))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    logging.info("Training completed.")

if __name__ == "__main__":
    train()