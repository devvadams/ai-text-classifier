# import joblib
# from sklearn.metrics import accuracy_score, classification_report

# from ml.utils import load_and_clean_data
# from config import DATA_PATH, MODEL_PATH, VECTORIZER_PATH

# def evaluate():
#     print("Loading dataset...")
#     df = load_and_clean_data(DATA_PATH)

#     X = df["text"]
#     y = df["label"]

#     print("Loading model...")
#     model = joblib.load(MODEL_PATH)
#     vectorizer = joblib.load(VECTORIZER_PATH)

#     X_vec = vectorizer.transform(X)
#     predictions = model.predict(X_vec)

#     print("\n=== EVALUATION ===")
#     print("Accuracy:", accuracy_score(y, predictions))
#     print(classification_report(y, predictions))

# if __name__ == "__main__":
#     evaluate()
import joblib
from sklearn.metrics import accuracy_score

from ml.utils import load_and_clean_data
from config import DATA_PATH, MODEL_PATH, VECTORIZER_PATH


def evaluate(return_dict=False):
    print("Evaluating model...")

    # Load dataset
    df = load_and_clean_data(DATA_PATH)

    X = df["text"]
    y = df["label"]

    # Load trained model
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    # Transform text
    X_vec = vectorizer.transform(X)

    # Predict
    predictions = model.predict(X_vec)

    # Accuracy
    accuracy = accuracy_score(y, predictions)

    if return_dict:
        return {
            "accuracy": round(float(accuracy), 4),
            "samples": len(df)
        }

    print(f"Accuracy: {accuracy}")