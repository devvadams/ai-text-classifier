"""Load a trained model once and serve predictions without refitting it."""

from pathlib import Path

from ml.utils import DEFAULT_MODEL_DIR, load_artifacts


class Predictor:
    """One reusable predictor per Flask application/worker."""

    API_LABELS = {"ham": "not spam", "spam": "spam"}

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR) -> None:
        self._model, self._vectorizer = load_artifacts(model_dir)

    def predict(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string.")
        features = self._vectorizer.transform([text])
        label = str(self._model.predict(features)[0])
        return self.API_LABELS[label]
