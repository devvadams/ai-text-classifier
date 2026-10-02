"""Train on the real SMS dataset; run with ``python -m ml.train``."""

import argparse
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from ml.evaluate import evaluate_model, print_report
from ml.utils import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_DIR,
    file_sha256,
    load_dataset,
    save_artifacts,
    split_dataset,
)


def train(
    data_path: str | Path = DEFAULT_DATASET_PATH,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Fit only on training messages, persist the bundle, and score the holdout."""
    data_path = Path(data_path)
    frame = load_dataset(data_path)
    train_frame, test_frame = split_dataset(
        frame, test_size=test_size, random_state=random_state
    )

    vectorizer = CountVectorizer(ngram_range=(1, 2))
    training_features = vectorizer.fit_transform(train_frame["text"])
    model = MultinomialNB(alpha=1.0)
    model.fit(training_features, train_frame["label"])
    metrics = evaluate_model(model, vectorizer, test_frame)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "filename": data_path.name,
            "sha256": file_sha256(data_path),
            "rows_after_deduplication": len(frame),
            "label_counts": {
                label: int(count)
                for label, count in frame["label"].value_counts().items()
            },
        },
        "split": {
            "random_state": random_state,
            "test_size": test_size,
            "train_rows": len(train_frame),
            "test_rows": len(test_frame),
            "test_indices": test_frame.index.tolist(),
        },
        "estimator": {"name": "MultinomialNB", "alpha": model.alpha},
        "vectorizer": {"name": "CountVectorizer", "ngram_range": [1, 2]},
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    save_artifacts(model, vectorizer, model_dir, metadata)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        metrics = train(
            args.data, args.model_dir, test_size=args.test_size, random_state=args.seed
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Saved model, vectorizer, and metadata to {args.model_dir.resolve()}")
    print_report(metrics)


if __name__ == "__main__":
    main()
