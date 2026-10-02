"""Report metrics on the saved, unseen holdout; run with ``python -m ml.evaluate``."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from ml.utils import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_DIR,
    LABELS,
    ArtifactError,
    file_sha256,
    load_artifacts,
    load_dataset,
    load_metadata,
    save_json,
)


def evaluate_model(model, vectorizer, frame: pd.DataFrame) -> dict:
    """Transform, but never fit, evaluation messages."""
    predictions = model.predict(vectorizer.transform(frame["text"]))
    return {
        "samples": len(frame),
        "accuracy": float(accuracy_score(frame["label"], predictions)),
        "classification_report": classification_report(
            frame["label"],
            predictions,
            labels=list(LABELS),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": {
            "labels": list(LABELS),
            "matrix": confusion_matrix(
                frame["label"], predictions, labels=list(LABELS)
            ).tolist(),
        },
    }


def evaluate(
    data_path: str | Path = DEFAULT_DATASET_PATH,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict:
    """Use the exact test rows saved by training, not a fresh random split."""
    metadata = load_metadata(model_dir)
    try:
        if file_sha256(data_path) != metadata["dataset"]["sha256"]:
            raise ValueError(
                "Dataset differs from the training dataset. Retrain first."
            )
        frame = load_dataset(data_path)
        indices = metadata["split"]["test_indices"]
        if (
            not isinstance(indices, list)
            or not indices
            or any(
                type(index) is not int or not 0 <= index < len(frame)
                for index in indices
            )
            or len(set(indices)) != len(indices)
            or len(indices) != metadata["split"]["test_rows"]
            or len(frame) != metadata["dataset"]["rows_after_deduplication"]
            or metadata["split"]["train_rows"] + len(indices) != len(frame)
        ):
            raise ValueError("Saved holdout indices are invalid. Retrain first.")
    except (KeyError, TypeError) as error:
        raise ArtifactError(
            "Invalid evaluation metadata. Retrain with: python -m ml.train"
        ) from error
    model, vectorizer = load_artifacts(model_dir)
    return evaluate_model(model, vectorizer, frame.iloc[indices])


def print_report(metrics: dict) -> None:
    """JSON includes per-class precision/recall/F1 and a labeled confusion matrix."""
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, help="Optional JSON report destination.")
    args = parser.parse_args()
    try:
        metrics = evaluate(args.data, args.model_dir)
        if args.output:
            save_json(metrics, args.output)
    except (OSError, ValueError, ArtifactError) as error:
        parser.error(str(error))
    print_report(metrics)


if __name__ == "__main__":
    main()
