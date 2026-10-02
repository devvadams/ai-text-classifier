"""Dataset validation, reproducible splitting, and trusted artifact persistence."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "sms_spam.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
LABELS = ("ham", "spam")
ARTIFACT_FILES = ("model.pkl", "vectorizer.pkl")
ARTIFACT_FORMAT_VERSION = 1


class ArtifactError(RuntimeError):
    """A model bundle is missing, incompatible, or corrupt."""


def load_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Validate a label/text CSV and remove duplicate messages before splitting.

    Labels must be ham or spam. Conflicting labels for the same trimmed message
    are rejected rather than silently choosing one. Literal strings like 'NA'
    remain messages, not pandas missing-value markers.
    """
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if not {"label", "text"}.issubset(frame.columns):
        raise ValueError("Dataset must contain 'label' and 'text' columns.")

    frame = frame.loc[:, ["label", "text"]].copy()
    frame["label"] = frame["label"].str.strip().str.lower()
    frame["text"] = frame["text"].str.strip()
    if frame.empty or frame["text"].eq("").any():
        raise ValueError("Dataset must contain non-empty messages.")
    if not frame["label"].isin(LABELS).all():
        raise ValueError("Dataset labels must be 'ham' or 'spam'.")
    if frame.groupby("text")["label"].nunique().gt(1).any():
        raise ValueError("Dataset contains conflicting labels for the same message.")

    frame = frame.drop_duplicates(subset="text").reset_index(drop=True)
    counts = frame["label"].value_counts()
    if any(counts.get(label, 0) < 2 for label in LABELS):
        raise ValueError("Dataset needs at least two unique messages per class.")
    return frame


def split_dataset(
    frame: pd.DataFrame, *, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a seeded, stratified holdout while retaining cleaned row indices."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1 (exclusive).")
    return train_test_split(
        frame, test_size=test_size, random_state=random_state, stratify=frame["label"]
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(payload: dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def save_artifacts(model, vectorizer, model_dir: str | Path, metadata: dict) -> None:
    """Stage a bundle and publish its checksum manifest last.

    Training is offline: do not replace artifacts while a server is starting.
    Existing workers keep their already-loaded model until they are restarted.
    """
    destination = Path(model_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".training-", dir=destination) as temporary:
        staging = Path(temporary)
        joblib.dump(model, staging / "model.pkl", compress=3)
        joblib.dump(vectorizer, staging / "vectorizer.pkl", compress=3)
        manifest = {
            **metadata,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "artifacts": {name: file_sha256(staging / name) for name in ARTIFACT_FILES},
        }
        save_json(manifest, staging / "metadata.json")
        for name in (*ARTIFACT_FILES, "metadata.json"):
            (staging / name).replace(destination / name)


def load_metadata(model_dir: str | Path = DEFAULT_MODEL_DIR) -> dict:
    directory = Path(model_dir)
    try:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if (
            not isinstance(metadata, dict)
            or metadata.get("format_version") != ARTIFACT_FORMAT_VERSION
        ):
            raise ValueError("Unsupported model metadata format.")
        return metadata
    except (OSError, ValueError) as error:
        raise ArtifactError(
            f"Cannot read model metadata in {directory}. "
            "Train artifacts first with: python -m ml.train"
        ) from error


def load_artifacts(model_dir: str | Path = DEFAULT_MODEL_DIR) -> tuple:
    """Load a compatible model/vectorizer pair; never train or download anything.

    Joblib uses pickle internally. Only load artifacts from trusted sources;
    checksums detect accidental changes, not maliciously supplied bundles.
    """
    directory = Path(model_dir)
    metadata = load_metadata(directory)
    try:
        if metadata["versions"]["scikit_learn"] != sklearn.__version__:
            raise ValueError(
                "scikit-learn version differs from the training environment."
            )
        for name in ARTIFACT_FILES:
            if file_sha256(directory / name) != metadata["artifacts"][name]:
                raise ValueError(f"Artifact checksum mismatch: {name}.")
        model = joblib.load(directory / "model.pkl")
        vectorizer = joblib.load(directory / "vectorizer.pkl")
        if set(model.classes_) != set(LABELS):
            raise ValueError("Model must classify 'ham' and 'spam'.")
        if model.n_features_in_ != len(vectorizer.get_feature_names_out()):
            raise ValueError("Model and vectorizer feature counts differ.")
    except Exception as error:
        raise ArtifactError(
            f"Cannot load model artifacts in {directory}: {error} "
            "Retrain with: python -m ml.train"
        ) from error
    return model, vectorizer
