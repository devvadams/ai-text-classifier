import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.feature_extraction.text import CountVectorizer

from ml.evaluate import evaluate
from ml.utils import (
    DEFAULT_DATASET_PATH,
    PROJECT_ROOT,
    ArtifactError,
    file_sha256,
    load_artifacts,
    load_dataset,
    load_metadata,
    split_dataset,
)

VALID_ROWS = [
    ("ham", "Lunch tomorrow"),
    ("ham", "See you tonight"),
    ("spam", "Claim your prize"),
    ("spam", "Free entry win cash"),
]


def write_dataset(tmp_path, rows, columns=("label", "text")):
    path = tmp_path / "sms.csv"
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def test_real_uci_dataset_is_included():
    raw = pd.read_csv(DEFAULT_DATASET_PATH, keep_default_na=False)
    assert list(raw.columns) == ["label", "text"]
    assert len(raw) == 5574
    assert raw["label"].value_counts().to_dict() == {"ham": 4827, "spam": 747}
    frame = load_dataset()
    assert len(frame) == 5160
    assert frame["text"].is_unique


def test_normalization_deduplication_and_literal_na(tmp_path):
    rows = VALID_ROWS + [(" HAM ", " Lunch tomorrow "), ("ham", "NA")]
    frame = load_dataset(write_dataset(tmp_path, rows))
    assert len(frame) == 5
    assert "NA" in frame["text"].tolist()
    train_frame, test_frame = split_dataset(frame, test_size=0.4)
    assert set(train_frame["text"]).isdisjoint(test_frame["text"])
    assert set(train_frame["label"]) == set(test_frame["label"]) == {"ham", "spam"}


@pytest.mark.parametrize(
    ("rows", "columns", "error"),
    [
        (VALID_ROWS, ("class", "message"), "columns"),
        ([], ("label", "text"), "non-empty"),
        (VALID_ROWS + [("ham", " \t")], ("label", "text"), "non-empty"),
        (VALID_ROWS + [("other", "hi")], ("label", "text"), "labels must"),
        (VALID_ROWS + [("", "hi")], ("label", "text"), "labels must"),
        (VALID_ROWS + [("spam", "Lunch tomorrow")], ("label", "text"), "conflicting"),
        (VALID_ROWS[:2], ("label", "text"), "two unique"),
        (
            [VALID_ROWS[0], VALID_ROWS[0], VALID_ROWS[2]],
            ("label", "text"),
            "two unique",
        ),
    ],
)
def test_invalid_datasets_are_rejected(tmp_path, rows, columns, error):
    with pytest.raises(ValueError, match=error):
        load_dataset(write_dataset(tmp_path, rows, columns))


@pytest.mark.parametrize("test_size", [0, 1, -0.1, 1.5, float("nan")])
def test_invalid_holdout_sizes(test_size):
    with pytest.raises(ValueError, match="between 0 and 1"):
        split_dataset(load_dataset(), test_size=test_size)


def test_training_has_no_duplicate_or_vocabulary_leakage(trained_bundle):
    model_dir, metrics = trained_bundle
    frame = load_dataset()
    train_frame, test_frame = split_dataset(frame)
    metadata = load_metadata(model_dir)
    model, vectorizer = load_artifacts(model_dir)
    assert metadata["split"]["test_indices"] == test_frame.index.tolist()
    assert metadata["dataset"]["sha256"] == file_sha256(DEFAULT_DATASET_PATH)
    assert metadata["split"]["train_rows"] == 4128
    assert metrics["samples"] == 1032
    assert set(train_frame["text"]).isdisjoint(test_frame["text"])
    expected = CountVectorizer(ngram_range=(1, 2)).fit(train_frame["text"])
    assert vectorizer.vocabulary_ == expected.vocabulary_
    assert model.n_features_in_ == len(expected.vocabulary_)
    assert model.class_count_.sum() == len(train_frame)
    assert metrics["accuracy"] > 0.90
    assert metrics["classification_report"]["spam"]["f1-score"] > 0.80


def test_evaluation_reuses_exact_unseen_rows(trained_bundle):
    model_dir, training_metrics = trained_bundle
    assert evaluate(model_dir=model_dir) == training_metrics


def test_evaluation_rejects_different_dataset(trained_bundle, tmp_path):
    modified = tmp_path / "modified.csv"
    modified.write_bytes(DEFAULT_DATASET_PATH.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="differs from the training dataset"):
        evaluate(modified, trained_bundle[0])


@pytest.mark.parametrize("filename", ["model.pkl", "vectorizer.pkl", "metadata.json"])
def test_missing_bundle_files_are_not_retrained(trained_bundle, tmp_path, filename):
    directory = tmp_path / "models"
    shutil.copytree(trained_bundle[0], directory)
    (directory / filename).unlink()
    with pytest.raises(ArtifactError, match="python -m ml.train"):
        load_artifacts(directory)
    assert not (directory / filename).exists()


def test_corrupt_artifacts_are_detected(trained_bundle, tmp_path):
    directory = tmp_path / "models"
    shutil.copytree(trained_bundle[0], directory)
    (directory / "vectorizer.pkl").write_bytes(b"corrupt")
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        load_artifacts(directory)


def test_version_mismatch_is_actionable(trained_bundle, tmp_path):
    directory = tmp_path / "models"
    shutil.copytree(trained_bundle[0], directory)
    metadata = load_metadata(directory)
    metadata["versions"]["scikit_learn"] = "0.0.0"
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ArtifactError, match="version differs"):
        load_artifacts(directory)


@pytest.mark.parametrize("contents", ["not json", "[]", '{"format_version": 999}'])
def test_invalid_metadata_is_actionable(tmp_path, contents):
    (tmp_path / "metadata.json").write_text(contents, encoding="utf-8")
    with pytest.raises(ArtifactError, match="python -m ml.train"):
        load_metadata(tmp_path)


@pytest.mark.parametrize("indices", [[], [True], [999_999], [0, 0], "invalid"])
def test_evaluation_rejects_invalid_holdout_indices(trained_bundle, tmp_path, indices):
    directory = tmp_path / "models"
    shutil.copytree(trained_bundle[0], directory)
    metadata = load_metadata(directory)
    metadata["split"]["test_indices"] = indices
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="holdout indices"):
        evaluate(model_dir=directory)


def run_module(module, *args, cwd):
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    return subprocess.run(
        [sys.executable, "-m", module, *map(str, args)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_training_and_evaluation_cli_from_another_directory(tmp_path):
    directory = tmp_path / "models"
    report = tmp_path / "reports" / "evaluation.json"
    training = run_module(
        "ml.train",
        "--model-dir",
        directory,
        "--seed",
        7,
        "--test-size",
        0.25,
        cwd=tmp_path,
    )
    assert training.returncode == 0, training.stderr
    result = run_module(
        "ml.evaluate", "--model-dir", directory, "--output", report, cwd=tmp_path
    )
    assert result.returncode == 0, result.stderr
    metrics = json.loads(result.stdout)
    assert metrics == json.loads(report.read_text(encoding="utf-8"))
    assert metrics["samples"] == 1290
    metadata = load_metadata(directory)
    assert metadata["split"]["random_state"] == 7
    assert metadata["split"]["test_size"] == 0.25


def test_cli_errors_are_clear_and_do_not_create_artifacts(tmp_path):
    directory = tmp_path / "models"
    result = run_module(
        "ml.train",
        "--data",
        tmp_path / "missing.csv",
        "--model-dir",
        directory,
        cwd=tmp_path,
    )
    assert result.returncode == 2
    assert "No such file" in result.stderr
    assert "Traceback" not in result.stderr
    assert not directory.exists()
    result = run_module("ml.evaluate", "--model-dir", directory, cwd=tmp_path)
    assert result.returncode == 2
    assert "python -m ml.train" in result.stderr
    assert "Traceback" not in result.stderr


def test_imports_have_no_training_side_effects(tmp_path):
    env = {
        **os.environ,
        "PYTHONPATH": str(PROJECT_ROOT),
        "MODEL_DIR": str(tmp_path / "models"),
    }
    result = subprocess.run(
        [sys.executable, "-c", "import app; import ml.train; import ml.evaluate"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert list(Path(tmp_path).iterdir()) == []
