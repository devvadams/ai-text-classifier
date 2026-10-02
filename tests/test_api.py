import json
from unittest.mock import patch

import pytest

from app import create_app
from app.services.predictor import Predictor
from ml.utils import ArtifactError, load_artifacts

SPAM_MESSAGE = (
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. "
    "Text FA to 87121 to receive entry question(std txt rate)T&C's apply "
    "08452810075over18's"
)


def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.get_data(as_text=True) == "AI Text Classifier API is running!"


@pytest.mark.parametrize(
    ("text", "prediction"),
    [(SPAM_MESSAGE, "spam"), ("Are we still meeting for lunch tomorrow?", "not spam")],
)
def test_predictions_keep_existing_api_contract(client, text, prediction):
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200
    assert response.get_json() == {"input": text, "prediction": prediction}


def test_original_text_is_preserved(client):
    text = "  See you tomorrow! ☕  "
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200
    assert response.get_json()["input"] == text


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        ["hello"],
        "hello",
        7,
        True,
        {},
        {"text": None},
        {"text": ""},
        {"text": " \t\n"},
        {"text": 12},
        {"text": False},
        {"text": ["hi"]},
        {"text": {"value": "hi"}},
    ],
)
def test_invalid_json_payloads_return_400(client, payload):
    with patch.object(client.application.extensions["predictor"], "predict") as predict:
        response = client.post(
            "/predict", data=json.dumps(payload), content_type="application/json"
        )
    assert response.status_code == 400
    assert "error" in response.get_json()
    predict.assert_not_called()


@pytest.mark.parametrize(
    ("body", "content_type"),
    [
        ("", "application/json"),
        ('{"text":', "application/json"),
        ('{"text":"hello"}', "text/plain"),
    ],
)
def test_missing_malformed_or_non_json_body(client, body, content_type):
    response = client.post("/predict", data=body, content_type=content_type)
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_limits_are_enforced(client):
    response = client.post("/predict", json={"text": "x" * 10_001})
    assert response.status_code == 400
    assert "10000" in response.get_json()["error"]
    response = client.post("/predict", json={"text": "x" * (64 * 1024)})
    assert response.status_code == 413
    assert response.get_json() == {"error": "Request body is too large."}


def test_loads_once_and_never_refits(trained_bundle):
    model_dir, _ = trained_bundle
    with patch("app.services.predictor.load_artifacts", wraps=load_artifacts) as load:
        app = create_app({"TESTING": True, "MODEL_DIR": model_dir})
        predictor = app.extensions["predictor"]
        with (
            patch.object(
                predictor._model, "fit", side_effect=AssertionError("No training")
            ),
            patch.object(
                predictor._vectorizer, "fit", side_effect=AssertionError("No fitting")
            ),
            patch.object(
                predictor._vectorizer,
                "fit_transform",
                side_effect=AssertionError("No fitting"),
            ),
        ):
            client = app.test_client()
            assert (
                client.post("/predict", json={"text": "Hello there"}).status_code == 200
            )
            assert (
                client.post("/predict", json={"text": SPAM_MESSAGE}).status_code == 200
            )
        assert load.call_count == 1


def test_missing_artifacts_fail_at_startup_without_training(tmp_path):
    directory = tmp_path / "missing-models"
    with patch("ml.train.train") as train:
        with pytest.raises(ArtifactError, match="python -m ml.train"):
            create_app({"MODEL_DIR": directory})
    train.assert_not_called()
    assert not directory.exists()


def test_model_dir_environment_override(monkeypatch, trained_bundle):
    model_dir, _ = trained_bundle
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    app = create_app({"TESTING": True})
    assert app.test_client().post("/predict", json={"text": "Hello"}).status_code == 200


def test_factory_configuration_is_not_shared(trained_bundle):
    model_dir, _ = trained_bundle
    short_app = create_app(
        {"TESTING": True, "MODEL_DIR": model_dir, "MAX_TEXT_LENGTH": 5}
    )
    default_app = create_app({"TESTING": True, "MODEL_DIR": model_dir})
    assert (
        short_app.test_client()
        .post("/predict", json={"text": "Hello there"})
        .status_code
        == 400
    )
    assert (
        default_app.test_client()
        .post("/predict", json={"text": "Hello there"})
        .status_code
        == 200
    )


@pytest.mark.parametrize("text", [None, 1, [], "", " \n"])
def test_predictor_rejects_non_text_inputs(trained_bundle, text):
    predictor = Predictor(trained_bundle[0])
    with pytest.raises(ValueError, match="non-empty string"):
        predictor.predict(text)
