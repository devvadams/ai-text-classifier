"""Use the real dataset and isolate generated artifacts from the checkout."""

import pytest

from app import create_app
from ml.train import train


@pytest.fixture(scope="session")
def trained_bundle(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("trained-models")
    metrics = train(model_dir=model_dir)
    return model_dir, metrics


@pytest.fixture
def client(trained_bundle):
    model_dir, _ = trained_bundle
    app = create_app({"TESTING": True, "MODEL_DIR": model_dir})
    return app.test_client()
