"""Flask application factory. Training is an explicit, offline step."""

import os

from flask import Flask

from app.services.predictor import Predictor
from ml.utils import DEFAULT_MODEL_DIR


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        MODEL_DIR=os.environ.get("MODEL_DIR", str(DEFAULT_MODEL_DIR)),
        MAX_CONTENT_LENGTH=64 * 1024,
        MAX_TEXT_LENGTH=10_000,
    )
    if config is not None:
        app.config.update(config)

    # Fail clearly at startup if artifacts are unavailable; never train here.
    app.extensions["predictor"] = Predictor(app.config["MODEL_DIR"])

    from app.routes import main

    app.register_blueprint(main)
    return app
