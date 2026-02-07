"""Model deployment module — promotes the champion model and serves predictions."""

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def promote_model(
    model,
    preprocessor,
    model_name: str,
    metrics: dict,
    feature_names: list[str],
    config: dict,
) -> str:
    """Promote the champion model to the deployment directory.

    Args:
        model: Trained sklearn estimator.
        preprocessor: Fitted ColumnTransformer.
        model_name: Name of the champion model.
        metrics: Evaluation metrics dict.
        feature_names: List of feature names.
        config: Pipeline configuration.

    Returns:
        Path to the deployed model directory.
    """
    deploy_cfg = config["deployment"]
    champion_dir = Path(deploy_cfg["champion_model_path"])

    # Archive previous champion if exists
    if champion_dir.exists():
        archive_name = f"archive_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        archive_dir = champion_dir.parent / archive_name
        shutil.move(str(champion_dir), str(archive_dir))
        logger.info("Archived previous champion to %s", archive_dir)

    champion_dir.mkdir(parents=True, exist_ok=True)

    # Save model artifacts
    joblib.dump(model, champion_dir / "model.joblib")
    joblib.dump(preprocessor, champion_dir / "preprocessor.joblib")

    # Save metadata
    metadata = {
        "model_name": model_name,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        "feature_names": feature_names,
        "model_class": type(model).__name__,
        "model_params": model.get_params(),
    }

    with open(champion_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("Promoted %s as champion model to %s", model_name, champion_dir)
    return str(champion_dir)


def load_champion(config: dict) -> tuple:
    """Load the deployed champion model and preprocessor.

    Returns:
        Tuple of (model, preprocessor, metadata).
    """
    champion_dir = Path(config["deployment"]["champion_model_path"])

    if not champion_dir.exists():
        raise FileNotFoundError(f"No champion model found at {champion_dir}")

    model = joblib.load(champion_dir / "model.joblib")
    preprocessor = joblib.load(champion_dir / "preprocessor.joblib")

    with open(champion_dir / "metadata.json") as f:
        metadata = json.load(f)

    logger.info("Loaded champion model: %s", metadata["model_name"])
    return model, preprocessor, metadata


def predict(model, preprocessor, input_data: pd.DataFrame) -> np.ndarray:
    """Run inference on new data using the deployed model."""
    from ml_pipeline.feature_engineering import add_engineered_features

    input_data = add_engineered_features(input_data)
    X = preprocessor.transform(input_data)
    return model.predict(X)


def predict_proba(model, preprocessor, input_data: pd.DataFrame) -> np.ndarray:
    """Run inference returning churn probability scores."""
    from ml_pipeline.feature_engineering import add_engineered_features

    input_data = add_engineered_features(input_data)
    X = preprocessor.transform(input_data)
    return model.predict_proba(X)[:, 1]


def create_flask_app(config: dict):
    """Create a Flask app for serving predictions.

    Returns:
        Flask application instance.
    """
    from flask import Flask, jsonify, request

    app = Flask(__name__)
    model, preprocessor, metadata = load_champion(config)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "healthy", "model": metadata["model_name"]})

    @app.route("/predict", methods=["POST"])
    def predict_endpoint():
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        try:
            df = pd.DataFrame([data] if isinstance(data, dict) else data)
            predictions = predict(model, preprocessor, df)
            probabilities = predict_proba(model, preprocessor, df)
            return jsonify({
                "predictions": predictions.tolist(),
                "probabilities": probabilities.tolist(),
            })
        except Exception as e:
            logger.exception("Prediction failed")
            return jsonify({"error": str(e)}), 500

    @app.route("/model/info", methods=["GET"])
    def model_info():
        return jsonify(metadata)

    return app
