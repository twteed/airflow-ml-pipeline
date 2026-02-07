"""Tests for model deployment module."""

import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from ml_pipeline.deployment import load_champion, predict, promote_model
from ml_pipeline.feature_engineering import build_features


class TestPromoteModel:
    def test_saves_model_artifacts(self, sample_dataframe, sample_config, tmp_path):
        X, y, preprocessor, feature_names = build_features(
            sample_dataframe, sample_config, fit=True,
        )
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)

        deploy_path = promote_model(
            model=model,
            preprocessor=preprocessor,
            model_name="logistic_regression",
            metrics={"f1": 0.80, "accuracy": 0.85},
            feature_names=feature_names,
            config=sample_config,
        )

        from pathlib import Path
        champion_dir = Path(deploy_path)
        assert (champion_dir / "model.joblib").exists()
        assert (champion_dir / "preprocessor.joblib").exists()
        assert (champion_dir / "metadata.json").exists()

        with open(champion_dir / "metadata.json") as f:
            metadata = json.load(f)
        assert metadata["model_name"] == "logistic_regression"
        assert metadata["metrics"]["f1"] == 0.80

    def test_archives_previous_champion(self, sample_dataframe, sample_config, tmp_path):
        X, y, preprocessor, feature_names = build_features(
            sample_dataframe, sample_config, fit=True,
        )
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)

        kwargs = dict(
            model=model, preprocessor=preprocessor,
            model_name="lr", metrics={"f1": 0.8},
            feature_names=feature_names, config=sample_config,
        )

        # First deploy
        promote_model(**kwargs)
        # Second deploy should archive the first
        promote_model(**kwargs)

        from pathlib import Path
        models_dir = Path(sample_config["deployment"]["champion_model_path"]).parent
        archives = [d for d in models_dir.iterdir() if d.name.startswith("archive_")]
        assert len(archives) == 1


class TestLoadChampion:
    def test_loads_deployed_model(self, sample_dataframe, sample_config, tmp_path):
        X, y, preprocessor, feature_names = build_features(
            sample_dataframe, sample_config, fit=True,
        )
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)

        promote_model(
            model=model, preprocessor=preprocessor,
            model_name="lr", metrics={"f1": 0.8},
            feature_names=feature_names, config=sample_config,
        )

        loaded_model, loaded_preprocessor, metadata = load_champion(sample_config)
        assert hasattr(loaded_model, "predict")
        assert metadata["model_name"] == "lr"

    def test_raises_when_no_model(self, sample_config):
        import pytest
        with pytest.raises(FileNotFoundError):
            load_champion(sample_config)


class TestPredict:
    def test_returns_predictions(self, sample_dataframe, sample_config, tmp_path):
        X, y, preprocessor, feature_names = build_features(
            sample_dataframe, sample_config, fit=True,
        )
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)

        predictions = predict(model, preprocessor, sample_dataframe)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(sample_dataframe)
        assert set(predictions).issubset({0, 1})
