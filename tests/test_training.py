"""Tests for model training module."""

import numpy as np

from ml_pipeline.feature_engineering import build_features
from ml_pipeline.training import train_all_models, train_model


class TestTrainModel:
    def test_logistic_regression(self, sample_dataframe, sample_config):
        X, y, _, _ = build_features(sample_dataframe, sample_config, fit=True)
        model, cv_results = train_model(
            model_name="logistic_regression",
            X_train=X,
            y_train=y,
            param_grid={"C": [1.0], "penalty": ["l2"], "max_iter": [200]},
            cv_folds=3,
        )
        assert hasattr(model, "predict")
        assert "best_cv_score" in cv_results
        assert cv_results["best_cv_score"] > 0

    def test_random_forest(self, sample_dataframe, sample_config):
        X, y, _, _ = build_features(sample_dataframe, sample_config, fit=True)
        model, cv_results = train_model(
            model_name="random_forest",
            X_train=X,
            y_train=y,
            param_grid={"n_estimators": [50], "max_depth": [5],
                        "min_samples_split": [2], "min_samples_leaf": [1]},
            cv_folds=3,
        )
        assert hasattr(model, "predict_proba")
        assert cv_results["best_cv_score"] > 0

    def test_unknown_model_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown model"):
            train_model("nonexistent", np.array([]), np.array([]), {})


class TestTrainAllModels:
    def test_trains_enabled_models(self, sample_dataframe, sample_config):
        X, y, _, _ = build_features(sample_dataframe, sample_config, fit=True)
        results = train_all_models(X, y, sample_config)
        assert "logistic_regression" in results
        assert "random_forest" in results
        assert "model" in results["logistic_regression"]
        assert "cv_results" in results["logistic_regression"]

    def test_skips_disabled_models(self, sample_dataframe, sample_config):
        sample_config["training"]["models"]["random_forest"]["enabled"] = False
        X, y, _, _ = build_features(sample_dataframe, sample_config, fit=True)
        results = train_all_models(X, y, sample_config)
        assert "random_forest" not in results
        assert "logistic_regression" in results
