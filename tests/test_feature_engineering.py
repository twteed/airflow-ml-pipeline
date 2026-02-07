"""Tests for feature engineering module."""

import numpy as np

from ml_pipeline.feature_engineering import (
    add_engineered_features,
    build_features,
    create_preprocessor,
    load_preprocessor,
    save_preprocessor,
)


class TestAddEngineeredFeatures:
    def test_adds_expected_columns(self, sample_dataframe):
        df = add_engineered_features(sample_dataframe)
        assert "charge_per_tenure" in df.columns
        assert "tenure_bucket" in df.columns
        assert "high_value" in df.columns
        assert "support_intensity" in df.columns

    def test_preserves_original_columns(self, sample_dataframe):
        original_cols = set(sample_dataframe.columns)
        df = add_engineered_features(sample_dataframe)
        assert original_cols.issubset(set(df.columns))

    def test_no_nans_in_engineered(self, sample_dataframe):
        df = add_engineered_features(sample_dataframe)
        assert df["charge_per_tenure"].isna().sum() == 0
        assert df["support_intensity"].isna().sum() == 0


class TestCreatePreprocessor:
    def test_creates_column_transformer(self, sample_config):
        preprocessor = create_preprocessor(sample_config)
        assert hasattr(preprocessor, "fit_transform")

    def test_fit_transform_produces_array(self, sample_dataframe, sample_config):
        preprocessor = create_preprocessor(sample_config)
        df = add_engineered_features(sample_dataframe)

        extended_config = {
            "features": {
                **sample_config["features"],
                "numerical": sample_config["features"]["numerical"] + [
                    "charge_per_tenure", "support_intensity",
                ],
                "categorical": sample_config["features"]["categorical"] + [
                    "tenure_bucket",
                ],
            }
        }
        preprocessor = create_preprocessor(extended_config)
        X = preprocessor.fit_transform(df)
        assert isinstance(X, np.ndarray)
        assert X.shape[0] == len(df)
        assert X.shape[1] > 0


class TestBuildFeatures:
    def test_returns_correct_tuple(self, sample_dataframe, sample_config):
        X, y, preprocessor, feature_names = build_features(
            sample_dataframe, sample_config, fit=True,
        )
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(X) == len(y) == len(sample_dataframe)
        assert len(feature_names) == X.shape[1]

    def test_transform_mode(self, sample_dataframe, sample_config):
        X_fit, y_fit, preprocessor, _ = build_features(
            sample_dataframe, sample_config, fit=True,
        )
        X_transform, y_transform, _, _ = build_features(
            sample_dataframe, sample_config, fit=False, preprocessor=preprocessor,
        )
        assert X_fit.shape == X_transform.shape

    def test_raises_without_preprocessor_in_transform(self, sample_dataframe, sample_config):
        import pytest
        with pytest.raises(ValueError, match="preprocessor must be provided"):
            build_features(sample_dataframe, sample_config, fit=False)


class TestPreprocessorPersistence:
    def test_save_and_load(self, sample_dataframe, sample_config, tmp_path):
        _, _, preprocessor, _ = build_features(
            sample_dataframe, sample_config, fit=True,
        )
        path = str(tmp_path / "preprocessor.joblib")
        save_preprocessor(preprocessor, path)
        loaded = load_preprocessor(path)
        assert hasattr(loaded, "transform")
