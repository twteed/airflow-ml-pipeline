"""Tests for data ingestion module."""

import pandas as pd
import pytest

from ml_pipeline.data_ingestion import generate_synthetic_data, ingest_data, validate_data


class TestGenerateSyntheticData:
    def test_returns_correct_shape(self):
        df = generate_synthetic_data(n_samples=100, random_state=42)
        assert len(df) == 100
        assert df.shape[1] == 9

    def test_has_expected_columns(self):
        df = generate_synthetic_data(n_samples=50)
        expected = {
            "tenure", "contract_type", "payment_method", "internet_service",
            "monthly_charges", "total_charges", "num_support_tickets",
            "avg_monthly_usage_gb", "churn",
        }
        assert set(df.columns) == expected

    def test_target_is_binary(self):
        df = generate_synthetic_data(n_samples=500)
        assert set(df["churn"].unique()).issubset({0, 1})

    def test_reproducibility(self):
        df1 = generate_synthetic_data(n_samples=100, random_state=42)
        df2 = generate_synthetic_data(n_samples=100, random_state=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_value_ranges(self):
        df = generate_synthetic_data(n_samples=1000)
        assert (df["tenure"] >= 1).all()
        assert (df["tenure"] <= 72).all()
        assert (df["monthly_charges"] >= 0).all()
        assert (df["total_charges"] >= 0).all()

    def test_categorical_values(self):
        df = generate_synthetic_data(n_samples=1000)
        assert set(df["contract_type"].unique()) == {
            "month-to-month", "one-year", "two-year",
        }
        assert set(df["internet_service"].unique()) == {
            "fiber_optic", "dsl", "none",
        }


class TestValidateData:
    def test_valid_data_passes(self, sample_dataframe):
        result = validate_data(sample_dataframe)
        assert result["passed"] is True

    def test_missing_column_fails(self, sample_dataframe):
        df = sample_dataframe.drop(columns=["churn"])
        with pytest.raises(ValueError, match="schema_valid"):
            validate_data(df)

    def test_too_few_samples_fails(self):
        df = generate_synthetic_data(n_samples=50)
        # This should still pass since 50 < 100 threshold
        with pytest.raises(ValueError, match="sufficient_samples"):
            validate_data(df)

    def test_non_binary_target_fails(self, sample_dataframe):
        df = sample_dataframe.copy()
        df.loc[0, "churn"] = 5
        with pytest.raises(ValueError, match="target_binary"):
            validate_data(df)


class TestIngestData:
    def test_generates_and_saves_data(self, sample_config):
        df = ingest_data(sample_config)
        assert len(df) == 500
        assert pd.read_csv(sample_config["data"]["raw_data_path"]) is not None

    def test_loads_existing_data(self, sample_config):
        # First call generates and saves
        df1 = ingest_data(sample_config)
        # Second call loads from disk
        df2 = ingest_data(sample_config)
        assert len(df1) == len(df2)
