"""Data ingestion module — generates synthetic customer churn data and validates it."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic customer churn dataset.

    Creates a realistic dataset with correlated features that mirror
    real telecom customer churn patterns.

    Args:
        n_samples: Number of customer records to generate.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with customer features and churn label.
    """
    rng = np.random.RandomState(random_state)

    # Customer tenure in months (0-72)
    tenure = rng.exponential(scale=24, size=n_samples).clip(1, 72).astype(int)

    # Contract type influences churn heavily
    contract_type = rng.choice(
        ["month-to-month", "one-year", "two-year"],
        size=n_samples,
        p=[0.50, 0.30, 0.20],
    )

    # Payment method
    payment_method = rng.choice(
        ["electronic_check", "mailed_check", "bank_transfer", "credit_card"],
        size=n_samples,
        p=[0.35, 0.20, 0.25, 0.20],
    )

    # Internet service type
    internet_service = rng.choice(
        ["fiber_optic", "dsl", "none"],
        size=n_samples,
        p=[0.45, 0.35, 0.20],
    )

    # Monthly charges — correlated with service type
    base_charge = np.where(
        internet_service == "fiber_optic",
        rng.normal(80, 15, n_samples),
        np.where(
            internet_service == "dsl",
            rng.normal(55, 10, n_samples),
            rng.normal(25, 5, n_samples),
        ),
    ).clip(18, 120)

    monthly_charges = np.round(base_charge, 2)
    total_charges = np.round(monthly_charges * tenure + rng.normal(0, 50, n_samples), 2).clip(0)

    # Support tickets — higher for churners
    num_support_tickets = rng.poisson(lam=1.5, size=n_samples)

    # Average monthly data usage in GB
    avg_monthly_usage_gb = np.where(
        internet_service == "none",
        0,
        rng.exponential(scale=15, size=n_samples).clip(0.5, 100),
    ).round(2)

    # Churn probability — influenced by multiple factors
    churn_logit = (
        -1.5
        + 0.8 * (contract_type == "month-to-month").astype(float)
        - 0.5 * (contract_type == "two-year").astype(float)
        + 0.4 * (payment_method == "electronic_check").astype(float)
        - 0.02 * tenure
        + 0.01 * monthly_charges
        + 0.15 * num_support_tickets
        - 0.01 * avg_monthly_usage_gb
        + rng.normal(0, 0.3, n_samples)
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn = (rng.uniform(size=n_samples) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "tenure": tenure,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "internet_service": internet_service,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_support_tickets": num_support_tickets,
            "avg_monthly_usage_gb": avg_monthly_usage_gb,
            "churn": churn,
        }
    )

    logger.info(
        "Generated %d samples — churn rate: %.1f%%",
        len(df),
        df["churn"].mean() * 100,
    )
    return df


def validate_data(df: pd.DataFrame) -> dict:
    """Run data quality checks on the ingested dataset.

    Args:
        df: Input DataFrame to validate.

    Returns:
        Dictionary with validation results.

    Raises:
        ValueError: If critical validation checks fail.
    """
    results = {"passed": True, "checks": {}}

    # Check for missing values
    missing = df.isnull().sum()
    results["checks"]["no_missing_values"] = int(missing.sum()) == 0
    if missing.sum() > 0:
        logger.warning("Missing values found:\n%s", missing[missing > 0])

    # Check for duplicate rows
    n_dupes = df.duplicated().sum()
    results["checks"]["no_duplicates"] = int(n_dupes) == 0
    if n_dupes > 0:
        logger.warning("Found %d duplicate rows", n_dupes)

    # Check expected columns exist
    expected_cols = {
        "tenure", "contract_type", "payment_method", "internet_service",
        "monthly_charges", "total_charges", "num_support_tickets",
        "avg_monthly_usage_gb", "churn",
    }
    schema_valid = expected_cols.issubset(set(df.columns))
    results["checks"]["schema_valid"] = schema_valid

    # Remaining checks require schema to be valid
    if schema_valid:
        results["checks"]["tenure_range"] = bool(df["tenure"].between(0, 100).all())
        results["checks"]["charges_positive"] = bool((df["monthly_charges"] >= 0).all())
        results["checks"]["target_binary"] = set(df["churn"].unique()).issubset({0, 1})
        churn_rate = df["churn"].mean()
        results["checks"]["class_balance"] = 0.05 < churn_rate < 0.95

    # Check minimum sample size
    results["checks"]["sufficient_samples"] = len(df) >= 100

    # Overall pass/fail
    results["passed"] = all(results["checks"].values())

    if not results["passed"]:
        failed = [k for k, v in results["checks"].items() if not v]
        raise ValueError(f"Data validation failed on checks: {failed}")

    logger.info("All data validation checks passed")
    return results


def ingest_data(config: dict) -> pd.DataFrame:
    """Full ingestion step: generate (or load) data, validate, and save.

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        Validated DataFrame.
    """
    raw_path = Path(config["data"]["raw_data_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists():
        logger.info("Loading existing data from %s", raw_path)
        df = pd.read_csv(raw_path)
    else:
        logger.info("Generating synthetic dataset")
        df = generate_synthetic_data(
            n_samples=config["data"]["n_samples"],
            random_state=config["data"]["random_state"],
        )
        df.to_csv(raw_path, index=False)
        logger.info("Saved raw data to %s", raw_path)

    validate_data(df)
    return df
