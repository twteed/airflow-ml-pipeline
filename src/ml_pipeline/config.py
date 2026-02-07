"""Configuration management for the ML pipeline."""

import os
from pathlib import Path

import yaml


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str | None = None) -> dict:
    """Load pipeline configuration from YAML file.

    Args:
        config_path: Path to config file. Defaults to config/pipeline_config.yaml.

    Returns:
        Configuration dictionary.
    """
    if config_path is None:
        config_path = os.environ.get(
            "PIPELINE_CONFIG_PATH",
            str(get_project_root() / "config" / "pipeline_config.yaml"),
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Apply environment variable overrides
    config["mlflow"]["tracking_uri"] = os.environ.get(
        "MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"]
    )
    config["mlflow"]["experiment_name"] = os.environ.get(
        "MLFLOW_EXPERIMENT_NAME", config["mlflow"]["experiment_name"]
    )

    data_dir = os.environ.get("PIPELINE_DATA_DIR")
    if data_dir:
        config["data"]["raw_data_path"] = os.path.join(data_dir, "raw_customers.csv")
        config["data"]["processed_data_path"] = os.path.join(
            data_dir, "processed_features.csv"
        )

    models_dir = os.environ.get("PIPELINE_MODELS_DIR")
    if models_dir:
        config["deployment"]["model_registry_path"] = os.path.join(
            models_dir, "registry"
        )
        config["deployment"]["champion_model_path"] = os.path.join(
            models_dir, "champion"
        )

    return config
