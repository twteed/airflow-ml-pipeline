from setuptools import setup, find_packages

setup(
    name="ml_pipeline",
    version="1.0.0",
    description="End-to-end ML pipeline orchestrated with Apache Airflow",
    author="Your Name",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "scikit-learn>=1.5",
        "pandas>=2.2",
        "numpy>=1.26",
        "xgboost>=2.1",
        "mlflow>=2.15",
        "joblib>=1.4",
        "pyyaml>=6.0",
        "python-dotenv>=1.0",
    ],
)
