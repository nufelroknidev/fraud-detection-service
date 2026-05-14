"""Central MLflow / DagsHub tracking initialisation.

Call init_mlflow() once at the top of any training or evaluation script.
"""

import os
import mlflow
from dotenv import load_dotenv

load_dotenv()


def init_mlflow(experiment_name: str) -> str:
    """Configure MLflow to track to DagsHub and return the experiment ID."""
    username = os.environ["DAGSHUB_USERNAME"]
    token = os.environ["DAGSHUB_TOKEN"]
    repo = os.environ["DAGSHUB_REPO"]

    tracking_uri = f"https://dagshub.com/{username}/{repo}.mlflow"

    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return tracking_uri
