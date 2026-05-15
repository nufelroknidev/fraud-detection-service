"""Central MLflow / DagsHub tracking initialisation.

Call init_mlflow() once at the top of any training or evaluation script.
Falls back to local file store when DagsHub credentials are not present
(e.g. CI environments).
"""

import os
import mlflow
from dotenv import load_dotenv

load_dotenv()


def init_mlflow(experiment_name: str) -> str:
    """Configure MLflow tracking and return the tracking URI.

    If DAGSHUB_USERNAME / DAGSHUB_TOKEN / DAGSHUB_REPO are all set,
    tracks to DagsHub. Otherwise falls back to a local mlruns/ directory
    so CI and offline development work without credentials.
    """
    username = os.environ.get("DAGSHUB_USERNAME")
    token    = os.environ.get("DAGSHUB_TOKEN")
    repo     = os.environ.get("DAGSHUB_REPO")

    if username and token and repo:
        tracking_uri = f"https://dagshub.com/{username}/{repo}.mlflow"
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        mlflow.set_tracking_uri(tracking_uri)
    else:
        tracking_uri = "/tmp/mlruns"
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)
    return tracking_uri
