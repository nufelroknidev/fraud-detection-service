"""Quick smoke-test: logs one dummy run to DagsHub to confirm auth + tracking work."""

import mlflow
from src.tracking import init_mlflow

tracking_uri = init_mlflow("dagshub-connection-test")
print(f"Tracking URI: {tracking_uri}")

with mlflow.start_run(run_name="smoke-test"):
    mlflow.log_param("check", "connection")
    mlflow.log_metric("dummy_metric", 1.0)
    run_id = mlflow.active_run().info.run_id

print(f"Success — run ID: {run_id}")
print(f"View at: {tracking_uri}/#/experiments")
