import os

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
ARTIFACT_STORE = os.path.abspath("artifacts")

os.makedirs(ARTIFACT_STORE, exist_ok=True)

# Command to start the MLflow server
os.system(f"mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root {ARTIFACT_STORE} --host 0.0.0.0 --port 5000")
