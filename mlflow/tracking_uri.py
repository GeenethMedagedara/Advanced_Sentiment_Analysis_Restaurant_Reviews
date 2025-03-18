"""
Handles the tracking uri for mlflow
"""

import mlflow

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def get_tracking_uri():
    return mlflow.get_tracking_uri()
