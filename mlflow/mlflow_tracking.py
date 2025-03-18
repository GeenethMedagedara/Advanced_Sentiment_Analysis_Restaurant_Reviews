"""
Handles MLflow tracking.
"""

import mlflow
import os

# Set local MLflow tracking URI
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Ensure MLflow Experiment Exists
EXPERIMENT_NAME = "ABSA Sentiment Analysis"
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    mlflow.create_experiment(EXPERIMENT_NAME)

mlflow.set_experiment(EXPERIMENT_NAME)

def log_metrics(metrics: dict):
    """Logs evaluation metrics to MLflow."""
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

def log_params(params: dict):
    """Logs hyperparameters to MLflow."""
    for key, value in params.items():
        mlflow.log_param(key, value)

def log_model(model, model_name: str = "ABSA_model"):
    """Logs a trained model to MLflow."""
    mlflow.pytorch.log_model(model, model_name)
