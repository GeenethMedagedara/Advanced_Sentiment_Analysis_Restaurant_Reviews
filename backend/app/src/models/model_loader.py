"""
Handles loading the BERT model and tokenizer, and instantiates the ABSAExplainability class.
"""

import os
import logging
from transformers import BertTokenizer
from src.models.explainability_api import ABSAExplainability
from config import config
import shutil
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import mlflow.transformers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MODEL_CACHE_DIR = "./mlflow_cache/"

def load_model():
    """Load BERT model & tokenizer, importing model only when needed."""
    
    logging.info("📥Loading MLflow model into cache")

    try:
        mlflow.set_tracking_uri("http://host.docker.internal:5000")
        
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("ABSA Sentiment Analysis")  
        if experiment is None:
            raise ValueError("Experiment not found. Please check your experiment name.")
        
        # Get latest run
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        if not runs:
            raise ValueError("No runs found in the experiment.")
        
        latest_run_id = runs[0].info.run_id
        model_uri = f"mlflow-artifacts:/{experiment.experiment_id}/{latest_run_id}/artifacts/ABSA_model/model"
        
        # Load from MLflow
        model_path = mlflow.artifacts.download_artifacts(model_uri)

        # Load Hugging Face model
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        if os.path.exists(MODEL_CACHE_DIR):
            shutil.rmtree(MODEL_CACHE_DIR)  # Ensure fresh download
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

        model.save_pretrained(MODEL_CACHE_DIR)
        tokenizer.save_pretrained(MODEL_CACHE_DIR)

        logging.info(f"Model loaded and cached at {MODEL_CACHE_DIR}.")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
    
    if not os.path.exists(config.MODEL_PATH):
        logger.error(f"Model path '{config.MODEL_PATH}' does not exist. Flask app will start, but model is unavailable.")
        return None, None

    try:
        # Import model dynamically after confirming the model exists
        from transformers import BertForSequenceClassification

        # Load model
        model = BertForSequenceClassification.from_pretrained(config.MODEL_PATH).to(config.DEVICE)
        model.eval()

        # Load tokenizer
        tokenizer_path = os.path.join(config.MODEL_PATH, "tokenizer")
        if os.path.exists(tokenizer_path):
            tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        else:
            logger.warning(f"Tokenizer not found at '{tokenizer_path}', using default 'bert-base-uncased'.")
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            tokenizer.save_pretrained(tokenizer_path)  # Save locally

        return tokenizer, model

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None

# Try loading the model; delay imports until model is ready
tokenizer, model = load_model()

if model:
    explainer = ABSAExplainability(config.MODEL_PATH)
else:
    explainer = None


