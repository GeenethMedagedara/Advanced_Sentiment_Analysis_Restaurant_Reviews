"""
Handles the automation of the prediction pipeline.
"""

import os
import subprocess
import logging
import pandas as pd
import time
from apscheduler.schedulers.background import BackgroundScheduler
import sys
import shutil
import mlflow.transformers
import threading
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from config.db_config import get_db

# Setup logging
LOG_DIR = os.path.abspath("../logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "predict_pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Paths
PROJECT_DIR = os.path.abspath("../")
NOTEBOOKS_DIR = os.path.join(PROJECT_DIR, "notebooks")
EDA_NOTEBOOK = os.path.join(NOTEBOOKS_DIR, "EDA.ipynb")
PREPROCESS_NOTEBOOK = os.path.join(NOTEBOOKS_DIR, "preprocessing.ipynb")
PREDICT_NOTEBOOK = os.path.join(NOTEBOOKS_DIR, "predict.ipynb")
PREDICTIONS_CSV = os.path.join(PROJECT_DIR, "data/visualization/cleaned_and_preprocessed_absa_sentimented_reviews.csv")

MODEL_CACHE_DIR = os.path.abspath("../mlflow_cache/") 

# Global flag to track model loading
MODEL_READY = False

def load_model():
    """Loads the MLflow model asynchronously into the cache directory at startup."""
    global MODEL_READY
    logging.info("📥 ------------------------Loading MLflow model into cache (this may take a while)-----------------------")

    try:
        mlflow.set_tracking_uri("http://host.docker.internal:5000")
        
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("ABSA Sentiment Analysis")  # Change this to your experiment name
        if experiment is None:
            raise ValueError("Experiment not found. Please check your experiment name.")
        
        # Get latest run
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        print(runs)
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
            shutil.rmtree(MODEL_CACHE_DIR) 
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

        model.save_pretrained(MODEL_CACHE_DIR)
        tokenizer.save_pretrained(MODEL_CACHE_DIR)
        logging.info(f"Model loaded and cached at {MODEL_CACHE_DIR}.")
        
        MODEL_READY = True  
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        MODEL_READY = False  

def run_scraper():
    """Runs the web scraper to fetch new data."""
    logging.info("🕵️ Running the web scraper...")
    try:
        subprocess.run(["/app/.venv/bin/scrapy", "crawl", "yelp"], cwd="../src/data_collection/yelp_scraper/yelp_scraper", check=True)
        logging.info("Scraping completed successfully.")
    except Exception as e:
        logging.error(f"Scraping failed: {e}")

def run_notebook(notebook_path):
    """Executes all cells in a Jupyter notebook using nbconvert."""
    logging.info(f"📓 Running notebook: {notebook_path}")
    try:
        subprocess.run(
            [
                "/app/.venv/bin/python",  
                "-m", "jupyter", "nbconvert", "--to", "notebook",
                "--execute", notebook_path, "--output", notebook_path,
                "--ExecutePreprocessor.kernel_name=python3"  
            ],
            check=True,
            capture_output=True,
            text=True
        )
        logging.info(f"{notebook_path} executed successfully.")
    except Exception as e:
        logging.error(f"Failed to execute {notebook_path}: {e}")


def save_predictions_to_mongo():
    """Load predictions from CSV and save them to MongoDB."""
    if not os.path.exists(PREDICTIONS_CSV):
        logging.error(f"Predictions file not found: {PREDICTIONS_CSV}")
        return
    
    try:
        predictions_df = pd.read_csv(PREDICTIONS_CSV)
        db = get_db()
        collection = db["predictions"]  # Collection name in MongoDB
        collection.insert_many(predictions_df.to_dict(orient="records"))
        logging.info("Predictions saved to MongoDB")
    except Exception as e:
        logging.error(f"Failed to save predictions to MongoDB: {e}")


def cleanup():
    """Removes the MLflow cached model on shutdown."""
    logging.info("Cleaning up cached MLflow model...")
    try:
        if os.path.exists(MODEL_CACHE_DIR):
            shutil.rmtree(MODEL_CACHE_DIR)
            logging.info("Cached model removed successfully.")
        else:
            logging.info("No cached model found.")
    except Exception as e:
        logging.error(f"Failed to remove cached model: {e}")


def run_pipeline():
    """Triggers the full prediction pipeline."""
    if not MODEL_READY:
        logging.warning("Model is still loading. Skipping this run to avoid interference.")
        return  # Skip pipeline execution if model isn't ready

    logging.info("Starting the prediction pipeline...")

    # 1. Run web scraper
    run_scraper()

    # 2. Run EDA notebook
    run_notebook(EDA_NOTEBOOK)

    # 3. Run Preprocessing notebook
    run_notebook(PREPROCESS_NOTEBOOK)

    # 4. Run Prediction notebook
    run_notebook(PREDICT_NOTEBOOK)
    
    # 5. Save predictions to MongoDB
    save_predictions_to_mongo()

    logging.info("Prediction pipeline completed successfully.")


if __name__ == "__main__":
    # Start model loading in a separate thread
    model_loader_thread = threading.Thread(target=load_model)
    model_loader_thread.start()
    
    # Schedule the pipeline to run every hour
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_pipeline, "interval", hour=1)
    scheduler.start()

    logging.info("Automation started. Running every hour.")

    try:
        while True:
            # pass  # Keep the script running
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Automation stopping...")
        scheduler.shutdown()
        cleanup()  # Remove model cache
        logging.info("Automation stopped.")