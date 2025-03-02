import os
import subprocess
import logging
from apscheduler.schedulers.background import BackgroundScheduler

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

def run_scraper():
    """Runs the web scraper to fetch new data."""
    logging.info("🕵️ Running the web scraper...")
    try:
        subprocess.run(["scrapy", "crawl", "yelp"], cwd="../src/data_collection/yelp_scraper/yelp_scraper", check=True)
        logging.info("✅ Scraping completed successfully.")
    except Exception as e:
        logging.error(f"❌ Scraping failed: {e}")

def run_notebook(notebook_path):
    """Executes all cells in a Jupyter notebook using nbconvert."""
    logging.info(f"📓 Running notebook: {notebook_path}")
    try:
        subprocess.run(
            [
                os.path.join("../.venv/bin/python"), 
                "-m", "jupyter", "nbconvert", "--to", "notebook",
                "--execute", notebook_path, "--output", notebook_path,
                "--ExecutePreprocessor.kernel_name=.venv"  # Ensure to use the correct kernel
            ],
            check=True
        )
        logging.info(f"✅ {notebook_path} executed successfully.")
    except Exception as e:
        logging.error(f"❌ Failed to execute {notebook_path}: {e}")


def run_pipeline():
    """Triggers the full prediction pipeline."""
    logging.info("🚀 Starting the prediction pipeline...")

    # 1. Run web scraper
    run_scraper()

    # 2. Run EDA notebook
    run_notebook(EDA_NOTEBOOK)

    # 3. Run Preprocessing notebook
    run_notebook(PREPROCESS_NOTEBOOK)

    # 4. Run Prediction notebook
    run_notebook(PREDICT_NOTEBOOK)

    logging.info("✅ Prediction pipeline completed successfully.")

if __name__ == "__main__":
    # Schedule the pipeline to run every hour
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_pipeline, "interval", minutes=4)
    scheduler.start()

    logging.info("🔄 Automation started. Running every hour.")

    try:
        while True:
            pass  # Keep the script running
    except KeyboardInterrupt:
        scheduler.shutdown()
        logging.info("🛑 Automation stopped.")
