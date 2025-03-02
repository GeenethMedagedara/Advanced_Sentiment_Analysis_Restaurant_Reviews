import os
import subprocess
import logging
from apscheduler.schedulers.background import BackgroundScheduler

# Setup logging
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../logs"))
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "scheduler.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

SCRAPY_PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../yelp_scraper/yelp_scraper"))

def run_scraper():
    logging.info("Starting Scrapy spider...")
    try:
        subprocess.run(["scrapy", "crawl", "yelp"], cwd=SCRAPY_PROJECT_PATH, check=True, shell=True)
        logging.info("Scraper finished successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Scraper failed: {e}")

# Set up scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(run_scraper, "interval", minutes=1)  # Runs every 1 minute

if __name__ == "__main__":
    logging.info("Scheduler is starting...")
    scheduler.start()

    try:
        while True:
            pass  # Keep script running
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logging.info("Scheduler stopped.")

