import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def log_request(endpoint, data):
    """Log incoming requests"""
    logger.info(f"Endpoint: {endpoint} | Data: {data}")
