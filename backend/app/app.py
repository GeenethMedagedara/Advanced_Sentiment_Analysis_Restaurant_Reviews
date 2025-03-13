import os
import logging
from flask import Flask, jsonify, request
from flask_cors import CORS
from src.routes.sentiment_routes import sentiment_bp
from src.routes.explainability_routes import explain_bp
from config import config
import mlflow.transformers
import atexit
import shutil
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs"))
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "api.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

app = Flask(__name__)
app.config["DEBUG"] = config.DEBUG

CORS(app, supports_credentials=True, resources={
    r"/*": {
        "origins": [
            "http://host.docker.internal:8080",
            "http://localhost:8080",
            "http://backend:4000"  
        ],
        "allow_headers": ["Content-Type", "Authorization"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    }
})

MODEL_CACHE_DIR = "./mlflow_cache/"
        
def cleanup():
    """Removes the MLflow cached model on shutdown."""
    logging.info("🧹 Cleaning up cached MLflow model...")
    try:
        if os.path.exists(MODEL_CACHE_DIR):
            shutil.rmtree(MODEL_CACHE_DIR)
            logging.info("Cached model removed successfully.")
        else:
            logging.info("No cached model found.")
    except Exception as e:
        logging.error(f"Failed to remove cached model: {e}")
        


# Register Blueprints
app.register_blueprint(sentiment_bp, url_prefix="/api/sentiment")
app.register_blueprint(explain_bp, url_prefix="/api/explain")

# Log each request
@app.before_request
def log_request():
    logging.info(f"Incoming request: {request.method} {request.path}")

# Health Check Endpoint
@app.route("/", methods=["GET"])
def health_check():
    logging.info("Health check endpoint accessed.")
    return jsonify({"message": "Server is running!"}), 200

# Handle errors and log them
@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Error: {str(e)}")
    return jsonify({"error": "Internal Server Error"}), 500

# Ensure cleanup when Flask shuts down
atexit.register(cleanup)

if __name__ == "__main__":
    logging.info("Starting API server...")
    app.run(host="0.0.0.0", port=4000)

