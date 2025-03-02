import os
import torch
from dotenv import load_dotenv

load_dotenv()

class Config:
    MODEL_PATH = os.getenv("MODEL_PATH", "./saved_model")
    DEVICE = "cuda" if os.getenv("USE_CUDA", "false").lower() == "true" and torch.cuda.is_available() else "cpu"
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

config = Config()