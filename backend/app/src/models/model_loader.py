from transformers import BertTokenizer, BertForSequenceClassification
from src.models.explainability_api import ABSAExplainability
from config import config

def load_model():
    """Load BERT model & tokenizer"""
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(config.MODEL_PATH).to(config.DEVICE)
    model.eval()
    return tokenizer, model

# Initialize globally
tokenizer, model = load_model()
explainer = ABSAExplainability(config.MODEL_PATH)