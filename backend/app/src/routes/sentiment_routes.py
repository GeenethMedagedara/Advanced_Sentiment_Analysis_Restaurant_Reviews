"""
Handles requests for sentiment analysis of reviews.
"""

from flask import Blueprint, request, jsonify
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
import spacy
from src.models.model_loader import tokenizer, model

# Load models and configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Sentence-BERT for aspect similarity
sbert_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Load spaCy for lexical similarity
nlp = spacy.load("en_core_web_md")

# Set similarity threshold for aspect presence
SIMILARITY_THRESHOLD = 0.4

sentiment_bp = Blueprint("sentiment", __name__)

def is_aspect_present(aspect: str, sentence: str) -> bool:
    """
    Uses BERT Sentence Similarity and spaCy Word Embeddings to check if an aspect is present in a sentence.
    """
    # Sentence-BERT similarity
    aspect_embedding = sbert_model.encode(aspect, convert_to_tensor=True)
    sentence_embedding = sbert_model.encode(sentence, convert_to_tensor=True)
    sbert_similarity = util.pytorch_cos_sim(aspect_embedding, sentence_embedding).item()

    # spaCy Word Embedding similarity
    aspect_token = nlp(aspect)
    sentence_doc = nlp(sentence)
    spacy_similarity = max((aspect_token.similarity(token) for token in sentence_doc), default=0.0)

    # Combine both similarities (weighted)
    combined_similarity = (0.7 * sbert_similarity) + (0.3 * spacy_similarity)

    return combined_similarity > SIMILARITY_THRESHOLD  # Return True if similarity is above threshold

def analyze_sentiment(aspect: str, sentence: str) -> dict:
    """
    Predict the sentiment of an aspect in a sentence using the fine-tuned BERT model.
    Only predicts if the aspect is present in the sentence.
    """
    # Check if aspect is present in the sentence
    if not is_aspect_present(aspect, sentence):
        return {"aspect": aspect, "sentiment": "not mentioned"}

    input_text = f"[ASPECT] {aspect} [SEP] {sentence}"

    encoding = tokenizer(
        input_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()

    label_map = {0: "negative", 1: "neutral", 2: "positive", 3: "conflict"}
    return {"aspect": aspect, "sentiment": label_map[predicted_label]}

@sentiment_bp.route("/predict", methods=["POST"])
def predict():
    """Predict sentiment for a given review and aspect."""
    try:
        data = request.get_json()
        aspect = data.get("aspect", "").strip()
        review = data.get("review", "").strip()

        # Validate input
        if not aspect or not review:
            return jsonify({"error": "Both 'aspect' and 'review' are required fields"}), 400

        # Perform sentiment analysis
        result = analyze_sentiment(aspect, review)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
