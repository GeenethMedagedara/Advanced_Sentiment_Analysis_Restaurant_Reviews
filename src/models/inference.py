"""
Handles the tracking uri for mlflow
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
import spacy

class BERTSentimentAnalysisStrategyPredict:
    """
    Hybrid Aspect-Based Sentiment Analysis (ABSA) using:
    - Fine-tuned BERT for sentiment analysis.
    - BERT Sentence Similarity (SBERT) for implicit aspect detection.
    - spaCy Word Embeddings for additional aspect similarity.
    """
    def __init__(self, model_path: str = "./saved_model", similarity_threshold: float = 0.4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Load Sentence-BERT for Aspect Similarity
        self.sbert_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        self.similarity_threshold = similarity_threshold  # Sentence similarity threshold

        # Load spaCy for Word Embeddings (lexical similarity)
        self.nlp = spacy.load("en_core_web_md")

    def is_aspect_present(self, aspect: str, sentence: str) -> bool:
        """
        Uses BERT Sentence Similarity and spaCy Word Embeddings to check if an aspect is present in a sentence.
        """
        # Sentence-BERT similarity
        aspect_embedding = self.sbert_model.encode(aspect, convert_to_tensor=True)
        sentence_embedding = self.sbert_model.encode(sentence, convert_to_tensor=True)
        sbert_similarity = util.pytorch_cos_sim(aspect_embedding, sentence_embedding).item()

        # spaCy Word Embedding similarity
        aspect_token = self.nlp(aspect)
        sentence_doc = self.nlp(sentence)
        spacy_similarity = max((aspect_token.similarity(token) for token in sentence_doc), default=0.0)

        # Combine both similarities (weighted)
        combined_similarity = (0.7 * sbert_similarity) + (0.3 * spacy_similarity)

        return combined_similarity > self.similarity_threshold  # Return True if similarity is above threshold

    def analyze_sentiment(self, aspect: str, sentence: str) -> dict:
        """
        Predict the sentiment of an aspect in a sentence using the fine-tuned BERT model.
        Only predicts if the aspect is present in the sentence.
        """
        # Check if aspect is present in the sentence
        if not self.is_aspect_present(aspect, sentence):
            return {"aspect": aspect, "sentiment": "not mentioned"}

        input_text = f"[ASPECT] {aspect} [SEP] {sentence}"

        encoding = self.tokenizer(
            input_text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()

        label_map = {0: "negative", 1: "neutral", 2: "positive", 3: "conflict"}
        return {"aspect": aspect, "sentiment": label_map[predicted_label]}

