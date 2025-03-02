import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

import torch
import shap
import lime.lime_text
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import io
from transformers import BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import spacy

class ABSAExplainability:
    def __init__(self, model_path: str = "./saved_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Load Sentence-BERT for similarity analysis
        self.sbert_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

        # Load spaCy for word embeddings
        self.nlp = spacy.load("en_core_web_md")

        # Initialize SHAP explainer
        self.shap_explainer = shap.Explainer(self._predict_proba, self.tokenizer)

        # Initialize LIME explainer
        self.lime_explainer = lime.lime_text.LimeTextExplainer(class_names=["negative", "neutral", "positive", "conflict"])
        
        # Placeholder for input text
        self.the_input_text = None 

    def _predict_proba(self, texts):
        """Helper function to get probability outputs for SHAP & LIME."""
        processed_texts = [self.the_input_text if "[MASK]" in text else text for text in texts]
        
        inputs = self.tokenizer(
            processed_texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        return torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

    def _save_plot_to_base64(self):
        """Converts the current matplotlib plot to a base64 string."""
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        return image_base64

    def explain_shap(self, aspect: str, sentence: str):
        """Generates SHAP explanation and returns base64 image."""
        input_text = f"[ASPECT] {aspect} [SEP] {sentence}"
        self.the_input_text = input_text
        
        shap_values = self.shap_explainer([input_text])

        plt.figure(figsize=(8, 4))
        shap.plots.force(self.shap_explainer.expected_value[0], shap_values[0])
        
        return self._save_plot_to_base64()

    def explain_lime(self, aspect: str, sentence: str):
        """Generates LIME explanation and returns base64 image."""
        input_text = f"[ASPECT] {aspect} [SEP] {sentence}"

        def predict_fn(texts):
            return self._predict_proba(texts)

        exp = self.lime_explainer.explain_instance(input_text, predict_fn, num_features=10)
        
        # Save LIME explanation as base64 image
        plt.figure()
        exp.as_pyplot_figure()
        return self._save_plot_to_base64()

    def explain_sbert_similarity(self, aspect: str, sentence: str):
        """Visualizes SBERT cosine similarity and returns base64 image."""
        aspect_embedding = self.sbert_model.encode(aspect, convert_to_tensor=True)
        sentence_embedding = self.sbert_model.encode(sentence, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(aspect_embedding, sentence_embedding).item()
        
        # Create visualization
        plt.figure()
        sns.heatmap([[similarity_score]], annot=True, cmap="coolwarm", xticklabels=[aspect], yticklabels=["Sentence"])
        plt.title("SBERT Cosine Similarity")
        
        return self._save_plot_to_base64()

    def explain_spacy_similarity(self, aspect: str, sentence: str):
        """Visualizes spaCy word similarity and returns base64 image."""
        aspect_token = self.nlp(aspect)
        sentence_doc = self.nlp(sentence)

        similarities = {token.text: aspect_token.similarity(token) for token in sentence_doc}
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        # Create plot
        plt.figure(figsize=(8, 4))
        words, scores = zip(*sorted_similarities)
        sns.barplot(x=scores, y=words, palette="coolwarm")
        plt.xlabel("Similarity Score")
        plt.title(f"spaCy Word Similarity to '{aspect}'")

        return self._save_plot_to_base64()