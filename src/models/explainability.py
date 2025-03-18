"""
Handles explainability for the ABSA model.
"""

import torch
import shap
import lime.lime_text
import seaborn as sns
import matplotlib.pyplot as plt
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
        
        # Initialize placeholder for input text
        self.the_input_text = None 

    def _predict_proba(self, texts):
        """Helper function to get probability outputs for SHAP."""
        print(f"Received texts: {texts}")
            
        processed_texts = [self.the_input_text if "[MASK]" in text else text for text in texts]
        print(f"Processed texts: {processed_texts}")

        
        inputs = self.tokenizer(
            processed_texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        return torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

    def explain_shap(self, aspect: str, sentence: str):
        """Explains the BERT sentiment prediction using SHAP."""
        input_text = f"[ASPECT] {aspect} [SEP] {sentence}"
        
        self.the_input_text = input_text
        shap_values = self.shap_explainer([input_text])
        shap.text_plot(shap_values)

    def explain_lime(self, aspect: str, sentence: str):
        """Explains the BERT sentiment prediction using LIME."""
        input_text = f"[ASPECT] {aspect} [SEP] {sentence}"

        def predict_fn(texts):
            return self._predict_proba(texts)

        exp = self.lime_explainer.explain_instance(input_text, predict_fn, num_features=10)
        exp.show_in_notebook()

    def explain_sbert_similarity(self, aspect: str, sentence: str):
        """Visualizes SBERT cosine similarity between aspect and sentence."""
        aspect_embedding = self.sbert_model.encode(aspect, convert_to_tensor=True)
        sentence_embedding = self.sbert_model.encode(sentence, convert_to_tensor=True)

        similarity_score = util.pytorch_cos_sim(aspect_embedding, sentence_embedding).item()
        
        print(f"SBERT Similarity between '{aspect}' and sentence: {similarity_score:.4f}")

    def explain_spacy_similarity(self, aspect: str, sentence: str):
        """Visualizes spaCy similarity between aspect and each word in the sentence."""
        aspect_token = self.nlp(aspect)
        sentence_doc = self.nlp(sentence)

        similarities = {token.text: aspect_token.similarity(token) for token in sentence_doc}

        # Sort words by similarity
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        # Plot results
        words, scores = zip(*sorted_similarities)
        sns.barplot(x=scores, y=words, palette="coolwarm")
        plt.xlabel("Similarity Score")
        plt.title(f"spaCy Word Similarity to '{aspect}'")
        plt.show()