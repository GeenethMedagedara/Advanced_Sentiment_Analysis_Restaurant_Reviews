"""
Handles the topic modeling of reviews using different strategies.
"""

from abc import ABC, abstractmethod
from bertopic import BERTopic
from umap import UMAP
import pandas as pd
from typing import List

# Define Strategy Interface
class TopicModelingStrategy(ABC):
    @abstractmethod
    def extract_topics(self, reviews: List[str]) -> List[int]:
        pass
    
    @abstractmethod
    def visualize_topics_data(self):
        pass


class BERTopicModelingStrategy(TopicModelingStrategy):
    def __init__(self, language: str = "english"):
        self.umap_model = UMAP(n_neighbors=5, n_components=3, metric="cosine", random_state=42)
        self.topic_model = BERTopic(language=language, umap_model=self.umap_model)
        self.probabilities = None  # Store probabilities for visualization

    def extract_topics(self, reviews: List[str]) -> List[int]:
        topics, probabilities = self.topic_model.fit_transform(reviews)
        self.probabilities = probabilities  # Save probabilities
        return topics

    def visualize_topics_data(self):
        
        topic_info = self.topic_model.get_topic_info()
        print(topic_info)  # Print topic details instead of calling `.show()`
        
        # Topic Overview
        fig1 = self.topic_model.visualize_barchart()
        fig1.show()

        # Topic Hierarchy
        fig2 = self.topic_model.visualize_hierarchy()
        fig2.show()

        # Term Rankings
        fig3 = self.topic_model.visualize_term_rank()
        fig3.show()

        # Document-Topic Distribution (Fix: Pass stored probabilities)
        if self.probabilities is not None:
            fig4 = self.topic_model.visualize_distribution(self.probabilities)
            fig4.show()
        else:
            print("No probabilities available. Ensure `extract_topics()` has been called.")

        print("Visualizations generated successfully.")

# Context Class
class TopicModelingContext:
    def __init__(self, strategy: TopicModelingStrategy):
        self.strategy = strategy

    def analyze_reviews(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """
        Analyze reviews and assign topics.
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")
        
        df = df.dropna(subset=[text_column])  # Remove NaN values
        df[text_column] = df[text_column].astype(str).str.strip()  # Ensure text format
        
        df["Topic"] = self.strategy.extract_topics(df[text_column].tolist())
        return df
