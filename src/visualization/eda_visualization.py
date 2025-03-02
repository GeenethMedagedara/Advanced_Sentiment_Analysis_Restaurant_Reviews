import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter

class EdaVisualizationStrategy:
    def execute(self, df):
        pass

class TextLengthDistribution(EdaVisualizationStrategy):
    def execute(self, df):
        """
        Display a graph of the distribution of the length of the review
        """
        df['review_length'] = df['text'].apply(len)
        sns.histplot(df['review_length'], kde=True, bins=30, color='blue')
        plt.title('Review Length Distribution')
        plt.show()

class WordCountDistribution(EdaVisualizationStrategy):
    def execute(self, df):
        """
        Display a graph of the distribution of the count of words
        """
        df['word_count'] = df['text'].apply(lambda x: len(x.split()))
        sns.histplot(df['word_count'], kde=True, bins=30, color='green')
        plt.title('Word Count Distribution')
        plt.show()
        
class CommonWordsDistribution(EdaVisualizationStrategy):
    def execute(self, df):
        """
        Display a graph of the most common words in the dataset
        """
        word_counts = Counter(" ".join(df['text']).split())
        common_words = pd.DataFrame(word_counts.most_common(20), columns=['Word', 'Frequency'])

        sns.barplot(data=common_words, x='Frequency', y='Word', palette='coolwarm')
        plt.title('Top 20 Common Words')
        plt.show()