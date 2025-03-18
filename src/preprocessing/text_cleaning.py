"""
Handles text cleaning operations using different strategies.
"""

import re
import spacy

# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")


# Define a base class for the cleaning strategies
class CleaningStrategy:
    def process(self, text: str) -> str:
        raise NotImplementedError("Subclasses must implement the process method.")


# Define concrete cleaning strategies
class LowercaseText(CleaningStrategy):
    def process(self, text: str) -> str:
        return text.lower()


class RemoveSpecialCharacters(CleaningStrategy):
    def process(self, text: str) -> str:
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)


class RemoveExtraWhitespace(CleaningStrategy):
    def process(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()


class PreprocessWithSpacy(CleaningStrategy):
    def process(self, text: str) -> str:
        doc = nlp(text)
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space
        ]
        return ' '.join(tokens)

