import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from textblob import TextBlob

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer for extracting textual features.
    Extracted features include:
    - Word count
    - Character count
    - Sentiment polarity (using TextBlob)
    """
    def fit(self, X, y=None):
        """
        Fit method. Does nothing as no fitting is required for this transformer.
        
        Parameters:
        - X: Input data.
        - y: Target labels.

        Returns:
        - self: The fitted transformer.
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform method to extract features from text.
        
        Parameters:
        - X: Input data

        Returns:
        - np.ndarray: An array of the extracted features.
        """
        return np.array([
            [
                len(text.split()),         # Count words
                len(text),                 # COunt characters
                TextBlob(text).sentiment.polarity  # Sentiment polarity
            ]
            for text in X
        ])
