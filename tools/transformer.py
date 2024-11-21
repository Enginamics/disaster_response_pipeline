import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from textblob import TextBlob

# Custom Transformer for Textual Features
class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Features: word count and sentiment polarity
        return np.array([
            [
                len(text.split()),  # Word count
                len(text),  # Character count
                TextBlob(text).sentiment.polarity,  # Sentiment polarity
            ]
            for text in X
        ])