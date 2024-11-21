# tokenizer.py

import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Ensure required NLTK resources are downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Cache stop words globally
STOP_WORDS = set(stopwords.words('english'))

def tokenize(text):
    """
    Tokenizes, lemmatizes and cleans a given text.

    Args:
        text (str): The text to process.

    Returns:
        list: A list of cleaned tokens.
    """

    # Replace Urls (if there are any)
    loc_url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    loc_detected_urls = re.findall(loc_url_regex, text)
    for url in loc_detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Preprocess text to exclude specific punctuation
    text = re.sub(r'[,\-\(\)]', ' ', text)

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    try:
        # Tokenize the input text
        tokens = word_tokenize(text)

        # Process tokens: remove stop words, lemmatize and clean
        clean_tokens = [
            lemmatizer.lemmatize(token).lower().strip()
            for token in tokens
            if token.lower() not in STOP_WORDS
        ]

        return clean_tokens

    except Exception as e:
        print(f"Error during tokenization: {e}")
        return []
