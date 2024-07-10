import re
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer

# Download NLTK stopwords (if not already downloaded)
import nltk
nltk.download('stopwords')

# Fetch Arabic stopwords from NLTK
arabic_stopwords = set(stopwords.words('arabic'))
stemmer = ISRIStemmer()

def preprocess_text(text):
    """
    Preprocess the input text by removing digits, English words, punctuation, and stopwords, and applying stemming.

    Args:
        text (str): Input text.

    Returns:
        str: Preprocessed text.
    """
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[a-zA-Z]+', '', text)  # Remove English words
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Remove punctuation and non-Arabic characters
    tokens = text.split()  # Tokenize
    tokens = [word for word in tokens if word not in arabic_stopwords]  # Remove stopwords

    tokens = [stemmer.stem(word) for word in tokens]  # Apply stemming
    return ' '.join(tokens)
