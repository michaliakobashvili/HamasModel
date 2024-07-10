from sklearn.feature_extraction.text import TfidfVectorizer

def compute_tfidf_matrix(texts):
    """
    Compute the TF-IDF matrix for the given texts.

    Args:
        texts (list): List of preprocessed texts.

    Returns:
        tuple: TF-IDF matrix and the vectorizer.
    """
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, max_df=0.95, min_df=0.05, ngram_range=(1, 3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    return tfidf_matrix, tfidf_vectorizer
