import os
from sklearn.feature_extraction.text import TfidfVectorizer


def get_corpus_index_matrix(corpus: list) -> tuple:
    texts = list(corpus.values())
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    vocabulary = vectorizer.get_feature_names_out()
    # matrix = X.toarray()
    return X, vectorizer
