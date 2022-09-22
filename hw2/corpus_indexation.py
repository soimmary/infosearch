import os

from directories import get_directories
from sklearn.feature_extraction.text import TfidfVectorizer


def create_corpus(main_dir: str) -> dict:
    corpus = dict()
    dirs = get_directories(main_dir)
    for dir in dirs:
        with open(dir, 'r', encoding='utf-8') as f:
            text = f.read()
            corpus[os.path.basename(dir)] = ' '.join(text.split())
    return corpus


def get_corpus_index_matrix(corpus: dict) -> tuple:
    episodes, texts = list(corpus.keys()), corpus.values()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    vocabulary = vectorizer.get_feature_names_out()
    matrix = X.toarray()
    return vectorizer, episodes, vocabulary, matrix
