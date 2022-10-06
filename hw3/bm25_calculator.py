import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def calculate_bm25(corpus: dict, k: int = 2, b: int = 0.75) -> np.array:
    texts = (corpus.values())
    # tf
    count_vectorizer = CountVectorizer()
    count = count_vectorizer.fit_transform(texts).toarray()

    tf = count

    # idf
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')
    tfidf = tfidf_vectorizer.fit_transform(texts).toarray()
    vocabulary = tfidf_vectorizer.get_feature_names_out()
    idf = tfidf_vectorizer.idf_
    idf = np.expand_dims(idf, axis=0)

    # расчет количества слов в каждом документе - l(d)
    len_d = tf.sum(axis=1)

    # расчет среднего количества слов документов корпуса - avdl
    avdl = len_d.mean()

    # расчет числителя
    A = idf * tf * (k + 1)

    # расчет знаменателя
    B_1 = (k * (1 - b + b * len_d / avdl))
    B_1 = np.expand_dims(B_1, axis=-1)
    B = tf + B_1

    # BM25
    matrix = A / B
    return matrix, count_vectorizer
