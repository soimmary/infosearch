from get_texts import get_texts
from corpus_preprocessing import create_corpus
from bm25_calculator import calculate_bm25
from bm25_query_indexation import get_query_index_matrix
from bm25_similarity_calculator import calculate_distance, get_closest_document_name

import numpy as np
import os.path
import logging
import pickle
from scipy import sparse


logging.info('Загружаю документы для вычисления BM25...')
if not os.path.isfile('corpus_matrix.npy'):
    texts = get_texts('data.jsonl')
    original_texts, corpus = create_corpus(texts)
    print('Создаю матрицу...')
    corpus_matrix, count_vectorizer = calculate_bm25(corpus)

    with open('corpus_matrix.npy', 'wb') as f:
        np.save(f, corpus_matrix.astype(np.half))
    with open('original_texts', 'wb') as f2:
        pickle.dump(original_texts, f2)
    with open('count_vectorizer', 'wb') as f3:
        pickle.dump(count_vectorizer, f3)

else:
    with open('original_texts', 'rb') as f1:
        original_texts = pickle.load(f1)
    with open('corpus_matrix.npy', 'rb') as f2:
        corpus_matrix = sparse.csr_matrix(np.load(f2))
    with open('count_vectorizer', 'rb') as f3:
        count_vectorizer = pickle.load(f3)


def main(query: str, n_results: int) -> None:
    query_matrix = get_query_index_matrix(query, count_vectorizer)
    logging.info('Вычисляю расстояния...')
    distances = calculate_distance(query_matrix, corpus_matrix)
    logging.info('Ранжирую ближайшие документы...')
    nearest_docs = get_closest_document_name(distances, list(original_texts.keys()))
    results = [original_texts[doc] for doc in nearest_docs[-n_results:]]
    return reversed(results)


logging.getLogger().setLevel(logging.INFO)
