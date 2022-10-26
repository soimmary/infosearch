from get_texts import get_texts
from tfidf_corpus_indexation import get_corpus_index_matrix
from tfidf_query_indexation import get_query_index_matrix
from tfidf_similarity_calculator import calculate_similarity, get_closest_document_name
from corpus_preprocessing import create_corpus
from scipy.sparse import save_npz, load_npz

import os
import logging
import pickle
import numpy as np

'''
Задача:
Реализуйте поиск, где
- в качестве метода векторизации документов корпуса - TF-IDF
- формат хранения индекса - матрица Document-Term
- метрика близости пар (запрос, документ) - косинусная близость
- в качестве корпуса - корпус Друзей из первого задания

Что должно быть в реализации:
- функция индексации корпуса, на выходе которой посчитанная матрица Document-Term
- функция индексации запроса, на выходе которой посчитанный вектор запроса
- функция с реализацией подсчета близости запроса и документов корпуса, на выходе которой вектор,
  i-й элемент которого обозначает близость запроса с i-м документом корпуса
- главная функция, объединяющая все это вместе; на входе - запрос, на выходе - отсортированные
  по убыванию имена документов коллекции
'''

logging.info('Загружаю документы для вычисления TF-IDF...')
if not os.path.isfile('tfidf_corpus_matrix.npy'):
    corpus = get_texts('data.jsonl')
    original_texts, corpus = create_corpus(corpus)
    print('Создаю матрицу...')
    corpus_matrix, tfidf_vectorizer = get_corpus_index_matrix(corpus)

    with open('tfidf_corpus_matrix.npy', 'wb') as f:
        save_npz(f, corpus_matrix)
    with open('original_texts', 'wb') as f2:
        pickle.dump(original_texts, f2)
    with open('tfidf_vectorizer', 'wb') as f3:
        pickle.dump(tfidf_vectorizer, f3)

else:
    with open('original_texts', 'rb') as f1:
        original_texts = pickle.load(f1)
    with open('tfidf_corpus_matrix.npy', 'rb') as f2:
        corpus_matrix = load_npz(f2)
    with open('tfidf_vectorizer', 'rb') as f3:
        tfidf_vectorizer = pickle.load(f3)


def main(query: str, n_results: int) -> None:
    logging.info('Рассчитываю вектор для запроса...')
    query_matrix = get_query_index_matrix(query, tfidf_vectorizer)
    logging.info('Вычисляю расстояния...')
    distances = calculate_similarity(query_matrix, corpus_matrix)
    logging.info('Ранжирую ближайшие документы...')
    nearest_docs = get_closest_document_name(distances, list(original_texts.keys()))
    results = [original_texts[doc] for doc in nearest_docs[:n_results]]
    return results


logging.getLogger().setLevel(logging.INFO)
