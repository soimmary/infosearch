from get_texts import get_texts
from preprocess_corpus import create_corpus
from bm25_calculator import calculate_bm25
from query_indexation import get_query_index_matrix
from similarity_calculator import calculate_distance, get_closest_document_name

import numpy as np
import os.path
import typer
import pickle


def main(query: str, path: str = 'data.jsonl') -> None:
    if not os.path.isfile('corpus_matrix.npy'):
        texts = get_texts(path)
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
            corpus_matrix = np.load(f2)
        with open('count_vectorizer', 'rb') as f3:
            count_vectorizer = pickle.load(f3)

    query_matrix = get_query_index_matrix(query, count_vectorizer)
    print('Вычисляю расстояния...')
    distances = calculate_distance(query_matrix, corpus_matrix)
    print('Ранжирую ближайшие документы...')
    nearest_docs = get_closest_document_name(distances, list(original_texts.keys()))
    print('Готово!', end='\n\n\n')
    for doc in nearest_docs:
        print(original_texts[doc[0]])


if __name__ == '__main__':
    typer.run(main)
