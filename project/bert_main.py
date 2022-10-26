from bert_corpus_indexation import get_corpus_embeddings
from bert_query_indexation import get_query_embedding
from bert_similarity_calculator import get_closest_document_name

import os
import numpy as np
import logging
import torch
from transformers import AutoTokenizer, AutoModel


'''
Задача #1:
Реализуйте поиск, где:
    - метод векторизации текстов - Bert (модель sbert_large_nlu_ru)
    - формат хранения индекса - матрица Document-Term
    - метрика близости пар (запрос, документ) - косинус, он же dot на нормированных векторах
    - в качестве корпуса - датасет про любовь Ответы Мейл 

В реализации
    - функция индексации корпуса
    - функция индексации запроса
    - функция с реализацией подсчета близости запроса и документов корпуса
    - главная функция, объединяющая все это вместе
'''


logging.info('Загружаю модель BERT...')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru").to(device)


def main(corpus: list, query: str, n_results: int) -> None:
    # Get corpus and query embeddings using Sbert
    if not os.path.isfile('corpus_embeddings.npy'):
        corpus_embeddings = get_corpus_embeddings(corpus, model, tokenizer, device)
    else:
        corpus_embeddings = np.load('corpus_embeddings.npy')
    query_embedding = get_query_embedding(query, model, tokenizer, device)
    # Calculate distances between the corpus matrix and the query
    nearest_docs = get_closest_document_name(corpus, query_embedding, corpus_embeddings)
    results = nearest_docs[:n_results]
    return results


logging.getLogger().setLevel(logging.INFO)
