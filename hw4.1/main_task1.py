from hw4.corpus_indexation import get_texts, get_corpus_embeddings
from hw4.query_indexation import get_query_embedding
from hw4.similarity_calculator import get_closest_document_name

import os
import time
import typer
import numpy as np

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


def main(query: str, path: str = '/Users/mariabocharova/PycharmProjects/infosearch/hw3/data.jsonl') -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru").to(device)

    # Create a corpus
    corpus = get_texts(path)
    # Get corpus and query embeddings using Sbert
    if not os.path.isfile('corpus_embeddings.npy'):
        corpus_embeddings = get_corpus_embeddings(corpus, model, tokenizer, device)
    else:
        corpus_embeddings = np.load('corpus_embeddings.npy')
    query_embedding = get_query_embedding(query, model, tokenizer, device)
    # Calculate distances between the corpus matrix and the query
    nearest_docs = get_closest_document_name(corpus, query_embedding, corpus_embeddings)
    for doc in nearest_docs[:10]:
        print(doc)


if __name__ == '__main__':
    start_time = time.time()
    # typer.run(main)
    main('любовь моя любимая')
    print("\nRuntime:", (time.time() - start_time))
