from hw4_1.corpus_indexation import get_corpus_embeddings

import os
import tqdm
import json
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_distances


def get_corpus_and_questions(path: str) -> list:
    corpus = []
    questions_and_answers = {}
    with open(path, 'r') as f:
        raw_data = list(f)[:10000]
    for item in tqdm(raw_data, desc='Создаю документы'):
        answers = json.loads(item)['answers']
        question = json.loads(item)['question']
        if answers:
            sorted_answers = sorted((d for d in answers if d['author_rating']['value'] != ''),
                                    key=lambda d: int(d['author_rating']['value']), reverse=True)
            corpus.append(sorted_answers[0]['text'])
            if question in corpus:
                questions_and_answers[question].append(sorted_answers[0]['text'])
            else:
                questions_and_answers[question] = [sorted_answers[0]['text']]
    return corpus, questions_and_answers


def main(path: str = '../input/thousands-of-questions-about-love/data.jsonl', n: int = 5) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru").to(device)

    # Create a corpus
    corpus, questions_and_answers = get_corpus_and_questions(path)
    queries = (list(questions_and_answers.keys()))[:10000]

    # Get corpus and query embeddings using Sbert
    corpus_embeddings = get_corpus_embeddings(corpus, model, tokenizer, device)
    queries_embeddings = get_corpus_embeddings(queries, model, tokenizer, device)

    # Calculate distances between the corpus matrix and the query
    distances = cosine_distances(queries_embeddings, corpus_embeddings)

    accuracy = []
    for i, column in tqdm(enumerate(distances.T)):
        indexes = np.argsort(column)[-n:]
        predicted_answers = np.take(list(questions_and_answers.values()), indexes)
        right_answers = questions_and_answers[queries[i]]
        if len(set(predicted_answers) & set(right_answers)) > 0:
            accuracy.append(1)
            print('match')
        else:
            accuracy.append(0)

    return np.mean(accuracy)
