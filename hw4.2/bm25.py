from hw3.get_texts import get_texts
from hw3.preprocess_corpus import create_corpus
from hw3.bm25_calculator import calculate_bm25
from hw3.query_indexation import get_query_index_matrix
from hw3.similarity_calculator import calculate_distance, get_closest_document_name

import numpy as np
import os.path
import pickle
from tqdm import tqdm
import json


def get_query_index_matrix(queries: list, count_vectorizer) -> np.array:
    tf = count_vectorizer.transform(queries).toarray()
    return tf


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


def bm25_accuracy(questions_and_answers: dict, path: str = 'data.jsonl', n: int = 5) -> None:
    curr_dir = '/'.join(os.getcwd().split('/')[0:-1])
    if not os.path.isfile(curr_dir + '/hw3/corpus_matrix.npy'):
        texts = get_texts(path)
        original_texts, corpus = create_corpus(texts)
        corpus_matrix, count_vectorizer = calculate_bm25(corpus)

        with open(curr_dir + '/hw3/corpus_matrix.npy', 'wb') as f:
            np.save(f, corpus_matrix.astype(np.half))
        with open(curr_dir + '/hw3/original_texts', 'wb') as f2:
            pickle.dump(original_texts, f2)
        with open(curr_dir + '/hw3/count_vectorizer', 'wb') as f3:
            pickle.dump(count_vectorizer, f3)

    else:
        with open(curr_dir + '/hw3/original_texts', 'rb') as f1:
            original_texts = pickle.load(f1)
        with open(curr_dir + '/hw3/corpus_matrix.npy', 'rb') as f2:
            corpus_matrix = np.load(f2)
        with open(curr_dir + '/hw3/count_vectorizer', 'rb') as f3:
            count_vectorizer = pickle.load(f3)

    queries = (list(questions_and_answers.keys()))[:10000]
    queries_matrix = get_query_index_matrix(queries, count_vectorizer)
    distances = calculate_distance(queries_matrix, corpus_matrix)

    accuracy = []
    for i, column in tqdm(enumerate(distances.T)):
        indexes = np.argsort(column)[-n:]
        nearest_docs = np.take(list(original_texts.keys()), indexes)
        predicted_answers = [original_texts[doc] for doc in nearest_docs]
        right_answers = questions_and_answers[queries[i]]
        if len(set(predicted_answers) & set(right_answers)) > 0:
            accuracy.append(1)
        else:
            accuracy.append(0)
    return np.mean(accuracy)


def main(path: str = 'data.jsonl') -> None:
    corpus, questions_and_answers = get_corpus_and_questions(path)
    print(bm25_accuracy(questions_and_answers))


if __name__ == '__main__':
    main()
