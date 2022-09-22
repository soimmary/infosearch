import typer
from preprocessing import preprocessing
from corpus_indexation import create_corpus, get_corpus_index_matrix
from query_indexation import get_query_index_matrix
from similarity_calculator import calculate_similarity, get_closest_document_name


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


def main(query: str, path: str = '../hw1/preprocessed-data') -> None:
    corpus = create_corpus(path)
    vectorizer, episodes, vocabulary, corpus_matrix = get_corpus_index_matrix(corpus)
    query = preprocessing(query)
    query_vectorized = get_query_index_matrix(query, vectorizer)
    distances = calculate_similarity(query_vectorized, corpus_matrix)
    nearest_doc = get_closest_document_name(distances, episodes)
    print(nearest_doc)


if __name__ == '__main__':
    typer.run(main)
