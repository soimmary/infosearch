from hw1.indexation import create_index_matrix
from hw1.directories import get_directories

import numpy as np
from collections import Counter

'''
a) какое слово является самым частотным
b) какое самым редким
c) какой набор слов есть во всех документах коллекции
d) кто из главных героев статистически самый популярный (упонимается чаще всего)? Имена героев:
    Моника, Мон
    Рэйчел, Рейч
    Чендлер, Чэндлер, Чен
    Фиби, Фибс
    Росс
    Джоуи, Джои, Джо
'''


def count_lemma(value):
    counter = 0
    for info in value:
        counter += info[1]
    return counter


def most_common_word(index_matrix: list):
    lemmas, vectors = index_matrix
    sums = vectors.sum(axis=0)
    max_value = sums.max()
    max_value_arg = sums.argmax()
    max_value_word = lemmas[max_value_arg]
    return max_value_word


def rarest_word(index_matrix: list):
    lemmas, vectors = index_matrix
    sums = vectors.sum(axis=0)
    lemma_sum_dict = {}
    for lemma, num in zip(lemmas, sums):
        if not lemma[0].isdigit():
            lemma_sum_dict[lemma] = num
    return Counter(lemma_sum_dict).most_common()[-1][0]


def ubiquitous_word(index_matrix: list, main_dir: str):
    lemmas, vectors = index_matrix
    ubiquitous_words = []
    arg = 0
    for array in vectors:
        if np.all((array > 0)):
            ubiquitous_words.append(array)
        arg += 1
    if len(ubiquitous_words) == 0:
        return 'такие слова отсутствуют.'
    else:
         return ubiquitous_words


def most_popular_character(index_matrix: list):
    characters = [('моника', 'мон'), ('рэйчел', 'рэйч'),
                  ('чендлер', 'чэндлер', 'чен'), ('фиби','фибс'),
                  ('росс'), ('джоуи', 'джои', 'джо')]

    lemmas, vectors = index_matrix
    sums = vectors.sum(axis=0)
    lemma_sum_dict = {}
    for lemma, num in zip(lemmas, sums):
        if not lemma[0].isdigit():
            lemma_sum_dict[lemma] = num

    characters_dict = {}
    for character in characters:
        if len(character) > 1:
            count = 0
            for name in character:
                if name in lemma_sum_dict:
                    count += lemma_sum_dict[name]
        characters_dict[character] = count
    return Counter(characters_dict).most_common(1)[0][0]


def show_answers_matrix(index_matrix):
    print('Самое частое слово:', most_common_word(index_matrix))
    print('Одно из самых редких слов:', rarest_word(index_matrix))
    print('Какой набор слов есть во всех документах коллекции:', ubiquitous_word(index_matrix, 'preprocessed-data'))
    print('Кто из главных героев статистически самый популярный:', most_popular_character(index_matrix))
