from hw1.indexation import create_index_dict
from hw1.directories import get_directories
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


def most_common_word(index_dict: list):
    new_dic = {}
    for key, value in index_dict.items():
        new_dic[key] = count_lemma(value)
    return Counter(new_dic).most_common(1)[0][0]


def rarest_word(index_dict: list):
    new_dic = {}
    for key, value in index_dict.items():
        new_dic[key] = -count_lemma(value)
    return Counter(new_dic).most_common(1)[0][0]


def ubiquitous_word(index_dict: list, main_dir: str):
    ubiquitous_words = []
    for key, value in index_dict.items():
        num_of_docs = len(get_directories(main_dir))
        if len(value) >= num_of_docs:
            ubiquitous_words.append(key)
    if len(ubiquitous_words) == 0:
        return 'такие слова отсутствуют.'
    else:
        return ubiquitous_words


def most_popular_character(index_dict: list):
    characters = [('моника', 'мон'), ('рэйчел', 'рэйч'),
                  ('чендлер', 'чэндлер', 'чен'), ('фиби',
                                                  'фибс'), ('росс'), ('джоуи', 'джои', 'джо')]
    new_dic = {}
    for key, value in index_dict.items():
        new_dic[key] = count_lemma(value)
    characters_dict = {}
    for character in characters:
        if len(character) > 1:
            count = 0
            for name in character:
                if name in new_dic:
                    count += new_dic[name]
        characters_dict[character] = count
    return Counter(characters_dict).most_common(1)[0][0]


def show_answers_dict(index_dict):
    print('Самое частое слово:', most_common_word(index_dict))
    print('Одно из самых редких слов:', rarest_word(index_dict))
    print('Какой набор слов есть во всех документах коллекции:', ubiquitous_word(index_dict, 'preprocessed-data'))
    print('Кто из главных героев статистически самый популярный:', most_popular_character(index_dict))
