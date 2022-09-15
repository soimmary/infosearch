from hw1.directories import get_directories
from hw1.preprocessing import write_preprocessed_data
from hw1.indexation import create_index_matrix, create_index_dict
from hw1.search_in_dict import show_answers_dict
from hw1.search_in_matrix import show_answers_matrix


if __name__ == '__main__':
    get_directories('friends-data')
    write_preprocessed_data('preprocessed-data')
    index_matrix = create_index_matrix('preprocessed-data')
    index_dict = create_index_dict('preprocessed-data')

    print('1. Ответы для словаря')
    show_answers_dict(index_dict)
    print('2. Ответы для матрицы')
    show_answers_matrix(index_matrix)
