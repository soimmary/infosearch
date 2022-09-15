from string import punctuation
from tqdm import tqdm
import os
import re

from pymystem3 import Mystem
from nltk.corpus import stopwords

from hw1.directories import get_directories

lemm = Mystem()
stopwords_rus = stopwords.words("russian")
punctuation += '—…«»'

'''
Препроцессинг данных:  
- приведение к одному регистру,
- удаление пунктуации и стоп-слов,
- лемматизация.
'''


def remove_punctuation(text: str) -> str:
    for i in punctuation:
        text = text.replace(i, ' ')
    return text


def lemmatization(text: str) -> list:
    return lemm.lemmatize(text.lower())


def remove_stopwords(text: list) -> str:
    clean_text = [word for word in text if word not in stopwords_rus]
    return clean_text


def preprocessing(text: str) -> str:
    text = remove_punctuation(text)
    text = lemmatization(text)
    text = remove_stopwords(text)
    return ' '.join(text)


def create_new_folder(new_dir):
    # Создаем новые директории
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)


def write_preprocessed_data(folder):
    create_new_folder(folder)
    # Записываем файлы в диpeктории
    dirs = get_directories('friends-data')
    for dir in tqdm(dirs):
        with open(dir, 'r', encoding='utf-8') as f:
            file = f.read()
            f.close()

        new_dir = f'{folder}/' + dir.split('/')[1]
        file_name = dir.split('/')[2]
        create_new_folder(new_dir)

        preprocessed_text = preprocessing(file)
        with open(f'{new_dir}/{file_name}', 'w', encoding='utf-8') as f:
            f.write(preprocessed_text)
            f.close()
