from string import punctuation

from pymystem3 import Mystem
from nltk.corpus import stopwords

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
