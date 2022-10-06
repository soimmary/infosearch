from string import punctuation
from tqdm import tqdm

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
    clean_text = [word for word in text if (word not in stopwords_rus) and (word != ' ')]
    return ' '.join(clean_text)


def create_corpus(texts: list) -> dict:
    corpus = {}
    original_texts = {}
    for text in tqdm(texts, desc='Лемматизирую тексты'):
        lemm_text = remove_punctuation(text)
        lemm_text = lemmatization(lemm_text)
        lemm_text = remove_stopwords(lemm_text)
        original_texts[lemm_text] = text
        corpus[lemm_text] = lemm_text

    return original_texts, corpus
