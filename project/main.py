from get_texts import corpus
from tfidf_main import main as main_tfidf
from bm25_main import main as main_bm25
from bert_main import main as main_bert

import streamlit as st
import logging
import time


def search(corpus, query: str, method: str, n_results: int) -> list:
    if method == 'TF-IDF':
        return main_tfidf(query, n_results)
    elif method == 'BM25':
        return main_bm25(query, n_results)
    elif method == 'BERT':
        return main_bert(corpus, query, n_results)


def main(corpus) -> None:
    st.title('В поисках любви')
    st.header('Поиск')
    st.info('Привет всем')

    query = st.text_input('Введите запрос')
    left_column, right_column = st.columns(2)
    method = left_column.selectbox('Выберите метод:', ['TF-IDF', 'BM25', 'BERT'])
    n_results = right_column.selectbox('Выберите число ответов на странице:', [i for i in range(5, 31, 5)])
    is_search_clicked = st.button('Search')
    st.markdown('---')

    # Show results
    if is_search_clicked:
        if len(query) > 10:
            st.header(f'Результаты поиска по запросу "{query[:10]}..."')
        else:
            st.header(f'Результаты поиска по запросу "{query}"')

    start_time = time.time()

    if query != '':
        results = search(corpus, query, method, n_results)
        st.write('Время поиска:', -(start_time - time.time()))
        for result in results:
            st.write(result)


logging.getLogger().setLevel(logging.INFO)
main(corpus)
