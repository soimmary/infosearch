import numpy as np


def get_query_index_matrix(query: str, count_vectorizer) -> np.array:
    tf = count_vectorizer.transform([query])
    return tf

