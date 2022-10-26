import numpy as np


def get_query_index_matrix(query: str, vectorizer) -> np.array:
    query_vector = vectorizer.transform([query])
    return query_vector
