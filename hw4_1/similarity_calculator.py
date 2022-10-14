import numpy as np
from sklearn.metrics.pairwise import cosine_distances


def get_closest_document_name(
        docs: list,
        query_vector: np.ndarray,
        corpus_matrix: np.ndarray
) -> str:

    distances = cosine_distances(query_vector, corpus_matrix)[0]
    sorted_indexes = np.argsort(distances)
    return np.take(docs, sorted_indexes)
