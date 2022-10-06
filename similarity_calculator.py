import numpy as np
from sklearn.metrics.pairwise import cosine_distances


def calculate_distance(
        query_vector: np.ndarray,
        corpus_matrix: np.ndarray) -> np.ndarray:
    bm25 = corpus_matrix @ query_vector.T
    return bm25


def get_closest_document_name(distances: np.ndarray, docs: list) -> str:
    sorted_indexes = np.argsort(distances, axis=0)[::-1]
    return np.take(docs, sorted_indexes)
