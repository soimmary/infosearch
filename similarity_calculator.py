import numpy as np
from sklearn.metrics.pairwise import cosine_distances


def calculate_similarity(
        query_vector: np.ndarray,
        corpus_matrix: np.ndarray) -> np.ndarray:
    return cosine_distances(query_vector, corpus_matrix)[0]


def get_closest_document_name(distances: np.ndarray, episodes: list) -> str:
    sorted_indexes = np.argsort(distances)
    return np.take(episodes, sorted_indexes)
