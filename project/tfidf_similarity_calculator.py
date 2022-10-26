import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import logging


def calculate_similarity(
        query_vector: np.ndarray,
        corpus_matrix: np.ndarray) -> np.ndarray:
    return 1 - cosine_similarity(query_vector, corpus_matrix)[0]


def get_closest_document_name(distances: np.ndarray, episodes: list) -> str:
    sorted_indexes = np.argsort(distances)
    return np.take(episodes, sorted_indexes)
