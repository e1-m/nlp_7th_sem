from abc import ABC, abstractmethod

import numpy as np


class Comparator(ABC):
    @abstractmethod
    def compare(self, query_vector, doc_matrix) -> np.ndarray:
        """Return similarity scores between query vector and all document vectors."""
        ...


class CosineSimilarity(Comparator):
    def compare(self, query_vector, doc_matrix):
        doc_norms = np.linalg.norm(doc_matrix, axis=1)
        query_norm = np.linalg.norm(query_vector)
        return (doc_matrix @ query_vector) / (doc_norms * query_norm + 1e-8)
