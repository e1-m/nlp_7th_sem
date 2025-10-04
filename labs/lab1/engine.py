import numpy as np

from lab1.comparator import Comparator
from lab1.preprocessor import Preprocessor
from lab1.repository import DocumentRepository
from lab1.vectorizer import Vectorizer


class SearchEngine:
    def __init__(self,
                 preprocessor: Preprocessor,
                 repository: DocumentRepository,
                 vectorizer: Vectorizer,
                 comparator: Comparator):
        self.preprocessor = preprocessor
        self.repository = repository
        self.vectorizer = vectorizer
        self.comparator = comparator

        self._fitted = False
        self._doc_ids = None
        self._doc_matrix = None

    def _preprocess_documents(self) -> dict[int, str]:
        docs = self.repository.get_all_documents()
        return {
            doc.id: self.preprocessor.process(f"{doc.display_name} {doc.description}")
            for doc in docs
        }

    def fit(self):
        processed_docs = self._preprocess_documents()
        self.vectorizer.fit(list(processed_docs.values()))

        self._doc_ids = np.array(list(processed_docs.keys()))
        self._doc_matrix = np.vstack([
            self.vectorizer.vectorize(text) for text in processed_docs.values()
        ])
        self._fitted = True

    def search(self, query: str, top_k: int = 10):
        if not self._fitted:
            self.fit()

        processed_query = self.preprocessor.process(query)
        query_vector = self.vectorizer.vectorize(processed_query)

        sims = self.comparator.compare(query_vector, self._doc_matrix)
        top_indices = np.argsort(sims)[::-1][:top_k]

        return [int(self._doc_ids[i]) for i in top_indices]
