from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from lab1.comparator import Comparator
from lab1.preprocessor import Preprocessor
from lab1.repository import DocumentRepository
from lab1.vectorizer import Vectorizer


class SearchEngine:
    def __init__(self,
                 preprocessor: Preprocessor,
                 repository: DocumentRepository,
                 vectorizer: Vectorizer,
                 comparator: Comparator, ):
        self.preprocessor = preprocessor
        self.repository = repository
        self.vectorizer = vectorizer
        self.comparator = comparator

    def _preprocess_documents(self) -> dict[int, str]:
        documents = self.repository.get_all_documents()
        processed_docs = {}
        for doc in documents:
            processed_docs[doc.id] = self.preprocessor.process(f"{doc.display_name} {doc.description}")
        return processed_docs

    def search(self, query: str):
        processed_documents = self._preprocess_documents()

        processed_query = self.preprocessor.process(query)
        self.vectorizer.fit(list(processed_documents.values()))
        query_vector = self.vectorizer.vectorize(processed_query)
        results = []
        for doc_id, document in processed_documents.items():
            vector = self.vectorizer.vectorize(document)
            similarity = self.comparator.compare(query_vector, vector)
            results.append((doc_id, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results


class SklearnSearchEngine:
    def __init__(self, repository, preprocessor):
        self.repository = repository
        self.preprocessor = preprocessor
        self.vectorizer = TfidfVectorizer()

    def _preprocess_documents(self) -> dict[int, str]:
        documents = self.repository.get_all_documents()
        processed_docs = {}
        for doc in documents:
            processed_docs[doc.id] = self.preprocessor.process(f"{doc.display_name} {doc.description}")
        return processed_docs

    def search(self, query: str, top_k: int | None = None) -> list[tuple[int, float]]:
        documents = self._preprocess_documents()

        doc_ids, texts = zip(*documents.items())
        doc_matrix = self.vectorizer.fit_transform(texts)

        query_vector = self.vectorizer.transform([query])

        similarities = cosine_similarity(query_vector, doc_matrix).flatten()

        results = sorted(zip(doc_ids, similarities), key=lambda x: x[1], reverse=True)

        return results if top_k is None else results[:top_k]
