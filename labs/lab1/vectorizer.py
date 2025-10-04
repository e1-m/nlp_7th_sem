from abc import abstractmethod, ABC
from collections import Counter

import numpy as np


class Vectorizer(ABC):
    @abstractmethod
    def fit(self, documents: list[str]):
        pass

    @abstractmethod
    def vectorize(self, test: str) -> list[float]:
        pass


class TFIDFVectorizer(Vectorizer):
    def __init__(self):
        self.idf = None
        self.vocab = None
        self._word_index = None

    def fit(self, documents: list[str]):
        vocab = sorted(set(word for doc in documents for word in doc.split()))
        n_docs = len(documents)

        doc_freq = np.zeros(len(vocab), dtype=np.float32)
        for i, word in enumerate(vocab):
            doc_freq[i] = sum(1 for doc in documents if word in doc)

        idf = np.log(n_docs / (1 + doc_freq))

        self.vocab = vocab
        self.idf = idf
        self._word_index = {w: i for i, w in enumerate(vocab)}

    def vectorize(self, text: str) -> np.ndarray:
        tokens = text.split()
        if not tokens:
            return np.zeros(len(self.vocab), dtype=np.float32)

        tf_counts = Counter(tokens)
        length = len(tokens)

        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for word, count in tf_counts.items():
            idx = self._word_index.get(word)
            if idx is not None:
                vec[idx] = (count / length) * self.idf[idx]

        return vec


class BM25Vectorizer(Vectorizer):
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.idf = None
        self.vocab = None
        self.avg_doc_len = None
        self._word_index = None

    def fit(self, documents: list[str]):
        vocab = sorted(set(word for doc in documents for word in doc.split()))
        n = len(documents)
        doc_freq = np.zeros(len(vocab), dtype=np.float32)

        for i, word in enumerate(vocab):
            doc_freq[i] = sum(1 for doc in documents if word in doc)

        idf = np.log((n - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
        avg_len = np.mean([len(doc.split()) for doc in documents])

        self.vocab = vocab
        self.idf = idf
        self.avg_doc_len = avg_len
        self._word_index = {w: i for i, w in enumerate(vocab)}

    def vectorize(self, text: str) -> np.ndarray:
        tokens = text.split()
        if not tokens:
            return np.zeros(len(self.vocab), dtype=np.float32)

        tf_counts = Counter(tokens)
        doc_len = len(tokens)

        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for word, tf in tf_counts.items():
            idx = self._word_index.get(word)
            if idx is None:
                continue

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            vec[idx] = self.idf[idx] * (numerator / denominator)

        return vec
