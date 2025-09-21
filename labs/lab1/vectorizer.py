import math
from abc import abstractmethod, ABC
from collections import Counter


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

    def fit(self, documents: list[str]):
        self.vocab = sorted(set(word for doc in documents for word in doc.split()))

        self.idf = {}
        for word in self.vocab:
            docs_with_word = sum(1 for doc in documents if word in doc)
            self.idf[word] = math.log(len(documents) / (1 + docs_with_word))

    def vectorize(self, text: str) -> list[float]:
        tokens = text.split()
        tf_counts = Counter(tokens)
        length = len(tokens) if tokens else 1

        vector = []
        for word in self.vocab:
            tf = tf_counts[word] / length
            vector.append(tf * self.idf[word])

        return vector


if __name__ == "__main__":
    docs = [
        "the cat sat on the mat",
        "the dog barked at the cat",
        "cats and dogs are friends"
    ]
    query = "cat dog"

    vectorizer = TFIDFVectorizer()
    vectorizer.fit(docs)
    vec = vectorizer.vectorize(query)
    print(vec)
