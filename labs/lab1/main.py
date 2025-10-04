import ast

import pandas as pd

from lab1.engine import SearchEngine
from lab1.preprocessor import TFIDFPreprocessor
from lab1.vectorizer import TFIDFVectorizer, BM25Vectorizer
from lab1.comparator import CosineSimilarity
from lab1.repository import CSVDocumentRepository


def main():
    queries = pd.read_csv("data/queries.csv", sep="|")
    queries["expected_results"] = queries["expected_results"].apply(ast.literal_eval)
    query_result_map = dict(zip(queries["queries"], queries["expected_results"]))

    engine = SearchEngine(
        repository=CSVDocumentRepository("data/corpus.csv"),
        preprocessor=TFIDFPreprocessor(language="english", use_stemming=True),
        vectorizer=BM25Vectorizer(k1=1.2, b=0.75),
        comparator=CosineSimilarity(),
    )

    total_precision = 0
    total_recall = 0
    total_f1 = 0
    n = len(query_result_map)

    for query, expected in query_result_map.items():
        results = engine.search(query, top_k=30)

        expected_set = set(expected)
        retrieved_set = set(results)

        true_positives = len(expected_set & retrieved_set)

        precision = true_positives / len(retrieved_set) if retrieved_set else 0
        recall = true_positives / len(expected_set) if expected_set else 0

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        total_precision += precision
        total_recall += recall
        total_f1 += f1

    avg_precision = total_precision / n
    avg_recall = total_recall / n
    avg_f1 = total_f1 / n

    print(f"Precision: {avg_precision:.3f}")
    print(f"Recall:    {avg_recall:.3f}")
    print(f"F1-score:  {avg_f1:.3f}")


if __name__ == "__main__":
    # nltk.download("stopwords")
    # nltk.download("wordnet")

    main()
