import re
import string
from abc import ABC, abstractmethod
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from bs4 import BeautifulSoup


class Preprocessor(ABC):
    @abstractmethod
    def process(self, text: str) -> str:
        pass


class TFIDFPreprocessor(Preprocessor):
    def __init__(self, language: str = "english", use_stemming: bool = True):
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.stemmer = SnowballStemmer(language) if use_stemming else None

    def _remove_html(self, text: str) -> str:
        return BeautifulSoup(text, "lxml").get_text()

    def _remove_urls(self, text: str) -> str:
        return re.sub(r"http\S+|www\S+", "", text)

    def _remove_emojis(self, text: str) -> str:
        return re.sub(r"[\U00010000-\U0010ffff]", "", text)

    def _tokenize(self, text: str) -> List[str]:
        return text.split()

    def _normalize(self, tokens: List[str]) -> List[str]:
        cleaned = []
        for token in tokens:
            token = token.strip(string.punctuation).lower()
            if not token or token in self.stop_words:
                continue
            if self.stemmer:
                token = self.stemmer.stem(token)
            cleaned.append(token)
        return cleaned

    def process(self, text: str) -> str:
        text = self._remove_html(text)
        text = self._remove_urls(text)
        text = self._remove_emojis(text)
        tokens = self._tokenize(text)
        tokens = self._normalize(tokens)
        return " ".join(tokens)


if __name__ == "__main__":
    pre = TFIDFPreprocessor(language="english", use_stemming=True)
    sample = "Check out this link: https://example.com 😎<b>Hello</b> Running, runs, ran!"
    print(pre.process(sample))
