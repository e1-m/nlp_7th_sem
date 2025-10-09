import re


def tokens_to_words_ratio(text: str, tokens: list[str]) -> float:
    words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    if not words or not tokens:
        return 0.0
    return len(tokens) / len(words)


def bytes_per_token(text: str, tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(text.encode("utf-8")) / len(tokens)


def chars_per_token(text: str, tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(text) / len(tokens)
