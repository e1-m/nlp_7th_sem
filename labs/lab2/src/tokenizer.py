import json
from collections import Counter


def count_adjacent_pairs(corpus):
    """
    Count the frequency of adjacent token pairs across the corpus.
    Example: [['t', 'h', 'e'], ['d', 'o', 'g']] → {('t','h'):1, ('h','e'):1, ('d','o'):1, ('o','g'):1}
    """
    pairs = Counter()

    for tokens in corpus:
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1

    return pairs


def _apply_merge(tokens, a, b):
    merged = a + b
    result = []
    i = 0

    while i < len(tokens):
        is_merge_candidate = (
                i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b
        )

        if is_merge_candidate:
            result.append(merged)
            i += 2
        else:
            result.append(tokens[i])
            i += 1

    return result


def _prepare_corpus(text):
    corpus = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            tokens = list(line.replace(" ", "▁"))
            corpus.append(tokens)
    return corpus


class BPETokenizer:
    def __init__(self, vocab_size=1000, min_freq=2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.vocab = {}  # token → id
        self.idx2token = {}  # id → token
        self.merges = []  # list of (a, b) merge rules in order

    def train(self, text):
        corpus = _prepare_corpus(text)
        self._initialize_vocab(corpus)

        while len(self.vocab) < self.vocab_size:
            if not (pair_frequencies := count_adjacent_pairs(corpus)):
                break

            (a, b), freq = pair_frequencies.most_common(1)[0]
            if freq < self.min_freq:
                break

            corpus = [_apply_merge(tokens, a, b) for tokens in corpus]
            self._register_merge((a, b))

        print(f"Training complete — vocab size: {len(self.vocab)}, merges: {len(self.merges)}")

    def _initialize_vocab(self, corpus):
        unique_tokens = sorted({token for tokens in corpus for token in tokens})
        self.vocab = {token: idx for idx, token in enumerate(unique_tokens)}
        self.idx2token = {idx: token for token, idx in self.vocab.items()}

    def _register_merge(self, pair):
        a, b = pair
        merged = a + b

        self.merges.append(pair)
        new_id = len(self.vocab)
        self.vocab[merged] = new_id
        self.idx2token[new_id] = merged

    def encode(self, text):
        tokens = list(text.replace(" ", "▁"))

        for a, b in self.merges:
            tokens = _apply_merge(tokens, a, b)

        return [self.vocab.get(token, self.vocab.get("▁", 0)) for token in tokens]

    def get_tokens(self, ids):
        return [self.idx2token[token_id] for token_id in ids]

    def decode(self, token_ids):
        tokens = [self.idx2token[token_id] for token_id in token_ids]
        return "".join(tokens).replace("▁", " ")

    def save(self, filepath):
        data = {
            "vocab": self.vocab,
            "merges": self.merges,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab = data["vocab"]
        self.merges = [tuple(pair) for pair in data["merges"]]
        self.idx2token = {idx: token for token, idx in self.vocab.items()}
