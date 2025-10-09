from pathlib import Path

from lab2.src.tokenizer import BPETokenizer
from lab2.src.utils import bytes_per_token, tokens_to_words_ratio, chars_per_token

BASE_PATH = Path(__file__).parent.parent


def print_stats(tokenizer, sentence):
    encoded = tokenizer.encode(sentence)
    tokens = tokenizer.get_tokens(encoded)
    decoded = tokenizer.decode(encoded)

    print("\n[TEST] Encoding/Decoding:")
    print(f"Original: {sentence}")
    print(f"Encoded IDs: {encoded}")
    print(f"Encoded Tokens: {tokens}")
    print(f"Decoded text: {decoded}")
    print(f"Bytes per token: {bytes_per_token(sentence, tokens)}")
    print(f"Tokens to words: {tokens_to_words_ratio(sentence, tokens)}")
    print(f"Chars per token: {chars_per_token(sentence, tokens)}")


def main():
    vocab_path = BASE_PATH / "vocabs" / "en_ukr_tokenizer_vocab.json"

    training_texts_base_path = BASE_PATH / 'data' / 'training'
    training_text_paths = [
        training_texts_base_path / 'constitutional_patriotism_en.txt',
        training_texts_base_path / 'habermas_ukr.txt',
        training_texts_base_path / 'rousseau_en.txt',
        training_texts_base_path / 'rousseau_ukr.txt',
    ]

    tokenizer = BPETokenizer(vocab_size=3000, min_freq=3)

    train = False

    if train:
        tokenizer.train("\n\n".join(path.read_text(encoding="utf-8") for path in training_text_paths))
        tokenizer.save(vocab_path)
    else:
        tokenizer.load(vocab_path)

    print("\nBPETokenizer")
    print(f"Vocab size: {len(tokenizer.vocab)}")
    print(f"Merges learned: {len(tokenizer.merges)}")


if __name__ == "__main__":
    main()
