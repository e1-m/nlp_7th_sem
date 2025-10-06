from pathlib import Path

from lab2.src.tokenizer import BPETokenizer

BASE_PATH = Path(__file__).parent.parent


def main():
    train = False
    vocab_path = BASE_PATH / "vocabs" / "bpe_tokenizer.json"

    tokenizer = BPETokenizer(vocab_size=1000, min_freq=2)

    if train:
        training_text_path = BASE_PATH / 'data' / 'training' / 'constitutional_patriotism.txt'
        tokenizer.train(training_text_path.read_text())
        tokenizer.save(vocab_path)
    else:
        tokenizer.load(vocab_path)

    print("\n[INFO] Training complete.")
    print(f"Vocab size: {len(tokenizer.vocab)}")
    print(f"Merges learned: {len(tokenizer.merges)}")

    test_sentence = "the dog sleeps on the constitution"
    encoded = tokenizer.encode(test_sentence)
    tokens = tokenizer.get_tokens(encoded)
    decoded = tokenizer.decode(encoded)

    print("\n[TEST] Encoding/Decoding demo:")
    print(f"Original: {test_sentence}")
    print(f"Encoded IDs: {encoded}")
    print(f"Encoded Tokens: {tokens}")
    print(f"Decoded text: {decoded}")


if __name__ == "__main__":
    main()
