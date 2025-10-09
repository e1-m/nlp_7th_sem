import csv
from pathlib import Path
from statistics import mean

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


def evaluate_corpus(tokenizer, csv_path: str, number_of_rows_to_process: int = 250):
    en_metrics = {"ttr": [], "bpt": [], "cpt": []}
    ukr_metrics = {"ttr": [], "bpt": [], "cpt": []}

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for i, row in enumerate(reader):
            if i % 50 == 0:
                print(f"Processing row {i}")
            if i == number_of_rows_to_process:
                break

            en_text = row["en"].strip()
            uk_text = row["fixed_uk"].strip()

            # English
            en_tokens = tokenizer.get_tokens(tokenizer.encode(en_text))
            en_metrics["ttr"].append(tokens_to_words_ratio(en_text, en_tokens))
            en_metrics["bpt"].append(bytes_per_token(en_text, en_tokens))
            en_metrics["cpt"].append(chars_per_token(en_text, en_tokens))

            # Ukrainian
            uk_tokens = tokenizer.get_tokens(tokenizer.encode(uk_text))
            ukr_metrics["ttr"].append(tokens_to_words_ratio(uk_text, uk_tokens))
            ukr_metrics["bpt"].append(bytes_per_token(uk_text, uk_tokens))
            ukr_metrics["cpt"].append(chars_per_token(uk_text, uk_tokens))

    en_avg = {k: mean(v) for k, v in en_metrics.items() if v}
    uk_avg = {k: mean(v) for k, v in ukr_metrics.items() if v}

    print("\n[METRICS] Average results on corpus:")
    print(
        f"English → Tokens-to-Words: {en_avg['ttr']:.3f}, Bytes/Token: {en_avg['bpt']:.3f}, Chars/Token: {en_avg['cpt']:.3f}")
    print(
        f"Ukrainian → Tokens-to-Words: {uk_avg['ttr']:.3f}, Bytes/Token: {uk_avg['bpt']:.3f}, Chars/Token: {uk_avg['cpt']:.3f}")

    print("\n[COMPARISON]")
    print(f"Ukrainian requires {uk_avg['ttr'] / en_avg['ttr']:.2f}× more tokens per word (fertility).")
    print(f"Ukrainian tokens carry {uk_avg['cpt'] / en_avg['cpt']:.2f}× more characters on average.")
    print(f"Ukrainian tokens encode {uk_avg['bpt'] / en_avg['bpt']:.2f}× more bytes per token.")


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
    print(f"Merges learned: {len(tokenizer.merges)}\n\n")

    evaluate_corpus(
        tokenizer,
        (BASE_PATH / 'data' / 'testing' / "laws_eng_ukr.csv").as_posix()
    )


if __name__ == "__main__":
    main()
