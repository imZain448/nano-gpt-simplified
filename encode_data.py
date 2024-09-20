import os


class DataEncoder:
    def __init__(self):
        self.c2i = None
        self.i2c = None

    def prepare_data(self, file_path, verbose=0):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.c2i = {ch: i for i, ch in enumerate(self.chars)}
        self.i2c = {i: ch for i, ch in enumerate(self.chars)}
        if verbose > 0:
            print(f"SIZE OF THE SET = {len(text)}")
            print(f"SYMBOL SET = {self.chars}")
            print(f"NUMBER OF SYMBOLS = {len(self.chars)}")
        data = [self.c2i[char] for char in text]
        return data

    def encode(self, text):
        if not self.c2i:
            raise Exception(
                "char to index map not exist please call prepare data first"
            )
        return [self.c2i[c] for c in text]

    def decode(self, ids):
        if not self.i2c:
            raise Exception(
                "index to character map not exist please call prepare data first"
            )
        return "".join([self.i2c[i] for i in ids])


if __name__ == "__main__":
    print(f"getting the data")

    with open("./data/tiny_shakespeare.txt", "r", encoding="utf-8") as file:
        data = file.read()

    print("SAMPEL FIRST 1000 CHARACTERS")
    print(data[:1000])

    print()
    print(f"TOTAL NUMBER OF CHARACTERS IN THE DATASET = {len(data)}")
    chars = sorted(list(set(data)))
    vocab_size = len(chars)

    print(f"SYMBOL SET = {chars}")
    print(f"NUMBER OF SYMBOLS = {len(chars)}")

    print("GENERATING ENCODINGS")
    c2i = {ch: i for i, ch in enumerate(chars)}
    i2c = {i: ch for i, ch in enumerate(chars)}

    def encode(text):
        return [c2i[c] for c in text]

    def decode(ids):
        return "".join([i2c[i] for i in ids])

    print(f"encoding of 'hi there' {encode('hi there')}")
    print(f"decoding back {decode(encode('hi there'))}")
