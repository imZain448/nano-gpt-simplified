import torch
from tqdm import tqdm

from encode_data import DataEncoder
from models import BigramLanguageModel

## constants
DATA_PATH = "./data/tiny_shakespeare.txt"
VAIDATION_SPLIT = 0.1
BLOCK_SIZE = 8
BATCH_SIZE = 4
EPOCHS = 10000

## setting the seed
torch.manual_seed(44)

data_encoder = DataEncoder()
data = data_encoder.prepare_data(DATA_PATH, 1)

data = torch.tensor(data, dtype=torch.long)
print(f"data shape -- {data.shape}  | data dtype -- {data.dtype}")

print("SPLITTING THE DATASET")
split_idx = int(VAIDATION_SPLIT * len(data))
train_set = data[:split_idx]
val_set = data[split_idx:]
print(f"size of train set -- {len(train_set)}")
print(f"size of val set -- {len(val_set)}")


def get_batch(data_set):
    ix = torch.randint(len(data_set) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data_set[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data_set[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x, y


xb, yb = get_batch(train_set)
print("example input batch")
print(xb.shape)
print(xb)

print("example targets batch")
print(yb.shape)
print(yb)

print("EXAMPLE INPUT - TARGET PAIRS")
for b in range(BATCH_SIZE):
    for t in range(BLOCK_SIZE):
        x = xb[b, : t + 1]
        y = yb[b, t]
        print(f"INPUT -- {x.tolist()} | OUTPUT -- {y}")

print("CHECKING THE BIGRAM MODEL")
bigram_model = BigramLanguageModel(data_encoder.vocab_size)

logits, loss = bigram_model(xb, yb)
print(f"logits for first batch - {logits}")
print(f"loss from first batch - {loss}")

idx = torch.zeros((1, 1), dtype=torch.long)

print("generating using bigram model")
generated_text = data_encoder.decode(bigram_model.generate(idx, 100)[0].tolist())
print(f"generated_text -- {generated_text}")


print(f"TRAINING THE BIGRAM MODEL FOR {EPOCHS} EPOCHS")
optimizer = torch.optim.AdamW(bigram_model.parameters(), lr=1e-3)
for steps in tqdm(range(EPOCHS)):
    xb, yb = get_batch(train_set)
    logits, loss = bigram_model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"final loss after training -- {loss.item()}")

print("TEXT GENERATED AFTER TRAINING THE BIGRAM MODEL")
generated_text = data_encoder.decode(bigram_model.generate(idx, 1000)[0].tolist())
print(generated_text)
