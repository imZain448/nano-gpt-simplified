import torch
from tqdm import tqdm

from encode_data import DataEncoder
from models import SelfAttentionModel

## constants
DATA_PATH = "./data/tiny_shakespeare.txt"
VAIDATION_SPLIT = 0.1
BLOCK_SIZE = 8
BATCH_SIZE = 4
EPOCHS = 10000
N_EMBED = 32
EVAL_INTERVALS = 500
HEAD_SIZE = 32

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

model = SelfAttentionModel(data_encoder.vocab_size, N_EMBED, BLOCK_SIZE, HEAD_SIZE)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for steps in tqdm(range(EPOCHS)):
    xb, yb = get_batch(train_set)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"final loss after training -- {loss.item()}")

print("TEXT GENERATED AFTER TRAINING THE BIGRAM MODEL")

idx = torch.zeros((1, 1), dtype=torch.long)
generated_text = data_encoder.decode(model.generate(idx, 1000)[0].tolist())
print(generated_text)
