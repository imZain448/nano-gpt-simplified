import torch
from tqdm import tqdm

from encode_data import DataEncoder
from models import SelfAttentionModel

## constants
DATA_PATH = "./data/tiny_shakespeare.txt"
VAIDATION_SPLIT = 0.1
BLOCK_SIZE = 256
BATCH_SIZE = 64
TRAIN_STEPS = 4000
EVAL_STEPS = 200
N_EMBED = 384
EVAL_INTERVALS = 500
EPOCHS = TRAIN_STEPS // EVAL_INTERVALS
NUM_HEADS = 6
HEAD_SIZE = N_EMBED // NUM_HEADS
NUM_LAYERS = 6
DROPOUT = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINTS_PATH = "./ckpt/attention_checkpoints_v1_ckpt.pt"
BEST_VAL_LOSS = float("inf")

print(f"RNNING ON {DEVICE}")

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


def get_batch(split):
    data_set = train_set if split == "train" else val_set
    ix = torch.randint(len(data_set) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data_set[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data_set[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


xb, yb = get_batch(train_set)
print("example input batch")
print(xb.shape)
print(xb)

print("example targets batch")
print(yb.shape)
print(yb)
model = SelfAttentionModel(
    data_encoder.vocab_size,
    N_EMBED,
    BLOCK_SIZE,
    HEAD_SIZE,
    NUM_HEADS,
    DROPOUT,
    NUM_LAYERS,
    DEVICE,
)

model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


@torch.no_grad()
def estimate_loss():
    output = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_STEPS)
        for k in tqdm(range(EVAL_STEPS), colour="green"):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        output[split] = losses.mean()
    model.train()
    return output


for epoch in range(EPOCHS):
    print(f"Running EPOCH {epoch+1}/{EPOCHS}")
    for steps in tqdm(range(EVAL_INTERVALS), colour="red"):
        xb, yb = get_batch(train_set)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print("Evaluating")
    losses = estimate_loss()
    print(f"Train loss : {losses['train']} | val loss : {losses['val']}")
    if losses["val"] < BEST_VAL_LOSS:
        print(
            f"""validation loss improved {BEST_VAL_LOSS} to {losses['val']}
                saving the model to {CHECKPOINTS_PATH}
            """
        )
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": losses["val"],
        }
        BEST_VAL_LOSS = losses["val"]
        torch.save(checkpoint, CHECKPOINTS_PATH)
    print()

# print(f"final loss after training -- {loss.item()}")

print("TEXT GENERATED AFTER TRAINING THE ATTENTION MODEL")

idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
generated_text = data_encoder.decode(model.generate(idx, 1000)[0].tolist())
print(generated_text)
