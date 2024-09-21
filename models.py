import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(44)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # output shape (B, T, V) --
        if targets == None:
            return logits, None
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_length):
        for _ in range(max_length):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            id_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, id_next), dim=1)
        return idx


class SelfAttentionHead(nn.Module):
    def __init__(self, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        weights = (
            q @ k.transpose(-2, -1) * C**-0.5
        )  # multiplying Q @ K_transpose with sqrt of C
        # to normalize the weights and preseve variance
        # mathematically (B, T, C) x (B , C, T) --> (B, T, T)

        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # B,T,T
        weights = F.softmax(weights, dim=-1)  # B,T,T
        weights = self.dropout(weights)
        v = self.value(x)  # B,T,C
        out = weights @ v  # B,T,T x B,T,C --> B,T,C
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(head_size, n_embed, block_size, dropout)
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForwardBlock(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU(),
            nn.Linear(n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    def __init__(self, n_embed, num_heads, block_size, dropout):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size, n_embed, block_size, dropout)
        self.ffn = FeedForwardBlock(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class SelfAttentionModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embed,
        block_size,
        head_size,
        num_heads=1,
        dropout=0,
        num_layers=0,
        device="cpu",
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        if num_layers > 0:
            print(
                "initializing the self attention model with multi head attention and multiple layers"
            )
            self.attention_block = nn.Sequential(
                *[
                    AttentionBlock(n_embed, num_heads, block_size, dropout)
                    for _ in range(num_layers)
                ]
            )
        elif num_heads > 1:  # use multi head attention
            print("initializing the self attention model with multi head attention")
            self.attention_block = MultiHeadAttention(
                num_heads, head_size, n_embed, block_size, dropout
            )
        else:  # use single self attention
            print("initializing the self attention model with single head attention")
            self.attention_block = SelfAttentionHead(
                head_size, n_embed, block_size, dropout
            )

        self.last_layer_norm = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.block_size = block_size
        self.device = device

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embedding = self.token_embedding_table(
            idx
        )  # output shape (B, T, n_embed) --
        position_embedding = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # output shape (B, T, V)
        x = token_embedding * position_embedding  # output shape -- (B,T,n_embed)
        x = self.attention_block(x)  # calling the self attention --> (B,T,n_embed)
        x = self.last_layer_norm(x)
        logits = self.lm_head(x)  # output shape -- (B, T, V)
        if targets == None:
            return logits, None
        # NOTE: the C in B, T, C is the chanel dimension which is same as V (vocab_size)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_length):
        for _ in range(max_length):
            idx_cropped = idx[
                :, -self.block_size :
            ]  # slicing the idx to limit to block size
            logits, loss = self(idx_cropped)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            id_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, id_next), dim=1)
        return idx
