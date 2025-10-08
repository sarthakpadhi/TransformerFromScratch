import torch
import torch.nn as nn
from torch.nn import functional as F


# --- hyperparameters ---
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
new_tokens_generated = 500
# -----------------------



torch.manual_seed(124)


with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)


string_to_ind = {ch: i for i, ch in enumerate(chars)}
ind_to_string = {i: ch for i, ch in enumerate(chars)}


def encode(string):
    return [string_to_ind[ch] for ch in string]


def decode(ind):
    return "".join(ind_to_string[i] for i in ind)


data = torch.tensor(
    encode(text), dtype=torch.long
)  ## here we tokenized the entire dataset in


n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
batch_size = 4


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()  ## this is so that we don't train the model during this inference. i think by default every inference trains the model
def estimate_loss():
    out = {}
    model.eval() ## different layers will have different behaviour when eval and training. 
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() ## here this is linked to eval. 
    return out


class BLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        """we are defining the table here helpful for inference"""
        self.token_embedding_table = nn.Embedding(
            vocab_size, vocab_size
        )  # B -> batch, T -> time step or block size, C -> dimension of each token
        ## this predicts for each char, what should be the next char.

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  ## B, C
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            logits, loss = self(idx)  ## this was B*T, C
            logits = logits[:, -1, :]  # becomes (B, C), we only take the last predicted
            probs = F.softmax(logits, dim=1)  # (B, C), softmax
            idx_next = torch.multinomial(
                probs, num_samples=1
            )  # (B, 1), this is the sampling thing. we don't take argmax, but we sample from a prob distribution.
            ## otherwise we would get deterministic outputs. in the end wre getting a prob distribution that we then choose from.
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1), add it back
        return idx


model = BLM(vocab_size)
m = model.to(device)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=new_tokens_generated)[0].tolist()))

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)


for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss: {losses['train']} | eval loss {losses['val']}")

    xb, yb = get_batch("train")
    logits, losses = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    losses.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=new_tokens_generated)[0].tolist()))
