import torch
import torch.nn as nn
from torch.nn import LayerNorm, functional as F

torch.manual_seed(124)


with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(vocab_size)


# --- hyperparameters ---
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
new_tokens_generated = 500
n_embed = 384
n_heads = 6
dropout = 0.2
n_layer = 4
# -----------------------


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


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()  ## this is so that we don't train the model during this inference. i think by default every inference trains the model
def estimate_loss():
    out = {}
    model.eval()  ## different layers will have different behaviour when eval and training.
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  ## here this is linked to eval.
    return out


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5  ## B, T, 16 @ B. 16, T = B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size=head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.sa_head = MultiHead(n_head, n_embed // n_head)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.sa_head(x)  ## residuals
        x = self.ln2(x)
        x = x + self.ffwd(x)
        return x


class BLM(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # B -> batch, T -> time step or block size, C -> dimension of each token
        ## this predicts for each char, what should be the next char.
        self.pos_embedding_table = nn.Embedding(block_size, n_embed)
        ## so here, we see that the n_embed is the dimension of vector space that we embed our token into.
        # self.sa_head = MultiHead(4, n_embed // 4)
        # self.ffwd = FeedForward(n_embed)

        self.blocks = nn.Sequential(*[Block(n_embed, n_heads) for _ in range(n_layer)])
        self.ln_final = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embed = self.token_embedding_table(
            idx
        )  ## B, T ,C -> C here would be n_embed
        pos_embed = self.pos_embedding_table(
            torch.arange(T, device=device)
        )  ## but this is really constant for any idx that comes in right?
        x = token_embed + pos_embed
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)  # B, T, vocab size

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
            idx_crop = idx[:, -block_size:]
            logits, loss = self(idx_crop)  ## this was B*T, C
            logits = logits[:, -1, :]  # becomes (B, C), we only take the last predicted
            probs = F.softmax(logits, dim=1)  # (B, C), softmax
            idx_next = torch.multinomial(
                probs, num_samples=1
            )  # (B, 1), this is the sampling thing. we don't take argmax, but we sample from a prob distribution.
            ## otherwise we would get deterministic outputs. in the end wre getting a prob distribution that we then choose from.
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1), add it back
        return idx


model = BLM()
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




"""
the following output is trained in google colab, with a 15gb T4 GPU. Run time  was 33 mins. 


sVjoPQ?f.IQaULmCj $tGIBmbc?!GnaJyezk!-A!k
ANWTnT
oNDRTgk-V3r,,oXWa-sQIwrcRcGaW.skZ?rQneRcbXlxLxVTEvFmEbbYcIOXlOHSg-SORJRkIm$$A.GWQ.cbugRM
$rjJBqw.iiMx:sABpez:s,,Dj :Xk-c&PwjA3WFM'nPmd-pq?3,yQj?ePHFjsWaZ!&CqtlavalWMwe;Eo&WR;IipU  FUe:tcu-JoxP's.p.FW-zslmH;:ZmEryiltAeMjgrqA:Z$ocXSbiZR$'3mmjlQa:cSdf
Ld!afj-AmEDVmh-E;BnO:b;p'DN!YNnYm'M!mSKw$i$VcHMNfJzzZVmP'Cym&!vVCIvV:TV!bcH:Zq:'ipZ::Xtns zqtLSbFt:VZlcqNp-YeSUgHrJTiTvv,kj:XYinFKXt:Th..z!sJyGISLLZ3 3s:SsreskM.mSnRunM.fI'XwnVTSzDznsDSLdnXXb
T'Uxyx:?n$
step 0: train loss: 4.323126792907715 | eval loss 4.322063446044922
step 500: train loss: 2.001328229904175 | eval loss 2.0856096744537354
step 1000: train loss: 1.6190102100372314 | eval loss 1.7991306781768799
step 1500: train loss: 1.4601850509643555 | eval loss 1.654475212097168
step 2000: train loss: 1.3712459802627563 | eval loss 1.5908710956573486
step 2500: train loss: 1.3144934177398682 | eval loss 1.5491490364074707
step 3000: train loss: 1.271044373512268 | eval loss 1.5179145336151123
step 3500: train loss: 1.2356680631637573 | eval loss 1.504164695739746
step 4000: train loss: 1.2041224241256714 | eval loss 1.492590308189392
step 4500: train loss: 1.1751224994659424 | eval loss 1.4917203187942505

Is she, deed, such it his commond lineament'd?

ISABELLA:
The hangest, prick my wife.

ALLET:
The sacred she nueon.

LUCIO:
ConDidinius of Isabe, you might heavenly know;
Good to myself are come, and now the world: Bohemia
botters thus thing so.

ISABELLA:
I hope by thy neither now she, bring of,
Never looks only for our so acquest
That was as languaged's with Romeo!
You are up her all: if thou leaven'd to flatter,
Nor than you, good rim or leave hand him from
And the down of by my trub.'

ANGEL

"""
