"""
Step 2 + 3: build a small GPT from scratch and pretrain it with next-token
prediction on data.txt. Saves the trained model to ckpt.pt.

usage: python train.py            (full run)
       python train.py 200        (quick test with 200 steps)
"""
import sys
import time
import torch
from model import GPTLanguageModel

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# character-level tokenizer: every unique character is one token
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
print(f'dataset: {len(data):,} tokens, vocab size {vocab_size}')

def get_batch(split):
    # inputs x and targets y, where y is x shifted one character to the right
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

config = dict(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd,
              n_head=n_head, n_layer=n_layer, dropout=dropout)
model = GPTLanguageModel(**config).to(device)
print(f'{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters, training on {device}')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

t0 = time.time()
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, {time.time()-t0:.0f}s elapsed")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save({'model': model.state_dict(), 'config': config, 'chars': chars}, 'ckpt.pt')
print('saved ckpt.pt')
