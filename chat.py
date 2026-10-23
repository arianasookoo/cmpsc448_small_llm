"""
Step 4: load the trained model and type prompts into it.

usage: python chat.py
"""
import torch
from model import GPTLanguageModel

max_new_tokens = 400
temperature = 0.8 # lower = safer/more repetitive, higher = more random
top_k = 40
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ckpt = torch.load('ckpt.pt', map_location=device)
chars = ckpt['chars']
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s if c in stoi] # drop characters the model never saw
decode = lambda l: ''.join([itos[i] for i in l])

model = GPTLanguageModel(**ckpt['config']).to(device)
model.load_state_dict(ckpt['model'])
model.eval()
print(f'loaded model ({sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters)')

while True:
    try:
        prompt = input('\nPrompt (Enter = empty prompt, Ctrl+C = quit): ')
    except (KeyboardInterrupt, EOFError):
        break
    ids = encode(prompt) or encode('\n')
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    print('-' * 60)
    print(decode(out[0].tolist()))
    print('-' * 60)
