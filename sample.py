import torch
from model import GPTConfig, GPT
import pickle

device = 'mps'
out_dir = 'out'

temp_setting = 0.7
tokens_to_generate = 500

# Load model
ckpt_path = f'{out_dir}/ckpt.pt'
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# Load metadata for decoding
with open(f'data/shakespeare/meta.pkl', 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Generate!
start = "\n"
x = torch.tensor(encode(start), dtype=torch.long, device=device)[None, ...]
y = model.generate(x, max_new_tokens=tokens_to_generate, temperature=temp_setting)
print(decode(y[0].tolist()))