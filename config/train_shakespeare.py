# Train a tiny GPT on Shakespeare
out_dir = 'out-shakespeare'
eval_interval = 250
eval_iters = 200
log_interval = 10

# Data
dataset = 'shakespeare'
gradient_accumulation_steps = 1
batch_size = 16
block_size = 256 # context of 256 characters

# Tiny Model
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.2

# Training
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99 # stable for small models

# System
compile = False # PyTorch compile is still buggy on Mac MPS