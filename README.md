# BabyGPT: A Lightweight Transformer for Character-Level Modeling

A minimalist, PyTorch-based implementation of a **Generative Pretrained Transformer (GPT)**. This project is optimized for training on small-to-medium datasets (like the Works of Shakespeare) and is specifically tuned for high performance on **Apple Silicon (M4)** via the Metal Performance Shaders (MPS) backend.

---

## 📋 Table of Contents
1. [Overview](#overview)
2. [Setup and Tokenization](#setup-and-tokenization)
3. [Training](#training)
4. [Inference & Sampling](#inference--sampling)
5. [Technical Architecture](#technical-architecture)
6. [Hardware Performance](#hardware-performance)

---

## 1. Overview
BabyGPT is a character-level language model. Unlike models that predict words, this model predicts the next character in a sequence. This allows for a very small vocabulary size (65 characters for Shakespeare) and makes it a perfect sandbox for understanding the inner workings of Transformers.

---

## 2. Setup and Tokenization
Before training, the raw text must be converted into a binary format that the model can ingest. We use a simple character-level encoder.

**Run the preparation script:**
```bash
python data/shakespeare/prepare.py
```

**Expected Console Output:**
```bash
length of dataset in characters: 1,115,394
all the unique characters:
 !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
vocab size: 65
train has 1,003,854 tokens
val has 111,540 tokens
```

## 3. Training
To launch the training process, use the provided Shakespeare configuration. The script handles checkpointing, logging, and performance metrics automatically.
```bash
python train.py config/train_shakespeare.py
```

**Key Training Parameters:**
* eval_interval: Frequency of validation loss checks.
* log_interval: Frequency of training loss updates.
* max_iters: Total training steps (5,000 is recommended for initial runs).

## 4. Inference & Sampling
Once a checkpoint is saved in the out/ directory, you can generate original text using the sampling script.
```bash
python sample.py
```

**Sample Output (0.8M Parameters, Iteration 5000):**
```text
First Sove Margaret:
Well met you, look your clouse natural your majesty
To me forswear to a sound thing and look:
Upon, therefore, he man of heavy more
Than we have from thy news blood at mercy,
I am this broad our fears and look troubled
That envious mean and brother; and like at it smell,
Than the marsh word his creation, this enters

PETRUS:
So,
```

## 5. Technical Architecture
The model follows a decoder-only Transformer architecture with the following hyperparameters:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **n_layer** | 4 | Number of transformer blocks (layers) |
| **n_head** | 4 | Number of self-attention heads |
| **n_embd** | 128 | Dimensionality of embeddings |
| **block_size** | 256 | Context window (sequence length) |
| **vocab_size** | 65 | Total unique characters |

## 6. Hardware Performance
**This project highlights the efficiency of the Apple M4 chip. By leveraging the mps device in PyTorch, we achieve significant throughput.**
* Iteration Speed: ~0.03 seconds per step.
* Total Parameters: 0.80 Million.
* Precision: float32 (MPS default).
* Estimated Completion: ~2.5 minutes for a full 5,000 iteration training run.

## License
MIT

Happy Training! If the model starts insulting you in iambic pentameter, it's working.
