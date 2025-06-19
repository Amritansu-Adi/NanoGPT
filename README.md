# nanoGPPT

A minimal GPT language model implementation in PyTorch, inspired by Andrej Karpathy's nanoGPT. This project trains a character-level transformer on the Tiny Shakespeare dataset.

## Features

- Minimal, readable code
- Character-level language modeling
- Transformer architecture (multi-head self-attention, feedforward, layer norm)
- Supports training on CPU, GPU, or TPU (with PyTorch/XLA)
- Easily extensible for experimentation

## Requirements

- Python 3.8+
- PyTorch
- (Optional) CUDA for GPU support
- (Optional) [PyTorch/XLA](https://github.com/pytorch/xla) for TPU support

## Usage

### 1. Install dependencies

```bash
pip install torch
# For TPU support:
pip install torch-xla
```

### 2. Download the dataset

The script will automatically download [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) if not present.

### 3. Train the model

**On CPU or GPU:**
```bash
python GPT.py
```

**On TPU (e.g., Google Colab or GCP TPU VM):**
```bash
python GPT.py
```
> Make sure you have selected a TPU runtime and installed `torch-xla`.

### 4. Generate text

After training, the script will generate sample text from the trained model and print it to the console.

## Configuration

You can adjust hyperparameters such as `batch_size`, `block_size`, `n_embd`, `n_head`, `n_layer`, and `max_iters` at the top of `GPT.py`.

## File Structure

```
GPT.py         # Main training and generation script
input.txt      # Training data (Tiny Shakespeare)
README.md      # This