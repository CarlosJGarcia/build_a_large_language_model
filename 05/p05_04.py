# Pretrain the LLM using 30 books from project gutemberg 
# Reinach 01/Jun/2026

# What the Model does during the training epochs:
# When you stopped the first run at Epoch 1, the model was still in "kindergarten." Over the next 9 epochs, it undergoes a massive conceptual shift
# Epoch 1: The model is just learning basic token mechanics. It figures out that the letter "t" is often followed by "h" and "e", and that common words like "the", "and", and "of" appear frequently. It has zero concept of how a full sentence is structured.
# Epochs 2–5: The model begins mastering local syntax. It learns parts of speech, that nouns follow adjectives, verbs follow subjects, and quotation marks need to close.
# Epochs 6–10: The model adapts to the macro-style of your 30 Project Gutenberg books. It begins tracking long-range context (remembering the subject from 50 tokens back) and heavily mimics the 19th-century vocabulary, formatting, and pacing of the novels.

import os
import glob
import torch
import tiktoken
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from rich.console import Console
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Configuration Constants
NUM_HEADS = 12
CONTEXT_LENGTH = 256

# GPT-2 124M Base Configuration
GPT_CONFIG_124M = {
    "vocab_size": 50257,               # Vocabulary size
    "context_length": CONTEXT_LENGTH,  # Context length
    "emb_dim": 768,                    # Embedding dimension
    "n_heads": NUM_HEADS,              # Number of attention heads
    "n_layers": 12,                    # Number of layers
    "drop_rate": 0.1,                  # Dropout rate
    "qkv_bias": False                  # Query-Key-Value bias
}

# Batch_size up to 4 for efficient GPU processing (4 GB VRAM usage)
# Batch_size = 16 for the RTX 3060 
BATCH_SIZE = 16

# 1 Epoch for large corpus training efficiency
# 10 Epochs better learning result. The ideal number for this dataset (30 books) is between 5 and 15 epochs
NUM_EPOCHS = 10

DATA_DIR = "03_bonus_pretraining_on_gutenberg/gutenberg_preprocessed"
IMAGE_FILE = "training_validation_losses.png"
MODEL_PATH = "../models/gpt_124m/gpt_124m_final.pth"


# =================
# GPT Archictecture
# =================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads    
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)    
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)         
        queries = self.W_query(x)    
        values = self.W_value(x)     

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)       
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)  
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)                                                                   

        keys = keys.transpose(1, 2)          
        queries = queries.transpose(1, 2)    
        values = values.transpose(1, 2)      

        attn_scores = queries @ keys.transpose(2, 3)   
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]    

        attn_scores.masked_fill_(mask_bool, -torch.inf)     

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)   
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)    
        return context_vec


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut      

        shortcut = x         
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut      
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# =============================
# Data processing and utilities
# =============================

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            chunk = token_ids[i : i + max_length + 1]
            input_chunk = chunk[:-1]  
            target_chunk = chunk[1:]  

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader


def generate_text_simple(model, idx, max_new_tokens, context_size): 
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]                      
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]                              
        probas = torch.softmax(logits, dim=-1)                 
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  
        idx = torch.cat((idx, idx_next), dim=1)                
    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)    
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)                
    return tokenizer.decode(flat.tolist())


# =======================
# Training and evaluation
# =======================

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)         
    target_batch = target_batch.to(device)      
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)                             
    else:
        num_batches = min(num_batches, len(data_loader))           
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()                              
        else:
            break
    return total_loss / num_batches                                


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  
    with torch.no_grad():                              
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))      
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, 
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []    
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):                             
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()                               
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()                                     
            optimizer.step()                                    
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:                    
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()                   
    ax2.plot(tokens_seen, train_losses, alpha=0)     
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.savefig(IMAGE_FILE, dpi=300) 
    print(f"\nPlot saved as {IMAGE_FILE}")
    plt.show()


# =========
# Execution
# =========
if __name__ == "__main__":    
    # --- Step A: Quick Toy Data Smoke Test ---
    inputs = torch.tensor([[16833, 3626, 6100], [40, 1107, 588]])   
    targets = torch.tensor([[3626, 6100, 345], [1107, 588, 11311]])  

    console = Console()
    console.print(f"\nTokenizer - Tiktoken GPT2", style="gold1")
    tokenizer = tiktoken.get_encoding("gpt2")

    console.print(f"\nInstance TransformerBlock", style="gold1")
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)

    with torch.no_grad():                      
        logits = model(inputs)
    probas = torch.softmax(logits, dim=-1)     
    print("Softmax shape:", probas.shape)

    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    print("Targets batch 1:", token_ids_to_text(targets[0], tokenizer))
    print("Outputs batch 1:", token_ids_to_text(token_ids[0].flatten(), tokenizer))

    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    console.print(f"Initial Baseline Loss: {loss:.4f}", style="bright_blue")

    # Load Multi-Book Dataset
    console.print(f"\nReading text files from '{DATA_DIR}'", style="gold1")
    raw_text = ""

    file_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    if not file_paths:
        raise FileNotFoundError(f"No text files found in '{DATA_DIR}'. Did you run prepare_dataset.py?")

    for file_path in file_paths:
        print(f" -> Loading: {os.path.basename(file_path)}")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text += f.read() + "\n"

    total_characters = len(raw_text)
    integers = tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})
    total_tokens = len(integers)

    print(f"\nTotal Dataset Characters: {total_characters:,}")
    print(f"Total Dataset Tokens: {total_tokens:,}")

    # Create Scaled Data Loaders
    train_ratio = 0.90
    split_idx = int(train_ratio * len(raw_text))
    train_data = raw_text[:split_idx]
    val_data = raw_text[split_idx:]

    torch.manual_seed(123)
    
  
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=BATCH_SIZE,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True
    )
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=BATCH_SIZE,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False
    )

    # Set Up Processing Engine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        console.print(f"\nTraining on GPU: {torch.cuda.get_device_name(0)}", style="bright_blue")
    else:
        console.print(f"\nCUDA not available. Training on CPU.", style="gold1")          
    
    model.to(device)                                                       

    # For sanity check, calculate initial (model with random weights) training and validation loss with a small number of batches
    with torch.no_grad():
        num_batches = 5    # Small number of batches                                                
        train_loss = calc_loss_loader(train_loader, model, device, num_batches)         
        val_loss = calc_loss_loader(val_loader, model, device, num_batches)
    print("Initial Untrained Training loss:", train_loss)
    print("Initial Untrained Validation loss:", val_loss)


    # Run Pretraining Cycle ---
    console.print(f"\nStarting Core Training Pipeline", style="gold1")
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=NUM_EPOCHS, eval_freq=25, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    # Generate and Metrics Output
    epochs_tensor = torch.linspace(0, NUM_EPOCHS, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    # Guarda el modelo
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved as {MODEL_PATH}")
    print()