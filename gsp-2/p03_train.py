# # GSP-2 Create the model and train it (gsp2_03_train.py)

# Loads the OpenWebText dataset from Hugging Face, using HF Datasets library
# Tokenized the datasets with Tiktoken
# Saves the data to disk

# Reinach 14/Jun/2026


# Pretrain the LLM using 30 books from project gutemberg 
# 05-04.py

# What the Model does during the training epochs
# When you stopped the first run at Epoch 1, the model was still in "kindergarten." Over the next 9 epochs, it underwent a massive conceptual shift
# Epoch 1 (What you see in Graph 1): The model is just learning basic token mechanics. It figures out that the letter "t" is often followed by "h" and "e", and that common words like "the", "and", and "of" appear frequently. It has zero concept of how a full sentence is structured.
#Epochs 2–5: The model begins mastering local syntax. It learns parts of speech—that nouns follow adjectives, verbs follow subjects, and quotation marks need to close.
#Epochs 6–10 (What you see in Graph 2): The model adapts to the macro-style of your 30 Project Gutenberg books. It begins tracking long-range context (remembering the subject from 50 tokens back) and heavily mimics the 19th-century vocabulary, formatting, and pacing of the novels.


import time
import torch
import wandb
import tiktoken
import matplotlib.pyplot as plt
from rich.console import Console
from datasets import load_from_disk
from matplotlib.ticker import MaxNLocator
from config import OPENWEBTEXT_TOKENIZED_PATH
from torch.utils.data import DataLoader, IterableDataset

# Import the architecture and configurations from gsp2_02_gpt_model.py
# Small model with 124M parameters. Does not work on RTX 3060, works in 5060 16 GB, reducing the batch size from 8 to 4
# Training a medium model 355M parameters requires more VRAM, 16 GB is not enough.
from p02_gpt_model import GPTModel, GPT_CONFIG_355M

# Configuration Constants
# OPENWEBTEXT_TOKENIZED_PATH = "../data/processed/openwebtext_tokenized"
IMAGE_FILE = "training_validation_losses_base.png"
MODEL_PATH = "../models/gsp-2/gsp2_355m_base.pth"

# Training hyperparameters
NUM_EPOCHS = 1                   # 1 Epoch for large corpus training efficiency and because OpenWebText has more tokens than needed
BATCH_SIZE = 8                   # Batch_size = 8 for better use of the RTX 3060 (12GB VRAM) with a 1024 context length
LEARNING_RATE = 0.0004
EARLY_STOPPING_PATIENCE = 100

# ==========================================
# 1. DATA PROCESSING & UTILITIES
# ==========================================

class OpenWebTextIterableDataset(IterableDataset):
    def __init__(self, hf_dataset, max_length):
        self.dataset = hf_dataset
        self.max_length = max_length

    def __iter__(self):
        buffer = []
        for example in self.dataset:
            buffer.extend(example['ids'])
            # Yield chunks of text equal to max_length + 1 to form input and target pairs
            while len(buffer) >= self.max_length + 1:
                chunk = buffer[:self.max_length + 1]
                buffer = buffer[self.max_length:]  
                
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                target_ids = torch.tensor(chunk[1:], dtype=torch.long)
                
                yield input_ids, target_ids

# Simple text generation function using argmax
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

# Advanced text generation function using Temperature and Top-K/Top-P Sampling
def generate_text(model, idx, max_new_tokens, context_size): 
    
    # Apply Temperature (higher = more creative/random, lower = more deterministic)
    temperature = 1.0
    
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]                      
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]                              
        logits = logits / temperature

        # Top-K filtering (keep only the top 50 highest probability choices)
        top_k = 50
        v, ix = torch.topk(logits, top_k)
        logits[logits < v[..., [-1]]] = -float('Inf')

        # Sample from the remaining distribution instead of taking the max
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)                
    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)    
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)                
    return tokenizer.decode(flat.tolist())


# ==========================================
# 2. TRAINING & EVALUATION FUNCTIONS
# ==========================================

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
    batches_processed = 0
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if num_batches is not None and i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()                              
        batches_processed += 1
        
    if batches_processed == 0:
        return float("nan")
    return total_loss / batches_processed                               

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  
    with torch.no_grad():                                       
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

# Simple generation and result printing function using simple text generation (argmax)
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


# Advanced generation and result printing function using the text generation function with Temperature and Top-K/Top-P Sampling
def generate_and_print(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text(model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))      
    model.train()


class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience  # How many evals to wait
        self.delta = delta        # Min change to qualify as improvement
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    
    # Initialize Early Stopping 
    stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, delta=0.001)
    
    train_losses, val_losses, track_tokens_seen = [], [], []    
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):                             
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  
            
            # Use bfloat16 mixed precision to reduce VRAM from ~15GB to ~8GB
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                
            loss.backward()                                     
            optimizer.step()                                    
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:                    
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)

                # Check for early stopping
                stopper(val_loss)
                if stopper.early_stop:
                    print("Early stopping triggered. Model has plateaued.")
                    return train_losses, val_losses, track_tokens_seen

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}, Tokens seen: {tokens_seen:,}")
                
                # Weights & Biases
                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "global_step": global_step,
                    "tokens_seen": tokens_seen,
                    "epoch": epoch + 1
                })

        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, filename):
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
    plt.savefig(filename, dpi=300) 
    print(f"\nPlot saved successfully as {filename}")
    plt.show()


# =========
# Execution
# =========

if __name__ == "__main__":
    # --- Step A: Quick Toy Data Smoke Test ---
    inputs = torch.tensor([[16833, 3626, 6100], [40, 1107, 588]])   
    targets = torch.tensor([[3626, 6100, 345], [1107, 588, 11311]])  

    console = Console()

    # Weights & Biases
    run = wandb.init(entity="cgarciams-carlos", project="GSP-2",
    config={
        "learning_rate": LEARNING_RATE,
        "architecture": "GPT-2",
        "dataset": "OpenWebText",
        "epochs": NUM_EPOCHS,
    })
    console.print(f"\nWeights & Biases initialized", style="gold1")

    
    tokenizer = tiktoken.get_encoding("gpt2")
    console.print(f"\nTokenizer created (Tiktoken GPT2)", style="gold1")

    console.print(f"\nInstance TransformerBlock", style="gold1")
    torch.manual_seed(123)
    
    # Initialize Small GPT-2 Model directly
    model = GPTModel(GPT_CONFIG_355M)

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
    print(f"Initial Baseline Loss: {loss:.4f}")


    # --- Step B: Load Multi-Book Dataset ---
    console.print(f"\nLoading tokenized dataset ({OPENWEBTEXT_TOKENIZED_PATH})", style="gold1", highlight=False)
    
    try:
        hf_dataset = load_from_disk(OPENWEBTEXT_TOKENIZED_PATH)
    except Exception as e:
        raise FileNotFoundError(f"Could not load dataset from '{OPENWEBTEXT_TOKENIZED_PATH}'. Did you run gsp2_01_prepare_data.py?")

    print(f"Total Dataset Documents: {len(hf_dataset):,}")


    # --- Step C: Create Scaled Data Loaders ---
    train_ratio = 0.90
    split_dataset = hf_dataset.train_test_split(test_size=1-train_ratio, seed=123)
    train_data = split_dataset['train']
    val_data = split_dataset['test']

    torch.manual_seed(123)
    context_length = GPT_CONFIG_355M["context_length"]

    train_dataset = OpenWebTextIterableDataset(train_data, max_length=context_length)
    val_dataset = OpenWebTextIterableDataset(val_data, max_length=context_length)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


    # --- Step D: Set Up Processing Engine ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        console.print(f"\nTraining on GPU: {torch.cuda.get_device_name(0)}", style="bright_blue", highlight=False)
    else:
        console.print(f"\nCUDA not available. Training on CPU.", style="gold1")          
    model.to(device)                                                                       

    with torch.no_grad():                                                                  
        # Using subset of batches to evaluate initial loss quickly
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=10)         
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=10)
    print("Initial Untrained Training loss:", train_loss)
    print("Initial Untrained Validation loss:", val_loss)


    # --- Step E: Run Pretraining Cycle. Initialize the AdamW optimizer
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
    
    start_time = time.time()
    # eval_freq raised to 500 considering OpenWebText's massive scale
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=NUM_EPOCHS, eval_freq=500, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    # Muestra el tiempo
    end_time = time.time()
    execution_time_hours = (end_time - start_time) / 3600
    execution_time_days = execution_time_hours / 24
    console.print(f"Training completed in {execution_time_hours:.2f} hours ({execution_time_days:.2f} days).\n", style="gold1", highlight=False)

    # Generate and Metrics Output ---
    epochs_tensor = torch.linspace(0, NUM_EPOCHS, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, IMAGE_FILE)

    # Guarda el modelo
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved as {MODEL_PATH}")
    print()

    # Weights & Biases
    run.finish()

    