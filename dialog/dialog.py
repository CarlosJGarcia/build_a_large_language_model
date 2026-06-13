# Fine-tuninig (training) the model to follow instructions
# Basel 12/Jun/2026


import os
import re
import sys
# import json
# import time
import torch

# import tiktoken
# from tqdm import tqdm
# from functools import partial
from rich.console import Console
# from torch.utils.data import DataLoader


# Point Python to the folder one level up, named '05' so "from p05_04, from p05_10 and from gpt_download" work
import_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../05'))
if import_dir not in sys.path:
    sys.path.insert(0, import_dir)
# from p05_04 import GPTModel, calc_loss_loader, train_model_simple, IMAGE_FILE, plot_losses
from p05_04 import GPTModel

"""
from p05_10 import load_weights_into_gpt, generate, text_to_token_ids, token_ids_to_text
from gpt_download import download_and_load_gpt2

from p07_02 import format_input, InstructionDataset, custom_collate_fn
"""

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

"""
MODEL_DIR = "../models/gpt2"

FILE_PATH = "../data/raw/instruction-data.json"
RESPONSES_FILE_PATH = "../data/raw/instruction-data-with-response.json"
NUM_WORKERS = 0                                     # Increase if the OS supports Python parallel processes 
BATCH_SIZE = 8
NUM_EPOCHS = 2

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

settings, params = download_and_load_gpt2(
    model_size=model_size, 
    models_dir=MODEL_DIR
)

"""
console = Console()


# console.print(f"\nLoading model", style="gold1")


model = GPTModel(BASE_CONFIG)

"""
load_weights_into_gpt(model, params)
model.eval()
"""
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("Device:", device)

"""
if torch.cuda.is_available():
        console.print(f"Loading model on GPU: {torch.cuda.get_device_name(0)}", style="bright_blue", highlight=False)
else:
        console.print(f"CUDA not available. Loading model on CPU.", style="gold1") 
model.to(device)
torch.manual_seed(123)


# Open the Dataset file and load its contents into the 'data' variable
with open(FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
console.print(f"\nDataset loaded\n", style="gold1")

# Divide the dataset into training, validation, and test
train_portion = int(len(data) * 0.85)                     # 85% Training
test_portion = int(len(data) * 0.1)                       # 10% Test
train_data = data[:train_portion]
val_data = data[train_portion + test_portion:]
test_data = data[train_portion:train_portion + test_portion]



# Creo un objeto tokenizer tipo GPT2
console.print(f"Tokenizer - Tiktoken GPT2\n", style="gold1")
tokenizer = tiktoken.get_encoding("gpt2")



customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)
train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=NUM_WORKERS
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=NUM_WORKERS
)

# Calculate the initial loss (Epoch 0). Besc practice in Deep Learning. Diagnostic before launching the training
print("Epoch 0 checks before fine-tunning:")
with torch.no_grad():
    train_loss = calc_loss_loader(
        train_loader, model, device, num_batches=5
    )
    val_loss = calc_loss_loader(
        val_loader, model, device, num_batches=5
)
print("- Training loss:", train_loss)
print("- Validation loss:", val_loss)
print()

console.print(f"Training", style="gold1")
start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=NUM_EPOCHS, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# Generate and Metrics Output
console.print(f"\nPlot", style="gold1")
epochs_tensor = torch.linspace(0, NUM_EPOCHS, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

console.print(f"\nTest Questions / Answers", style="gold1")
for entry in test_data[:3]:              #1
    input_text = format_input(entry)
    token_ids = generate(                #2
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)

    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nModel response:\n>> {response_text.strip()}")
    print("-------------------------------------")
#1 Iterates over the first three test set samples
#2 Uses the generate function imported in section 7.5

# Generating test set responses
console.print(f"\nGenerating test set answers", style="gold1")
for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)

    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    test_data[i]["model_response"] = response_text

# Crea un nuevo fichero .json y guarda ahí el dataset
with open(RESPONSES_FILE_PATH, "w") as file:
    json.dump(test_data, file, indent=4)         # Indentado para que se lea mejor
print("Test entry[0]:", test_data[0])

# Guarda el modelo fine-tuned
console.print(f"\nGuardo el modelo", style="gold1")
"""

file_name = f"../models/07-06-{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft.pth"

# torch.save(model.state_dict(), file_name)
# print(f"Model saved as {file_name}\n")



# Cargo el modelo de nuevo
print(f"Loading model {file_name}")
if torch.cuda.is_available():
        console.print(f"Using GPU: {torch.cuda.get_device_name(0)}", style="bright_blue", highlight=False)
else:
        console.print(f"CUDA not available. Using CPU.", style="gold1") 

model_state_dict = torch.load(file_name, map_location=device, weights_only=True)
model.load_state_dict(model_state_dict)

"""
model.eval()
print("Model loaded\n")
"""