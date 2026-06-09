import os
import sys

import torch
import tiktoken
import pandas as pd
from pathlib import Path
from rich.console import Console
from torch.utils.data import Dataset, DataLoader

# Point Python to the folder one level up, named '05' so "from p05_04 import GPTModel" works
import_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../05'))
if import_dir not in sys.path:
    sys.path.insert(0, import_dir)
from gpt_download import download_and_load_gpt2
from p05_04 import GPTModel, generate_text_simple, text_to_token_ids, token_ids_to_text
from p05_10 import load_weights_into_gpt


# Dictionary that lists the differences between the different GPT model sizes
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# Configuration Constants
CONTEXT_LENGTH = 1024

# GPT-2 124M Base Configuration
CHOOSE_MODEL = "gpt2-small (124M)"
BASE_CONFIG = {
    "vocab_size": 50257,               # Vocabulary size
    "context_length": CONTEXT_LENGTH,  # Context length
    "drop_rate": 0.0,                  # Dropout rate
    "qkv_bias": True                  # Query-Key-Value bias
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
INPUT_PROMPT = "Every effort moves"

NUM_WORKERS = 0      #Ensures compatibility with most computers
BATCH_SIZE = 8

DATA_PATH = "../data/processed/sms_spam_collection"
train_file_path = Path(DATA_PATH) / "train.csv"
validation_file_path = Path(DATA_PATH) / "validation.csv"
test_file_path = Path(DATA_PATH) / "test.csv"


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None,
                 pad_token_id=50256):
        self.data = pd.read_csv(csv_file)                          #1 
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length                           #2
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]                                                      #3
        self.encoded_texts = [
            encoded_text + [pad_token_id] * 
            (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]
#1 Pretokenizes texts
#2 Truncates sequences if they are longer than max_length
#3 Pads sequences to the longest sequence

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


# Calculating the classification accuracy
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)


            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]              # Logits of last output token1
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (
                (predicted_labels == target_batch).sum().item()
            )

        else:
            break
    return correct_predictions / num_examples


console = Console()
console.print(f"\nTokenizer - Tiktoken GPT2", style="gold1")
tokenizer = tiktoken.get_encoding("gpt2")

print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
print()

train_dataset = SpamDataset(csv_file=train_file_path, max_length=None, tokenizer=tokenizer)
print("Number of tokens in the longest sequence:", train_dataset.max_length)

val_dataset = SpamDataset(csv_file=validation_file_path, max_length=train_dataset.max_length, tokenizer=tokenizer)
test_dataset = SpamDataset(csv_file=test_file_path, max_length=train_dataset.max_length, tokenizer=tokenizer)

console.print(f"\nDataLoader", style="gold1")
torch.manual_seed(123)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, drop_last=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, drop_last=False)

# Iterate over the training loader and then print the tensor dimensions of the last batch
for input_batch, target_batch in train_loader:
    pass
print("Input batch dimensions:", input_batch.shape)
print("Label batch dimensions", target_batch.shape)
print()

# Print the total number of batches in each dataset:
print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")
print()


MODEL_DIR = "../models/gpt2"

# Descarga en la carpeta "gpt2"
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir=MODEL_DIR)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()
print()

# Ensure that the model generates coherent text
console.print(f"Check that the model generates coherent text", style="gold1")
text_1 = "Every effort moves you"
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))
print()

# Before fine-tunning, check whether the model already classifies spam messages
console.print(f"Chack if, before fine-tunning, can classify spam", style="gold1")
text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_2, tokenizer),
    max_new_tokens=23,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))

#Show the model's architecture
console.print(f"\nModel architecture", style="gold1")
print(model)

# Freeze the model: make all layers nontrainable (requires_grad = False)
for param in model.parameters():
    param.requires_grad = False

# Add a classification layer
torch.manual_seed(123)
num_classes = 2
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"], 
    out_features=num_classes
)

# Make the LayerNorm and last transformer block trainable (requires_grad = True)
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True

# Ensure that the model keeps being able to generates coherent text
console.print(f"\nCheck that the model keeps being able to generate coherent text", style="gold1")
text_1 = "Every effort moves you"
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))
print()

inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)
print("Inputs:", inputs)
print("Inputs dimensions:", inputs.shape)    # shape: (batch_size, num_tokens)

with torch.no_grad():
    outputs = model(inputs)
print("Outputs:\n", outputs)
print("Outputs dimensions:", outputs.shape)
print("Last output token:", outputs[:, -1, :])

probas = torch.softmax(outputs[:, -1, :], dim=-1)
label = torch.argmax(probas)
print("Class label (probas):", label.item())

logits = outputs[:, -1, :]
label = torch.argmax(logits)
print("Class label (logits):", label.item())

# Determine the classification accuracies across various datasets estimated from 10 batches for efficiency:
console.print(f"\nDetermine the classification accuracies", style="gold1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
        console.print(f"Loading model on GPU: {torch.cuda.get_device_name(0)}", style="bright_blue", highlight=False)
else:
        console.print(f"CUDA not available. Loading model on CPU.", style="gold1") 
model.to(device)

torch.manual_seed(123)
train_accuracy = calc_accuracy_loader(
    train_loader, model, device, num_batches=10
)
val_accuracy = calc_accuracy_loader(
    val_loader, model, device, num_batches=10
)
test_accuracy = calc_accuracy_loader(
    test_loader, model, device, num_batches=10
)

print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")