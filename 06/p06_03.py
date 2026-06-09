import os
import sys

import time
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

# Number of fine-tunning (training) epochs
NUM_EPOCHS = 5

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
                logits = model(input_batch)[:, -1, :]               # Logits of last output token1
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (
                (predicted_labels == target_batch).sum().item()
            )

        else:
            break
    return correct_predictions / num_examples

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]                           # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

# Calculating the classification loss
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:                                        
        # Ensures number of batches doesn’t exceed batches in data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


# evaluate_model() similar to the one used for pre-training
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss


# Fine-tuning (training) the model to classify spam
def train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []    #1
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):                                    #2
        model.train()                                                  #3

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()                                      #4
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()                                            #5
            optimizer.step()                                           #6
            examples_seen += input_batch.shape[0]                      #7
            global_step += 1
                                                                       #8
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )
                                                                         #9
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)

        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen



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

# Compute the initial loss for each data set
with torch.no_grad():                 
    train_loss = calc_loss_loader(
        train_loader, model, device, num_batches=5
    )
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)
print(f"Training loss: {train_loss:.3f}")
print(f"Validation loss: {val_loss:.3f}")
print(f"Test loss: {test_loss:.3f}")

# Training
start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

console.print(f"\nTraining", style="gold1")
if torch.cuda.is_available():
        console.print(f"Using GPU: {torch.cuda.get_device_name(0)}", style="bright_blue", highlight=False)
else:
        console.print(f"CUDA not available. Using CPU.", style="gold1") 
model.to(device)
train_losses, val_losses, train_accs, val_accs, examples_seen = \
    train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=NUM_EPOCHS, eval_freq=50,
        eval_iter=5
    )

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")