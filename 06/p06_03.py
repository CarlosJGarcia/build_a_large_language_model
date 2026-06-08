import torch
import tiktoken
import pandas as pd
from pathlib import Path
from rich.console import Console
from torch.utils.data import Dataset, DataLoader

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