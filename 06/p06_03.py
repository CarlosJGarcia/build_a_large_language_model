import torch
import tiktoken
import pandas as pd
from pathlib import Path
from rich.console import Console
from torch.utils.data import Dataset

DATA_PATH = "../data/processed/sms_spam_collection"
train_file_path = Path(DATA_PATH) / "train.csv"

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