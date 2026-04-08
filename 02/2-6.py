# A dataset for batched imputs and targets
# BPE (byte pair enconding) tokenization - Tiktoken library
# Basel 08/Apr/2026

import torch
from torch.utils.data import Dataset, DataLoader

import tiktoken
from importlib.metadata import version


# Compruebo versión de torch
print(f"Torch version: {torch.__version__}")

# Compruebo versión de tiktoker
print(f"Tiktoken version: {version('tiktoken')}")

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_lenght, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_lenght, stride):
            input_chunk = token_ids[i:i + max_lenght + 1]
            target_chunk = token_ids[i + 1: i + max_lenght + 1]
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


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)




