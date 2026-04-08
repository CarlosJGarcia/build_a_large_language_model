# A dataset for batched imputs and targets
# BPE (byte pair enconding) tokenization - Tiktoken library
# Basel 08/Apr/2026

import torch
from torch.utils.data import Dataset, DataLoader

import tiktoken
from importlib.metadata import version

# Compruebo versión de torch
print(f"Torch version: {torch.__version__}")


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_lenght, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)




# Compruebo versión de tiktoker
print(f"tiktoken version: {version('tiktoken')}")





# Creo un objeto tokenizer tipo GPT2
print("Creando objeto tokenizer - Tiktoker GPT2")
tokenizer = tiktoken.get_encoding("gpt2")
print()





