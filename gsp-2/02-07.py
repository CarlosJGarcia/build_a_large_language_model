# Creating token embeddings
# Reinach 08/Apr/2026

import torch
from torch.utils.data import Dataset, DataLoader

import tiktoken
from importlib.metadata import version


# Compruebo versión de torch
print(f"Torch version: {torch.__version__}")

# Compruebo versión de tiktoker
print(f"Tiktoken version: {version('tiktoken')}")
print()

# Tensor 2x4
#input_ids = torch.tensor([2, 3, 5, 1])
input_ids = torch.tensor([[2, 3, 5, 1],[1, 2, 3, 4]])
print("input_ids: ", input_ids)
print("input_ids.dtype: ", input_ids.dtype)
print("input_ids.shape: ", input_ids.shape)
print()

# Tensor 1x4
input_ids = torch.tensor([2, 3, 5, 1])
print("input_ids: ", input_ids)
print("input_ids.dtype: ", input_ids.dtype)
print("input_ids.shape: ", input_ids.shape)
print()

# Número de palabras en el vocabulario. El tokenizer solo entiende 6 palabras, en lugar de 50.257
vocab_size = 6

# Cuantos números se utilizan para describir cada item
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print("Muestro la matriz de pesos de la embedding layer")
print("embedding_layer.weight: \n", embedding_layer.weight)