import torch
from functools import partial
from p07_02 import custom_collate_fn

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print()
print("Device:", device)

customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)
print("Todo bien")
print()