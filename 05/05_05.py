from p05_04 import GPTModel
from p05_04 import GPT_CONFIG_124M, MODEL_PATH

import torch
from rich.console import Console

console = Console()
console.print(f"\nInstance TransformerBlock", style="gold1")
model = GPTModel(GPT_CONFIG_124M)

# Set Up Processing Engine
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    console.print(f"\nLoading model on GPU: {torch.cuda.get_device_name(0)}", style="bright_blue")
else:
    console.print(f"\nCUDA not available. Loading model on CPU.", style="gold1")          

# Carga el modelo
print(f"\nLoading model {MODEL_PATH}")
model.to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model loaded\n")

# Pause 0
print()
key = input("Press ENTER to exit.")