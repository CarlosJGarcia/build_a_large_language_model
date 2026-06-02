from p05_04 import GPTModel
from p05_04 import GPT_CONFIG_124M, MODEL_PATH

import torch
from rich.console import Console


from p05_04 import generate_and_print_sample
import tiktoken


console = Console()
console.print(f"\nTokenizer - Tiktoken GPT2", style="gold1")
tokenizer = tiktoken.get_encoding("gpt2")

console.print(f"Model instantiated from class GPTModel(), size 124M", style="gold1", highlight=False)
model = GPTModel(GPT_CONFIG_124M)

# Set Up Processing Engine
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    console.print(f"Loading model on GPU: {torch.cuda.get_device_name(0)}", style="bright_blue")
else:
    console.print(f"CUDA not available. Loading model on CPU.", style="gold1")          

# Carga el modelo
print(f"Loading model {MODEL_PATH}")
model.to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
print("Model loaded\n")


start_context="Every effort moves you"
generate_and_print_sample(model, tokenizer, device, start_context)

start_context="The sky is"
generate_and_print_sample(model, tokenizer, device, start_context)

start_context="Alicia y Carlos"
generate_and_print_sample(model, tokenizer, device, start_context)

start_context="Le jeune homme"
generate_and_print_sample(model, tokenizer, device, start_context)
print()

# Pause 0
# print()
# key = input("Press ENTER to exit.")