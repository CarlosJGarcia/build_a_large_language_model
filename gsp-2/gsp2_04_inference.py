# GSP-2 Inference test (gsp2_04_inference.py)

# Loads the trained model
# Runs a test inference to check that the model works

# Reinach 14/Jun/2026

from gsp2_02_gpt_model import GPTModel, GPT_CONFIG_355M
from gsp2_03_train import generate_and_print_sample, MODEL_PATH

import torch
from rich.console import Console

console = Console()
console.print(f"\nTokenizer created (Tiktoken GPT2)", style="gold1")
tokenizer = tiktoken.get_encoding("gpt2")

console.print(f"\nInstance TransformerBlock", style="gold1")
model = GPTModel(GPT_CONFIG_355M)

# Set Up Processing Engine
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    console.print(f"\nLoading model on GPU: {torch.cuda.get_device_name(0)}", style="bright_blue")
else:
    console.print(f"\nCUDA not available. Loading model on CPU.", style="gold1")          

# Carga el modelo
print(f"\nLoading model {MODEL_PATH}")
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
key = input("Press ENTER to exit.")