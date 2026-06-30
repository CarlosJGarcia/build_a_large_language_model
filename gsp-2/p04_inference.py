# GSP-2 Inference test (p04_inference.py)

# Loads the trained model
# Runs a test inference to check that the model works
# A properly trained base model is a neural network that has been taught to speak english from scratch

# Lugano 17/Jun/2026


import torch
import tiktoken
from rich.console import Console
from p02_gpt_model import GPTModel, GPT_CONFIG_355M
from p03_train import generate_and_print_sample, generate_and_print, MODEL_PATH


console = Console()
console.print(f"\nTokenizer created (Tiktoken GPT2)", style="gold1")
tokenizer = tiktoken.get_encoding("gpt2")

console.print(f"\nInstance TransformerBlock", style="gold1")
model = GPTModel(GPT_CONFIG_355M)

# Set Up Processing Engine
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    console.print(f"\nLoading model on GPU: {torch.cuda.get_device_name(0)}", style="bright_blue", highlight=False)
else:
    console.print(f"\nCUDA not available. Loading model on CPU.", style="gold1")          

# Loads the model
print(f"Loading model {MODEL_PATH}")
model.to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
torch.manual_seed(123) 
console.print(f"Model loaded\n", style="gold1")

# Solo pruebas en inglés porque OpenWebText es 99% texto inglés
test_prompts = [
        "Every effort moves you",
        "The sky is",
        "Alicia and Carlos"
    ]

for n, prompt in enumerate(test_prompts, start=1):
    console.print(f"{n}. {prompt}", style="white", highlight=False)
    generate_and_print(model, tokenizer, device, prompt)
    print()

# Pause 0
key = input("Press ENTER to exit.")
print()
