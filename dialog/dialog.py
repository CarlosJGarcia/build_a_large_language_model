# Intento de díalogo entre el modelo que he creado y Llama
# Reinach 13/Jun/2026

import os
import re
import sys
import json
import torch
import tiktoken
import urllib.request
from rich.console import Console

import_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../05'))
if import_dir not in sys.path:
    sys.path.insert(0, import_dir)
from p05_04 import GPTModel
from p05_10 import generate, text_to_token_ids, token_ids_to_text

import_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../07'))
if import_dir not in sys.path:
    sys.path.insert(0, import_dir)
from p07_02 import format_input


def query_model(
    prompt, 
    model="llama3", 
    url="http://localhost:11434/api/chat"
):
    data = {             #1
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {         #2
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }

    payload = json.dumps(data).encode("utf-8")    #3
    request = urllib.request.Request(                       #4
        url,                                                #4
        data=payload,                                       #4
        method="POST"                                       #4
    ) #4

    request.add_header("Content-Type", "application/json")   #4

    response_data = ""
    with urllib.request.urlopen(request) as response:   #5
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

ITERACIONES = 5
CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# Colores
SYSTEM = "white"
AMBER = "#FFB000"

# Camabiar indicando el FQDN del servidor ollama 
URL = "http://workstation.fqdn:11434/api/chat"
OLLAMA_MODEL = "llama3"

# ***********
# Main ******
# ************

console = Console()

# Creo un objeto tokenizer  tipo GPT-2
console.print(f"Tokenizer", style=SYSTEM)
tokenizer = tiktoken.get_encoding("gpt2")

# Creo un modelo tipo GPT-2
console.print(f"Model", style=SYSTEM)
model = GPTModel(BASE_CONFIG)

# Cargo los pesos del modelo 
file_name = f"../models/07-06-{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft.pth"
console.print(f"Loading parameters {file_name}", style=SYSTEM, highlight=False)
if  torch.cuda.is_available():
        device = "cuda"
        console.print(f"Using {device}: {torch.cuda.get_device_name(0)}", style=SYSTEM, highlight=False)
else:
        device = "cpu"
        console.print(f"Using {device}: CUDA not available.", style=SYSTEM) 
model_state_dict = torch.load(file_name, map_location=device, weights_only=True)
model.load_state_dict(model_state_dict)

# Put the model in evaluation mode
model.eval()

# Send the model to the GPU
model.to(device)
console.print(f"Model loaded\n", style=SYSTEM)


# -------------------------------------------

# Format the input, convert it to token IDs, pass it through the generate function and decode the output back to text
question = "You are easily irritable writer."
input = "Generate a random short story three lines long and ask me to continue it."

formatted_prompt = format_input({"instruction": question, "input": input})
print(f"Prompt to model: {formatted_prompt}")

#  Tokenize and move to device (GPU/CPU)
token_ids = text_to_token_ids(formatted_prompt, tokenizer).to(device)

# Generate the response
console.print(f"\nModel generating response", style=SYSTEM)
with torch.no_grad(): # Disable gradient calculation for faster inference
    generated_ids = generate(
        model=model,
        idx=token_ids,
        max_new_tokens=2048,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )

# Convert tokens to text
generated_text = token_ids_to_text(generated_ids, tokenizer)

# Clean up the output to show just the model's response
response_text = (
    generated_text[len(formatted_prompt):]
    .replace("### Response:", "")
    .strip()
)

print(f"\nModel response:{response_text}")
print()

# -------------------------------------------
for i in range(ITERACIONES):

    console.print(f"Quering: {URL}", style=SYSTEM, highlight=False)
    ollama_pregunta = response_text

    console.print(f"{ollama_pregunta}", style=AMBER)
    ollama_result = query_model(ollama_pregunta, OLLAMA_MODEL, url=URL)
    console.print(f"{ollama_result}", style=AMBER, highlight=False)

    # -----------------

    formatted_prompt = format_input({"instruction": question, "input": ollama_result})
    print(f"Prompt to model: {formatted_prompt}")

    #  Tokenize and move to device (GPU/CPU)
    token_ids = text_to_token_ids(formatted_prompt, tokenizer).to(device)

    # Generate the response
    console.print(f"\nModel generating response", style=SYSTEM)
    with torch.no_grad(): # Disable gradient calculation for faster inference
        generated_ids = generate(
            model=model,
            idx=token_ids,
            max_new_tokens=2048,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )

    # Convert tokens to text
    generated_text = token_ids_to_text(generated_ids, tokenizer)

    # Clean up the output to show just the model's response
    response_text = (
        generated_text[len(formatted_prompt):]
        .replace("### Response:", "")
        .strip()
    )

    print(f"\nModel response:{response_text}")
    print()
