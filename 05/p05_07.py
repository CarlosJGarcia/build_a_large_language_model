# Top-k sampling
# Reinach 03/Jun/2026

from p05_04 import GPTModel, generate_and_print_sample
from p05_04 import GPT_CONFIG_124M, MODEL_PATH

import torch
import tiktoken
from rich.console import Console

# A text generation function with more diversity
def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):            #1
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:                #2
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        if temperature > 0.0:                  #3
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:    #4
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:              #5
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
#1 The for loop is the same as before: gets logits and only focuses on the last time step.
#2 Filters logits with top_k sampling
#3 Applies temperature scaling
#4 Carries out greedy next-token selection as before when temperature scaling is disabled
#5 Stops generating early if end-of-sequence token is encountered


# from p05_04 import generate_text_simple, text_to_token_ids, token_ids_to_text
# import matplotlib.pyplot as plt

# IMAGE_FILE = "temperature.png"



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
print()

"""

start_context="The sky is"
generate_and_print_sample(model, tokenizer, device, start_context)

start_context="Alicia y Carlos"
generate_and_print_sample(model, tokenizer, device, start_context)

start_context="Le jeune homme"
generate_and_print_sample(model, tokenizer, device, start_context)
print()

# llevamos el modelo de vuelta a la CPU
console.print(f"Loading model on CPU", style="gold1")    
model.to("cpu")
model.eval()

start_context="Every effort moves you"
generate_and_print_sample(model, tokenizer, "cpu", start_context)

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output text:", token_ids_to_text(token_ids, tokenizer))
print()

# A very small vocabulary for illustration purposes:
vocab = { 
    "closer": 0,
    "every": 1, 
    "effort": 2, 
    "forward": 3,
    "inches": 4,
    "moves": 5, 
    "pizza": 6,
    "toward": 7,
    "you": 8,
} 
inverse_vocab = {v: k for k, v in vocab.items()}
"""
next_token_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])

"""
# We convert the logits into probabilities via the softmax function
probas = torch.softmax(next_token_logits, dim=0)

# We obtain the token ID corresponding to the generated token via the argmax function
next_token_id = torch.argmax(probas).item()
print("Inverse vocabulary:", inverse_vocab[next_token_id])

# Replace argmax with the PyTorch multinomial function
torch.manual_seed(123) 
next_token_id = torch.multinomial(probas, num_samples=1).item()
print("Inverse vocabulary:", inverse_vocab[next_token_id])
print()

# A function that repeats this sampling 1000 times
def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item()
             for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

print_sampled_tokens(probas)
print()

# Temperature scaling
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

temperatures = [1, 0.1, 5]                                     #1
scaled_probas = [softmax_with_temperature(next_token_logits, T)
                for T in temperatures]
x = torch.arange(len(vocab))
bar_width = 0.15
fig, ax = plt.subplots(figsize=(5, 3))
for i, T in enumerate(temperatures):
    rects = ax.bar(x + i * bar_width, scaled_probas[i], 
                   bar_width, label=f'Temperature = {T}')
ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()
plt.tight_layout()
plt.savefig(IMAGE_FILE, dpi=300) 
print(f"\nPlot saved as {IMAGE_FILE}")
plt.show()

"""

# Selection of the tokens with the largest logit values
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
print("Top logits:", top_logits)
print("Top positions:", top_pos)
print()

new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],    #1
    input=torch.tensor(float('-inf')),               #2
    other=next_token_logits                          #3
)
#1 Identifies logits less than the minimum in the top 3
#2 Assigns –inf to these lower logits
#3 Retains the original logits for all other tokens
print("New logits:", new_logits)


# Apply the softmax function to turn the new logits into next-token probabilities
topk_probas = torch.softmax(new_logits, dim=0)
print("Next-token probabilities:", topk_probas)
print()

