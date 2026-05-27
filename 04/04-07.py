# Generating text

import torch
import tiktoken
from rich.console import Console

def generate_text_simple(model, idx,                           #1
                         max_new_tokens, context_size): 
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]                      #2
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]                              #3
        probas = torch.softmax(logits, dim=-1)                 #4
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  #5
        idx = torch.cat((idx, idx_next), dim=1)                #6

    return idx

#1 idx is a (batch, n_tokens) array of indices in the current context.
#2 Crops current context if it exceeds the supported context size, e.g., if LLM supports only 5 tokens, and the context size is 10, then only the last 5 tokens are used as context
#3 Focuses only on the last time step, so that (batch, n_token, vocab_size) becomes (batch, vocab_size)
#4 probas has shape (batch, vocab_size).
#5 idx_next has shape (batch, 1).
#6 Appends sampled index to the running sequence, where idx has shape (batch, n_tokens+1)

# Creo un objeto tokenizer tipo GPT2
console = Console()
console.print(f"\nTokenizer - Tiktoken GPT2", style="gold1")
tokenizer = tiktoken.get_encoding("gpt2")

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)

# Disables dropout since we have not trained the model yet
encoded_tensor = torch.tensor(encoded).unsqueeze(0)    
print("encoded_tensor.shape:", encoded_tensor.shape)


