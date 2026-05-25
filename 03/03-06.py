# Implementing causal attention

import torch
import torch.nn as nn

# Causal attention class
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)            #1
        self.register_buffer(
           'mask',
           torch.triu(torch.ones(context_length, context_length),
           diagonal=1)
        )             #2

    def forward(self, x):
        b, num_tokens, d_in = x.shape                   #3
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)   
        attn_scores.masked_fill_(                    #4
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) 
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec


inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

d_in = inputs.shape[1]
d_out = 2   

# To simulate batches consisting of more than one input, we duplicate the input text example
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)
print()              

torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)
print()


# By the end of the book, you will have a script where you type a prompt like "The quick brown fox", and you will watch your very own model generate the next words.
# It is one of the most satisfying moments in programming. You will know exactly how the text was converted to numbers, how the attention mechanism mixed those numbers
# and how the final linear layer predicted the most mathematically probable next word. 
# It strips away all the "AI magic" and proves that, what is running under the hood is just your hard work and matrix multiplication


# Sebastian Raschka designed this book so that anyone with a standard laptop could follow along and train a small, educational model, 
# specifically, a GPT-2 style model with about 124 M (million) parameters.

# Because you are using a workstation with a RTX 5090 32GB and a Ryzen 9 9950X, following the book's default code will not push your hardware to 98%.
# Your RTX 5090 is a titan. If you run the book's default training loop, your GPU will likely sit at around 10% to 20% utilization. 
# It will chew through the math so fast that it will basically be yawning. The training process that takes a normal reader a few hours might take
# your workstation a matter of minutes.

# If your goal is to hit 98% utilization, hear those fans spin up to maximum and watch your Task Manager graph hit 98%, you have the perfect hardware to do it
# you will just need to scale up the book's parameters once you finish the tutorial!

# Increase the Batch Size: Instead of processing 4 or 8 sequences of text at a time, crank it up to 64 or 128. 
# This floods the GPU memory with data, ensuring the 5090's thousands of CUDA cores are all being used simultaneously.

# Increase the Model Size: Change the model configuration to build the 355 million or even the 1.5 billion parameter version of the model. 
# You do this by increasing the number of layers (n_layers), the embedding dimension (emb_dim), and the number of attention heads (n_heads).

# Expand the Context Window: The book might default to a context window of 256 or 1024 tokens. Try pushing it to 2048 or 4096.

# You have built a workstation that is genuinely capable of training meaningful, moderately-sized language models entirely offline. 
# Stick with the book to build the foundation, and once that first line of text generates, you can start turning the dials up to see what that 5090 can really do.