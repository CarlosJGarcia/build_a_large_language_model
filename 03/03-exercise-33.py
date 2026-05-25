
# Implementing multi-head attention with weight splits

# Dimensions:
# 12 attention heads
# Input and output embedding with 768 dimensions
# Context length of 1,024 tokens. 


import torch
import torch.nn as nn
from rich.console import Console

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, 
                 context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads    #1
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)    #2
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)         #3
        queries = self.W_query(x)    #3
        values = self.W_value(x)     #3

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)       #4
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)  
        queries = queries.view(                                             
            b, num_tokens, self.num_heads, self.head_dim                    
        )                                                                   

        keys = keys.transpose(1, 2)          #5
        queries = queries.transpose(1, 2)    #5
        values = values.transpose(1, 2)      #5

        attn_scores = queries @ keys.transpose(2, 3)   #6
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]    #7

        attn_scores.masked_fill_(mask_bool, -torch.inf)     #8

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)   #9
        #10
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        context_vec = self.out_proj(context_vec)    #11
        return context_vec

NUM_HEADS = 12
CONTEXT_LENGHT = 1024


console = Console()
console.print(f"\nMulti-head attention with GPT-2/BERT base dimensions", style="gold1")
torch.manual_seed(123)

# Input and output embedding size
d_in = 768
d_out = 768

batch_size = 2 # Simulate a batch of 2 sequences

# En lugar de poner los datos a mano, uso generación aleatorioa de los valores
batch = torch.randn(batch_size, CONTEXT_LENGHT, d_in)
print(f"Input batch shape: {batch.shape}")

# Instantiate the model
mha = MultiHeadAttention(
    d_in=d_in, 
    d_out=d_out, 
    context_length=CONTEXT_LENGHT, 
    dropout=0.0, 
    num_heads=NUM_HEADS
)

# Run the forward pass
context_vecs = mha(batch)
print("Output context_vecs shape:", context_vecs.shape)
print()