# The GPT (Generative Pretrained Transformer) model architecture implementation

import torch
import tiktoken
import torch.nn as nn
from rich.console import Console


CONTEXT_LENGTH = 1024
VOCAB_SIZE = 50257                     # GPT-2 vocabulary size

OUTPUT_DIM_SMALL = 768                 # Small GPT-2 embedding vector size 
NUM_HEADS_SMALL = 12                   # Small GPT-2 attention heads

OUTPUT_DIM_MEDIUM = 1024               # Medium GPT-2 embedding vector size
NUM_HEADS_MEDIUM = 16                  # Medium GPT-2 attention heads


# Python dictionary. Configuration of the small GPT-2 model
GPT_CONFIG_124M = {
    "vocab_size": VOCAB_SIZE,          # Vocabulary size
    "context_length": CONTEXT_LENGTH,  # Context length
    "emb_dim": OUTPUT_DIM_SMALL,       # Embedding dimension
    "n_heads": NUM_HEADS_SMALL,        # Number of attention heads
    "n_layers": 12,                    # Number of layers
    "drop_rate": 0.1,                  # Dropout rate
    "qkv_bias": False                  # Query-Key-Value bias
}

# Python dictionary. Configuration of the medium GPT-2 model (~355M params)
GPT_CONFIG_355M = {
    "vocab_size": VOCAB_SIZE,          # Vocabulary size
    "context_length": CONTEXT_LENGTH,  # Context length
    "emb_dim": OUTPUT_DIM_MEDIUM,      # 1024 (Hidden size / Vector size)
    "n_heads": NUM_HEADS_MEDIUM,       # 16 (Attention heads)
    "n_layers": 24,                    # 24 (Double the depth of small)
    "drop_rate": 0.1,                  # Dropout rate remains the same
    "qkv_bias": False                  # Query-Key-Value bias
}


# Multi-head attention with weight splits
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


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


# A feed forward neural network module
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


# Defino una clase GELU a partir de (herencia) la clase Module de la librería torch.nn (Torch Neural Network)
class GELU(nn.Module):
    # Contructor de la clase GELU que llama al constructor de la superclase (Module)
    def __init__(self):
        super().__init__()

    # Override (sobreescritura de un método de la superclase, con un nuevo método) - código para ejecución en GPU
    def forward(self, x):
        # Added device=x.device
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device)) * (x + 0.044715 * torch.pow(x, 3))
        ))

# The Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
 #1
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut      #2

        shortcut = x         #3
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut      #4
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
 #1
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    


# =========
# Execution
# =========
if __name__ == "__main__":  

    # Data
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    # Creo un objeto tokenizer tipo GPT2
    console = Console()
    console.print(f"\nTokenizer - Tiktoken GPT2", style="gold1")
    tokenizer = tiktoken.get_encoding("gpt2")

    batch = []
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print("Tokenized batch data:", batch)



    # Instantiate a GPTModel and feed it some sample data:
    console = Console()
    console.print(f"\nInstance TransformerBlock", style="gold1")

    # Configuración GPU RTX
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()

    # Intantiate the model in the GPU
    torch.manual_seed(123)
    #model = GPTModel(GPT_CONFIG_124M).to(device)
    model = GPTModel(GPT_CONFIG_355M).to(device)

    # Move the data batch to the GPU
    batch = batch.to(device)


    out = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape:", out.shape)
    print(out)

    # Calcula y muestra el número de parámetros del modelo
    console.print(f"\nNumber of parameters", style="gold1")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,} = {total_params//1000000:,} M")
    print()

    print("Token embedding layer shape:", model.tok_emb.weight.shape)
    print("Output layer shape:", model.out_head.weight.shape)

    total_params_gpt2 = (
        total_params - sum(p.numel()
        for p in model.out_head.parameters())
    )

    print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,} = {total_params_gpt2//1000000:,} M")


    # Calcula la memoria que ocupa el modelo
    console.print(f"\nModel memory usage", style="gold1")
    total_size_bytes = total_params * 4       #1
    total_size_mb = total_size_bytes / (1024 * 1024)     #2
    print(f"Total size of the model: {total_size_mb:.2f} MB")
    print()