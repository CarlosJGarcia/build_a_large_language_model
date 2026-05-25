import torch
import torch.nn as nn

NUM_HEADS = 12
CONTEXT_LENGHT = 1024

# Input and output embedding size (no se usan, pero las pongo por referencia)
d_in = 768
d_out = 768

# Python dictionary. Configuration of the small GPT-2 model
GPT_CONFIG_124M = {
    "vocab_size": 50257,               # Vocabulary size
    "context_length": CONTEXT_LENGHT,  # Context length
    "emb_dim": 768,                    # Embedding dimension
    "n_heads": NUM_HEADS,              # Number of attention heads
    "n_layers": 12,                    # Number of layers
    "drop_rate": 0.1,                  # Dropout rate
    "qkv_bias": False                  # Query-Key-Value bias
}


# GPT model architecture class
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(               #1
            *[DummyTransformerBlock(cfg)               #1
              for _ in range(cfg["n_layers"])]         #1
        )                                              #1
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])     #2
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class DummyTransformerBlock(nn.Module):                    #3
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):                                  #4
        return x

class DummyLayerNorm(nn.Module):                           #5
    def __init__(self, normalized_shape, eps=1e-5):        #6
        super().__init__()

    def forward(self, x):
        return x

#1 Uses a placeholder for TransformerBlock
#2 Uses a placeholder for LayerNorm
#3 A simple placeholder class that will be replaced by a real TransformerBlock later
#4 This block does nothing and just returns its input.
#5 A simple placeholder class that will be replaced by a real LayerNorm later
#6 The parameters here are just to mimic the LayerNorm interface.


print("Todo bien!")
print()