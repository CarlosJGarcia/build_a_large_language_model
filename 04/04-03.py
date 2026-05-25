# Implementing the GELU activation function
# GELU: Gaussian Error Linear Unit
# ReLU: Rectified Linear Unit

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

NUM_HEADS = 12
CONTEXT_LENGHT = 1024
IMAGE_FILE = "activation_functions.png"

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

# Defino una clase GELU a partir de (herencia) la clase Module de la librería Torch Neural Network (torch.nn)
class GELU(nn.Module):
    # Contructor de la clase GELU que llama al constructor de la superclase (Module)
    def __init__(self):
        super().__init__()

    # Override (sobreescritura de un método de la superclase, con un nuevo método)
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
    

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

# Comparison between GELU() and nn.ReLU()
gelu, relu = GELU(), nn.ReLU()

x = torch.linspace(-3, 3, 100)     #1
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
plt.tight_layout()
plt.show()

plt.savefig(IMAGE_FILE, dpi=300) 
print(f"Plot saved successfully as {IMAGE_FILE}")

# Initialize a FeedForward module with a token embedding size of 768 and feed it a batch input with two samples and three tokens each:
ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)          
out = ffn(x)
print(out.shape)
