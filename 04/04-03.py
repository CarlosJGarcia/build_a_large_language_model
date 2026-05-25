# Implementing the GELU activation function
# GELU: Gaussian Error Linear Unit
# ReLU: Rectified Linear Unit

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

IMAGE_FILE = "activation_functions.png"

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