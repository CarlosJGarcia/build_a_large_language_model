# Implementing the GELU activation function
# GELU: Gaussian Error Linear Unit
# ReLU: Rectified Linear Unit

import torch
import torch.nn as nn

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
    
print("Todo bien!")
print()