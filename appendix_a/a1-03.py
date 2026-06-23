import torch
from rich.console import Console

# Red neuronal de 3 capas y número de entradas y salidas variable
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):    # Number of inputs and outputs as variables so the class can be reused
        super().__init__()

        self.layers = torch.nn.Sequential(

            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),       # The Linear layer takes the number of input and output nodes as arguments.
            torch.nn.ReLU(),                       # Nonlinear activation functions are placed between the hidden layers.

            # 2nd hidden layer
            torch.nn.Linear(30, 20),               # The number of output nodes of one hidden layer has to match the number of inputs of the next layer
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits                              # The outputs of the last layer are called logits
    


# =========
# Execution
# =========
console = Console()

# Instantiate a new neural network
model = NeuralNetwork(50, 3)
print()
print(model)
print()

# Calculate the total number of trainable parameters of the model
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)
print()

# Mostrar los parámetros (los valores entrenables) de la primera capa
layer0parameters = model.layers[0].weight
print("Numer of parameters in layer 0:", layer0parameters.numel())
print("Type of the parameters:", layer0parameters.dtype)
print("List of layer 0 parameters:")
print(layer0parameters)
print()

# Ahora muestro el número de parámetros de cada capa
layer = 0
total = 0
indices = [0, 2, 4] # Lista de los índices de las capas con parámetros. En PyTorch las funciones ReLU tienen su índice, como si fueran una capa
for n in indices:
    weights = model.layers[n].weight.numel()
    biases = model.layers[n].bias.numel()
    print(f"Layer {layer} (index {n}):")
    print(f"  - Weights: {weights}")
    print(f"  - Biases: {biases}")
    print(f"  - Total (sum): {weights + biases}")
    total += (weights + biases)
    layer += 1
print(f"\nTotal numer of parameters (grand total): {total}\n")

# Inicialización
console.print(f"Inicialización", style="gold1")
torch.manual_seed(123)
model = NeuralNetwork(50, 3)
print(model.layers[0].weight)

# Forward pass. 
X = torch.rand((1, 50)) # Genero números aleatorios para la entrada
out = model(X)          # Forward pass = pasar los datos al modelo. PyTorch los multiplica por los pesos, suma los bias, aplica ReLU
console.print(f"\nForward pass result:", style="gold1")
print(out)              # Muestra el resultado
print()