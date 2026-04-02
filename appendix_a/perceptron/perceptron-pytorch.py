# perceptron-pytorch.ipynb
# 02/Apr/2026
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print()
print("1. Carga de datos")
dataset = np.load("dataset.npz")
train_points      = dataset["train_points"]
train_labels      = dataset["train_labels"]
validation_points = dataset["validation_points"]
validation_labels = dataset["validation_labels"]
test_points       = dataset["test_points"]
test_labels       = dataset["test_labels"]

# Convert numpy arrays to PyTorch tensors
train_points      = torch.tensor(train_points,      dtype=torch.float32).to(device)
train_labels      = torch.tensor(train_labels,      dtype=torch.float32).to(device)
validation_points = torch.tensor(validation_points, dtype=torch.float32).to(device)
validation_labels = torch.tensor(validation_labels, dtype=torch.float32).to(device)
test_points       = torch.tensor(test_points,       dtype=torch.float32).to(device)
test_labels       = torch.tensor(test_labels,       dtype=torch.float32).to(device)

# DataLoaders — PyTorch's equivalent of Keras automatic batching
train_dataset = TensorDataset(train_points, train_labels)
val_dataset   = TensorDataset(validation_points, validation_labels)
test_dataset  = TensorDataset(test_points, test_labels)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=128, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False)

print()
print("2. Definición del modelo y preparación de los datos")
# Sequential model equivalent: nn.Sequential
# input: 2 features → output: 1 neuron with sigmoid
network = nn.Sequential(
    nn.Linear(2, 1),   # equivalent to Dense(1) with input shape (2,)
    nn.Sigmoid()        # equivalent to activation='sigmoid'
).to(device)

# Loss function and optimizer — explicit in PyTorch, implicit in Keras
criterion = nn.BCELoss()                              # binary_crossentropy
optimizer = torch.optim.RMSprop(network.parameters()) # rmsprop

print()
print("3. Resumen del modelo")
print(network)
total_params = sum(p.numel() for p in network.parameters())
print(f"Total parameters: {total_params}")

print()
print("4. Model training")

NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):

    # --- Training phase ---
    network.train()
    train_loss     = 0.0
    train_correct  = 0
    train_total    = 0

    for batch_points, batch_labels in train_loader:
        optimizer.zero_grad()                              # reset gradients
        outputs = network(batch_points).squeeze()          # forward pass
        loss    = criterion(outputs, batch_labels)         # compute loss
        loss.backward()                                    # backpropagation
        optimizer.step()                                   # update weights

        train_loss    += loss.item() * batch_points.size(0)
        predicted      = (outputs >= 0.5).float()
        train_correct += (predicted == batch_labels).sum().item()
        train_total   += batch_labels.size(0)

    avg_train_loss = train_loss    / train_total
    avg_train_acc  = train_correct / train_total

    # --- Validation phase ---
    network.eval()
    val_loss     = 0.0
    val_correct  = 0
    val_total    = 0

    with torch.no_grad():                                  # no gradients needed
        for batch_points, batch_labels in val_loader:
            outputs  = network(batch_points).squeeze()
            loss     = criterion(outputs, batch_labels)

            val_loss    += loss.item() * batch_points.size(0)
            predicted    = (outputs >= 0.5).float()
            val_correct += (predicted == batch_labels).sum().item()
            val_total   += batch_labels.size(0)

    avg_val_loss = val_loss    / val_total
    avg_val_acc  = val_correct / val_total

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - "
          f"loss: {avg_train_loss:.4f} - accuracy: {avg_train_acc:.4f} - "
          f"val_loss: {avg_val_loss:.4f} - val_accuracy: {avg_val_acc:.4f}")

print()
print("5. Evaluación con datos de test que el modelo no ha visto")
network.eval()
test_loss    = 0.0
test_correct = 0
test_total   = 0

with torch.no_grad():
    for batch_points, batch_labels in test_loader:
        outputs   = network(batch_points).squeeze()
        loss      = criterion(outputs, batch_labels)

        test_loss    += loss.item() * batch_points.size(0)
        predicted     = (outputs >= 0.5).float()
        test_correct += (predicted == batch_labels).sum().item()
        test_total   += batch_labels.size(0)

avg_test_loss = test_loss    / test_total
avg_test_acc  = test_correct / test_total

print(f"Test loss: {avg_test_loss:.4f} - Test accuracy: {avg_test_acc:.4f}")
