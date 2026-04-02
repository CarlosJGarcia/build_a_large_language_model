import os
import numpy as np
import matplotlib.pyplot as plt

# Constants for range
MIN_VAL = -100
MAX_VAL = 100


# Define the filename
# filename = "dataset_coords.npy"


def convierte_a_float32(array):
    return array.astype(np.float32)

# print(f"NumPy version: {np.__version__}")
# print()

# 1. Generate values between 0 and 1
# 2. Multiply by 200 (makes range 0 to 200)
# 3. Subtract 100 (makes range -100 to 100)
train_data = np.random.rand(200, 2).astype(np.float32) * (MAX_VAL - MIN_VAL) + MIN_VAL

# 2. Extract x and y columns
x = train_data[:, 0]
y = train_data[:, 1]

# 3. Define the line: y_line = 2*x - 5
# Logic: If (y < 2*x - 5) → 1 (Right), else 0 (Left)
train_labels = (y < (2 * x - 25)).astype(np.float32)

# (Optional) Save dataset
# np.save(filename, train_data)


# Show the first 5 entries to verify
print("\nFirst 5 (x, y) couples and their labels (0=Left, 1=Right):")
for i in range(5):
    print(f"Point: {train_data[i].tolist()} -> Label: {train_labels[i].item()}")


# Convert labels to boolean masks
mask_0 = train_labels == 0
mask_1 = train_labels == 1

# Plot points with different colors
plt.figure(figsize=(6, 6))
plt.scatter(x[mask_0], y[mask_0], color='blue', s=5, label='Label 0')
plt.scatter(x[mask_1], y[mask_1], color='red', s=5, label='Label 1')

# Optional: plot the decision boundary y = 2x - 5
x_line = np.linspace(-100, 100, 100)
y_line = 2 * x_line - 5
plt.plot(x_line, y_line, color='green', linestyle='--', label='y = 2x - 5')

plt.xlabel("x")
plt.ylabel("y")
plt.title("Dataset Visualization")
plt.legend()
plt.grid(True)



"""
import os
import torch


# Constants for our range
MIN_VAL = -100
MAX_VAL = 100

# Define the filename
filename = "dataset_coords.pt"


def convierte_a_float32(tensor):
    tensor = tensor.to(torch.float32)
    return tensor

print(f"Torch version: {torch.__version__}")
print(f"CUDA is available. {torch.cuda.is_available()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
print()

# 1. Generate values between 0 and 1
# 2. Multiply by 200 (makes range 0 to 200)
# 3. Subtract 100 (makes range -100 to 100)
train_data = torch.rand(60000, 2, dtype=torch.float32) * (MAX_VAL - MIN_VAL) + MIN_VAL


# 2. Extract x and y columns for the calculation
# [:, 0] gets all rows, first column (x)
# [:, 1] gets all rows, second column (y)
x = train_data[:, 0]
y = train_data[:, 1]

# 3. Define the line: y_line = 2*x - 5
# To the left/above means y > 2*x - 5. 
# You requested: Left = 0, Right = 1.
# Logic: If (y < 2*x - 5) is True, it returns 1 (Right), otherwise 0 (Left).
train_labels = (y < (2 * x - 5)).to(torch.float32)


print(f"tensor dimensions: {train_data.ndim}")
print(f"tensor dtype: {train_data.dtype}")
print(f"tensor shape: {train_data.shape}")
print(f"tensor type: {type(train_data)}")

print(f"Features shape: {train_data.shape}")
print(f"Labels shape: {train_labels.shape}")
print(f"Labels dtype: {train_labels.dtype}")

# Show the first 5 entries to verify
print("\nFirst 5 (x, y) couples and their labels (0=Left, 1=Right):")
for i in range(5):
    print(f"Point: {train_data[i].tolist()} -> Label: {train_labels[i].item()}")

# Quick check on the distribution
ones = torch.sum(train_labels)
zeros = 60000 - ones
print(f"\nDistribution - Left (0): {int(zeros)}, Right (1): {int(ones)}")

# --- SAVING ---
# We store them in a dictionary so they stay paired together
print(f"Saving tensors to {filename}...")
data_to_save = {
    'samples': train_data,
    'labels': train_labels
}
torch.save(data_to_save, filename)
print("Save complete.\n")


# --- LOADING ---
if os.path.exists(filename):
    print(f"Loading tensors from {filename}...")
    loaded_checkpoint = torch.load(filename, weights_only=True)

    # Extract the individual tensors
    loaded_data = loaded_checkpoint['samples']
    loaded_labels = loaded_checkpoint['labels']

    print(f"Loaded Data Shape: {loaded_data.shape}")
    print(f"Loaded Labels Shape: {loaded_labels.shape}")

    # Quick sanity check: are they the same as the original?
    is_same = torch.equal(train_data, loaded_data)
    print(f"Verified: Data matches original? {is_same}")
else:
    print("File not found!")
"""
