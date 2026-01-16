import sys
import torch

print()
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Torch version:", torch.__version__)
print("Apple Silicon acceleration:", torch.backends.mps.is_available())
print("CUDA enabled:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Compute Platform:", torch.version.cuda)
print()
