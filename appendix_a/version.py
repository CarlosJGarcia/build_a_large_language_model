import sys
import torch

print()
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Torch version:", torch.__version__)
print("Apple Silicon acceleration:", torch.backends.mps.is_available())
print("CUDA enabled:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Compute Platform.", torch.version.cuda)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print(f"CUDA not available, using CPU.") 
print()