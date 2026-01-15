import torch
import time

# Check MPS availability
if not torch.backends.mps.is_available():
    print("❌ MPS not available on this system.")
    print("Check PyTorch installation and macOS version (needs macOS 12.3+).")
    exit()

# Matrix size (increase for heavier test)
N = 4000

# Generate two random matrices on CPU
a_cpu = torch.randn(N, N)
b_cpu = torch.randn(N, N)

# Move copies to MPS (Apple GPU)
a_mps = a_cpu.to("mps")
b_mps = b_cpu.to("mps")

# --- CPU benchmark ---
start = time.time()
for _ in range(10):
    c_cpu = torch.mm(a_cpu, b_cpu)
torch.cpu.synchronize() if hasattr(torch, 'cpu') else None
end = time.time()
cpu_time = end - start

# --- MPS benchmark ---
torch.mps.synchronize()
start = time.time()
for _ in range(10):
    c_mps = torch.mm(a_mps, b_mps)
torch.mps.synchronize()
end = time.time()
mps_time = end - start

# --- Results ---
print(f"Matrix size: {N}x{N}")
print(f"CPU time: {cpu_time:.3f} s")
print(f"MPS time: {mps_time:.3f} s")
print(f"Speedup: {cpu_time / mps_time:.2f}× faster on MPS")
