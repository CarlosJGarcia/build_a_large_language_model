import torch
import torch.nn.functional as F     # Convención para que el código sea breve
from torch.autograd import grad

y = torch.tensor([1.0])             # True label
x1 = torch.tensor([1.1])            # Input feature
w1 = torch.tensor([2.2])            # Weight parameter
b = torch.tensor([0.0])             # Bias unit
z = x1 * w1 + b                     #6 Net input
a = torch.sigmoid(z)                # Activation and output
loss = F.binary_cross_entropy(a, y)

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

z = x1 * w1 + b 
a = torch.sigmoid(z)

loss = F.binary_cross_entropy(a, y)

# By default, PyTorch destroys the computation graph after calculating the gradients to free memory. 
# Since we want to reuse this computation graph shortly, we set retain_graph=True so that it stays in memory
grad_L_w1 = grad(loss, w1, retain_graph=True)   
grad_L_b = grad(loss, b, retain_graph=True)

print()
print("grad_L_w1:", grad_L_w1)
print("grad_L_b:", grad_L_b)
print()

loss.backward()
print("w1.grad:", w1.grad)
print("b.grad:", b.grad)
print()