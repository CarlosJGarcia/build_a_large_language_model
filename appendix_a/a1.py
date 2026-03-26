import torch

print(f"Torch version: {torch.__version__}")
print()

# Tensor 0D - Escalar - Definido  a partir de un entero python
tensor0d = torch.tensor(1)
print(f"tensor0d type: {type(tensor0d)}")
print(f"tensor0d dtype: {tensor0d.dtype}")
print(f"tensor0d dimensions: {tensor0d.ndim}")
print()

# Tensor 1D - Vector -  Definido a partir de una lista python
tensor1d = torch.tensor([1, 2, 3])
print(f"tensor1d type: {type(tensor1d)}")
print(f"tensor1d dtype: {tensor1d.dtype}")
print(f"tensor1d dimensions: {tensor1d.ndim}")
print()


# Tensor 2D - Matriz -  Definido a partir de una lista anidada python
# Modela una imagen de 3x3=9 pixels en blanco y negro
tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"tensor2d type: {type(tensor2d)}")
print(f"tensor2d dtype: {tensor2d.dtype}")
print(f"tensor2d dimensions: {tensor2d.ndim}")
print()


# Tensor 3D - Definido a partir de una lista doblemente anidada python
# Modela una imagen de 3x3=9 pixels en color (3 Canales RGB)
tensor3d = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[11, 12, 13], [14, 15, 16], [17, 18, 19]], [[21, 22, 23], [24, 25, 26], [27, 28, 29]]])
print(f"tensor3d type: {type(tensor3d)}")
print(f"tensor3d dtype: {tensor3d.dtype}")
print(f"tensor3d dimensions: {tensor3d.ndim}")
print()


# Tensor 4D - Video corto - Definido a partir de una lista triplemente anidada python
# Modela un vídeo con 4 frames, donde cada frame tiene 3x3=9 pixels en color (3 Canales RGB)
# Shape (batch=4, depth=3, height=3, width=3)
tensor4d = torch.tensor([
    [[[1, 2, 3], [4, 5, 6], [7, 8, 9]],           # batch 1 - starts at 1
     [[11, 12, 13], [14, 15, 16], [17, 18, 19]],
     [[21, 22, 23], [24, 25, 26], [27, 28, 29]]],
    [[[101, 102, 103], [104, 105, 106], [107, 108, 109]],   # batch 2 - starts at 11
     [[111, 112, 113], [114, 115, 116], [117, 118, 118]],
     [[121, 122, 123], [124, 125, 126], [127, 128, 129]]],
    [[[21, 22, 23], [24, 25, 26], [27, 28, 29]],   # batch 3 - starts at 21
     [[30, 31, 32], [33, 34, 35], [36, 37, 38]],
     [[39, 40, 41], [42, 43, 44], [45, 46, 47]]],
    [[[31, 32, 33], [34, 35, 36], [37, 38, 39]],   # batch 4 - starts at 31
     [[40, 41, 42], [43, 44, 45], [46, 47, 48]],
     [[49, 50, 51], [52, 53, 54], [55, 56, 57]]]
])
print(f"tensor4d type: {type(tensor4d)}")
print(f"tensor4d dtype: {tensor4d.dtype}")
print(f"tensor4d dimensions: {tensor4d.ndim}")
print(f"tensor4d shape: {tensor4d.shape}")
print()

print("Fin del programa.")

