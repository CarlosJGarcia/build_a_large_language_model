import torch


def convierte_a_float32(tensor):
    tensor = tensor.to(torch.float32)
    return tensor

print(f"Torch version: {torch.__version__}")
print()

# Tensor 0D - Escalar - Definido  a partir de un entero python
tensor0d = torch.tensor(1)
print(f"tensor0d dimensions: {tensor0d.ndim}")
print(f"tensor0d dtype: {tensor0d.dtype}")
print(f"tensor0d shape: {tensor0d.shape}")
print(f"tensor0d type: {type(tensor0d)}")
print()

# Tensor 1D - Vector -  Definido a partir de una lista python
tensor1d = torch.tensor([1, 2, 3])
print(f"tensor1d dimensions: {tensor1d.ndim}")
print(f"tensor1d dtype: {tensor1d.dtype}")
print(f"tensor1d shape: {tensor1d.shape}")
print(f"tensor1d type: {type(tensor1d)}")
print()


# Tensor 2D - Matriz -  Definido a partir de una lista anidada python
# Modela una imagen de 3x3=9 pixels en blanco y negro
tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"tensor2d dimensions: {tensor2d.ndim}")
print(f"tensor2d dtype: {tensor2d.dtype}")
print(f"tensor2d shape: {tensor2d.shape}")
print(f"tensor2d type: {type(tensor2d)}")
print()


# Tensor 3D - Definido a partir de una lista doblemente anidada python
# Modela una imagen de 3x3=9 pixels en color (3 Canales RGB)
tensor3d = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[11, 12, 13], [14, 15, 16], [17, 18, 19]], [[21, 22, 23], [24, 25, 26], [27, 28, 29]]])
print(f"tensor3d dimensions: {tensor3d.ndim}")
print(f"tensor3d dtype: {tensor3d.dtype}")
print(f"tensor3d shape: {tensor3d.shape}")
print(f"tensor3d type: {type(tensor3d)}")
print()


# Tensor 4D - Video corto - Definido a partir de una lista triplemente anidada python
# Modela un vídeo con 4 frames, donde cada frame tiene 3x3=9 pixels en color (3 Canales RGB)
# Shape (batch=4, depth=3, height=3, width=3)
tensor4d = torch.tensor([
    [[[1, 2, 3], [4, 5, 6], [7, 8, 9]],                     # batch 1 
     [[11, 12, 13], [14, 15, 16], [17, 18, 19]],
     [[21, 22, 23], [24, 25, 26], [27, 28, 29]]],
    [[[101, 102, 103], [104, 105, 106], [107, 108, 109]],   # batch 2
     [[111, 112, 113], [114, 115, 116], [117, 118, 119]],
     [[121, 122, 123], [124, 125, 126], [127, 128, 129]]],
    [[[201, 202, 203], [204, 205, 206], [207, 208, 209]],   # batch 3
     [[211, 212, 213], [214, 215, 216], [217, 218, 219]],
     [[221, 222, 223], [224, 225, 226], [227, 228, 229]]],
    [[[301, 302, 303], [304, 305, 306], [307, 308, 309]],   # batch 4
     [[311, 312, 313], [314, 315, 316], [317, 318, 319]],
     [[321, 322, 323], [324, 325, 326], [327, 328, 329]]]
])
print(f"tensor4d dimensions: {tensor4d.ndim}")
print(f"tensor4d dtype: {tensor4d.dtype}")
print(f"tensor4d shape: {tensor4d.shape}")
print(f"tensor4d type: {type(tensor4d)}")
print()


print("Convierto los tensores a float32:")
tensor0d = convierte_a_float32(tensor0d)
tensor1d = convierte_a_float32(tensor1d)
tensor2d = convierte_a_float32(tensor2d)
tensor3d = convierte_a_float32(tensor3d)
tensor4d = convierte_a_float32(tensor4d)
print(f"tensor0d dtype: {tensor0d.dtype}")
print(f"tensor1d dtype: {tensor1d.dtype}")
print(f"tensor2d dtype: {tensor2d.dtype}")
print(f"tensor3d dtype: {tensor3d.dtype}")
print(f"tensor4d dtype: {tensor4d.dtype}")
print()


print("Fin del programa.")

