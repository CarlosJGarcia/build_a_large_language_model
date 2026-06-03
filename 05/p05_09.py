# conda activate deep_learning_cuda
# Aparecen unos mensajes de TensorFlow C++ y XLA, pero desde Python no se pueden quitar

from gpt_download import download_and_load_gpt2

# Descarga
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

print()
print("Settings:", settings)
print("Parameter dictionary keys:", params.keys())

print()
print(params["wte"])
print("Token embedding weight tensor dimensions:", params["wte"].shape)
print()

