# Es posible que en la primera ejecución de un error OOM (out of memory). El motivo es que para descargar los ficheros con los parámetros de Internet y guardarlos en disco reserva memoria, que no da tiempo a liberar antes de cargar el modelo en memoria
# En una segunda ejecución, cargará los ficheros desde el fichero de disco con los parámetros, que no requiere espacio temporal, con lo cual toda la memoria ya está dispoonible para cargar el modelo.

import os
import sys

# Point Python to the folder one level up, named '05' so "from p05_04, from p05_10 and from gpt_download" work
import_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../05'))
if import_dir not in sys.path:
    sys.path.insert(0, import_dir)
from p05_04 import GPTModel
from p05_10 import load_weights_into_gpt
from gpt_download import download_and_load_gpt2

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"
MODEL_DIR = "../models/gpt2"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

settings, params = download_and_load_gpt2(
    model_size=model_size, 
    models_dir=MODEL_DIR
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

print("Todo bien")
print()