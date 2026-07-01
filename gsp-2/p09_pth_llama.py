# GSP-2 pth 2 ollama converter (p09_pth_llama.py)
# Converts the GSP-2 model weights in .pth (PyTorch) file to a folder with .safetensors and config.json to ollama can load the model

# Reinach 30/Jun/2026

import os
import json
from safetensors.torch import save_file

import torch
import tiktoken
from rich.console import Console
from p02_gpt_model import GPTModel, GPT_CONFIG_355M, VOCAB_SIZE, NUM_HEADS_MEDIUM, OUTPUT_DIM_MEDIUM
from p03_train import MODEL_PATH

LLAMA_PATH = "../models/gsp-2/gsp2_355m_base_llama"

# Creates the tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
console = Console()
console.print(f"\nTokenizer created (Tiktoken GPT2)", style="gold1")

# Set Up Processing Engine
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    console.print(f"Loading model on GPU: {torch.cuda.get_device_name(0)}", style="bright_blue", highlight=False)
else:
    console.print(f"CUDA not available. Loading model on CPU.", style="gold1")          

# Loads the model
model = GPTModel(GPT_CONFIG_355M)
model.to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
console.print(f"Model {MODEL_PATH} loaded", style="gold1", highlight=False)


# Creates the output directory if it doesn't exist
os.makedirs(LLAMA_PATH, exist_ok=True)

# 1. Translate Weights to Hugging Face GPT-2 Format
original_sd = model.state_dict()
hf_sd = {}

# Map standard embeddings and final layer norm
hf_sd["transformer.wte.weight"] = original_sd["tok_emb.weight"]
hf_sd["transformer.wpe.weight"] = original_sd["pos_emb.weight"]
hf_sd["transformer.ln_f.weight"] = original_sd["final_norm.scale"]
hf_sd["transformer.ln_f.bias"] = original_sd["final_norm.shift"]
hf_sd["lm_head.weight"] = original_sd["out_head.weight"]

# Map Transformer Blocks
for i in range(GPT_CONFIG_355M["n_layers"]):
    # Layer Norms
    hf_sd[f"transformer.h.{i}.ln_1.weight"] = original_sd[f"trf_blocks.{i}.norm1.scale"]
    hf_sd[f"transformer.h.{i}.ln_1.bias"] = original_sd[f"trf_blocks.{i}.norm1.shift"]
    hf_sd[f"transformer.h.{i}.ln_2.weight"] = original_sd[f"trf_blocks.{i}.norm2.scale"]
    hf_sd[f"transformer.h.{i}.ln_2.bias"] = original_sd[f"trf_blocks.{i}.norm2.shift"]
    
    # Attention: Concatenate Q, K, V and Transpose (HF expects [in_features, out_features])
    W_q = original_sd[f"trf_blocks.{i}.att.W_query.weight"]
    W_k = original_sd[f"trf_blocks.{i}.att.W_key.weight"]
    W_v = original_sd[f"trf_blocks.{i}.att.W_value.weight"]
    c_attn_weight = torch.cat([W_q, W_k, W_v], dim=0)
    
    # ADDED .contiguous() HERE
    hf_sd[f"transformer.h.{i}.attn.c_attn.weight"] = c_attn_weight.T.contiguous() 
    
    # Inject zeros for QKV bias 
    hf_sd[f"transformer.h.{i}.attn.c_attn.bias"] = torch.zeros(3 * GPT_CONFIG_355M["emb_dim"], device=device)

    # Attention Output (Transpose) - ADDED .contiguous() HERE
    hf_sd[f"transformer.h.{i}.attn.c_proj.weight"] = original_sd[f"trf_blocks.{i}.att.out_proj.weight"].T.contiguous()
    hf_sd[f"transformer.h.{i}.attn.c_proj.bias"] = original_sd[f"trf_blocks.{i}.att.out_proj.bias"]
    
    # FeedForward (MLP) (Transpose weights) - ADDED .contiguous() HERE
    hf_sd[f"transformer.h.{i}.mlp.c_fc.weight"] = original_sd[f"trf_blocks.{i}.ff.layers.0.weight"].T.contiguous()
    hf_sd[f"transformer.h.{i}.mlp.c_fc.bias"] = original_sd[f"trf_blocks.{i}.ff.layers.0.bias"]
    hf_sd[f"transformer.h.{i}.mlp.c_proj.weight"] = original_sd[f"trf_blocks.{i}.ff.layers.2.weight"].T.contiguous()
    hf_sd[f"transformer.h.{i}.mlp.c_proj.bias"] = original_sd[f"trf_blocks.{i}.ff.layers.2.bias"]

safetensors_file = os.path.join(LLAMA_PATH, "model.safetensors")
save_file(hf_sd, safetensors_file)
console.print(f"Weights saved to: {safetensors_file}", style="bright_green")

# 2. Create a standard Hugging Face config.json
# Without this, conversion tools (like llama.cpp) won't know how to turn this into a GGUF
hf_config = {
    "architectures": ["GPT2LMHeadModel"],
    "model_type": "gpt2",
    "vocab_size": VOCAB_SIZE,
    "n_positions": 1024,
    "n_ctx": 1024,
    "n_embd": OUTPUT_DIM_MEDIUM,
    "n_layer": 24,
    "n_head": NUM_HEADS_MEDIUM,
    "layer_norm_epsilon": 1e-5,
    "bos_token_id": 50256,
    "eos_token_id": 50256
}

config_file = os.path.join(LLAMA_PATH, "config.json")
with open(config_file, "w") as f:
    json.dump(hf_config, f, indent=4)
console.print(f"Configuration saved to: {config_file}", style="bright_green")

# Note: tiktoken is just an algorithm and doesn't have a "save" method for Hugging Face.
# Because you are using standard GPT-2 encoding, GGUF conversion tools will automatically 
# pull the standard GPT-2 tokenizer configuration based on the "model_type" in config.json.
print("Fin del programa.\n")