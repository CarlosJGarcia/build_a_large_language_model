import tiktoken
from rich.console import Console
from datasets import load_dataset

DATASET_ID = "Skylion007/openwebtext"
SPLIT = "train"  
NUM_PROC = 22                # Number of parallel process = number or CPU threads used. The Z640 Xeon has 14 cores = 28 threads
                             # Pero, como openwebtext está dividido en solo 12 ficheros (shards), solo habrá 12 procesos corriendo en la CPU, uno en cada thread

# Define a wrapper function for the map. This function encodes the text and returns the IDs. encode_ordinary is slightly faster than encode
def process_text(example):
    ids = tokenizer.encode_ordinary(example['text']) 
    return {'ids': ids, 'len': len(ids)}

# Creo un objeto tokenizer tipo GPT2
console = Console()
console.print(f"\nTokenizer - Tiktoken GPT2", style="gold1")
tokenizer = tiktoken.get_encoding("gpt2")

# Load the dataset
dataset = load_dataset(DATASET_ID, split=SPLIT)

# Fire up all the Xeon Cores!
console.print(f"\nTokenizing", style="gold1")
tokenized_dataset = dataset.map(
    process_text,
    remove_columns=['text'], # Drop the raw text to save memory
    desc="Tokenizing OpenWebText",
    num_proc=NUM_PROC,
    load_from_cache_file=False  # Forces the CPU to do the work again 
)
