import time
import tiktoken
from rich.console import Console
from datasets import load_dataset

DATASET_ID = "Skylion007/openwebtext"  # The canonical Openwebtext dataset in Hugging Face
SPLIT = "train"  
NUM_PROC = 12                          # Número de procesos en paralelo = número de ficheros (shards) en el dataset. Cada uno va a un thread de la CPU

   
# Define a wrapper function for the map. This function encodes the text and returns the IDs. encode_ordinary is slightly faster than encode
def process_text(example):
    ids = tokenizer.encode_ordinary(example['text']) 
    return {'ids': ids, 'len': len(ids)}

# Creo un objeto tokenizer tipo GPT2
console = Console()
console.print(f"\nTokenizer created (Tiktoken GPT2)", style="gold1")
tokenizer = tiktoken.get_encoding("gpt2")

# Load dataset, Training split
console.print(f"\nLoad dataset ({DATASET_ID})", style="gold1")
dataset = load_dataset(DATASET_ID, split=SPLIT, num_proc=NUM_PROC)

# Fire up all the Xeon Cores!
console.print(f"\nTokenizing", style="gold1")
start_time = time.time()
tokenized_dataset = dataset.map(
    process_text,
    remove_columns=['text'], # Drop the raw text to save memory
    desc="Tokenizing OpenWebText",
    num_proc=NUM_PROC,
    load_from_cache_file=False  # Forces the CPU to do the work again 
)

# Muestra el tiempo
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
console.print(f"\nTokenizing openwebtext completed in {execution_time_minutes:.2f} minutes.", style="gold1", highlight=False)
print()