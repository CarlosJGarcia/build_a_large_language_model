import tiktoken
from datasets import load_dataset

# 1. Load the tokenizer and the dataset
enc = tiktoken.get_encoding("gpt2")
dataset = load_dataset("openwebtext", split="train")

# 2. Define a wrapper function for the map
def process_text(example):
    # Encode the text and return the IDs
    ids = enc.encode_ordinary(example['text']) # encode_ordinary is slightly faster than encode!
    return {'ids': ids, 'len': len(ids)}

# 3. Fire up all the Xeon Cores!
# We set num_proc=14 to map exactly to your CPU's physical cores
tokenized_dataset = dataset.map(
    process_text,
    remove_columns=['text'], # Drop the raw text to save memory
    desc="Tokenizing OpenWebText",
    num_proc=14, 
)
