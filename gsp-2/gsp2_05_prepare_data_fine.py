# GSP-2 Fine-tuning (gsp2_04_instruct_tune.py)

# Loads the trained model
# Fine-tunes the LLM to follow instructions and to be able to perform other conversational tasks
# SFT: Supervised Fine-Tuning
# Uses the Alpaca dataset (50.000 entradas) filtered to remove the ones (35%) with two different input fields

# Reinach 18/Jun/2026

import json
from datasets import load_dataset

ALPACA_FILTERED_PATH = "../data/processed/alpaca_filtered.json"

# Load official Alpaca dataset from Hugging Face
print()
print("Downloading/Loading dataset from Hugging Face")
dataset = load_dataset("tatsu-lab/alpaca", split="train")

print(f"Original dataset size: {len(dataset)} entries.")

# Filter out entries that have extra context in the 'input' field
print("Filtering out entries with extra inputs.")
filtered_dataset = dataset.filter(lambda x: x["input"].strip() == "")

print(f"Filtered dataset size: {len(filtered_dataset)} entries.")

# Convert the Hugging Face Dataset format into a standard list of dictionaries
filtered_data_list = [row for row in filtered_dataset]

# Save locally as a standard, indented JSON file
print(f"Saving to {ALPACA_FILTERED_PATH}")

with open(ALPACA_FILTERED_PATH, "w", encoding="utf-8") as f:
    json.dump(filtered_data_list, f, indent=4)
print("Done.")
print()


