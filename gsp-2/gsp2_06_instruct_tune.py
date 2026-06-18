# GSP-2 Fine-tuning (gsp2_04_instruct_tune.py)

# Loads the trained model
# Fine-tunes the LLM to follow instructions and to be able to perform other conversational tasks
# SFT: Supervised Fine-Tuning
# Uses the Alpaca dataset (50.000 entradas) filtered to remove the ones (35%) with two different input fields

# Reinach 18/Jun/2026

import json
from datasets import load_dataset

# Load official Alpaca dataset from Hugging Face
print("Downloading/Loading dataset from Hugging Face...")
dataset = load_dataset("tatsu-lab/alpaca", split="train")

print(f"Original dataset size: {len(dataset)} entries.")

# Filter out entries that have extra context in the 'input' field
print("Filtering out entries with extra inputs...")
filtered_dataset = dataset.filter(lambda x: x["input"].strip() == "")

print(f"Filtered dataset size: {len(filtered_dataset)} entries.")

# Convert the Hugging Face Dataset format into a standard list of dictionaries
filtered_data_list = [row for row in filtered_dataset]

"""
# 4. Save locally as a standard, indented JSON file
output_filename = "alpaca_data_filtered.json"
print(f"Saving to {output_filename}...")

with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(filtered_data_list, f, indent=4)
"""
print("Done. Ready to train.")


