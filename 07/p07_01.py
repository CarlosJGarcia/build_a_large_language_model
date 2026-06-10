# Descarga el fichero instruction-data.json

import os
import json
import urllib.request

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

FILE_PATH = "../data/raw/instruction-data.json"
URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"

data = download_and_load_file(FILE_PATH, URL)
print(f"\nFile saved as {FILE_PATH}")
print(f"Number of entries: {len(data)}\n")

print(f"Example entry:\n {data[50]}\n")
print(f"Another example entry:\n {data[999]}\n")