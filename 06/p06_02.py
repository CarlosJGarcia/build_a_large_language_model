import pandas as pd

from p06_01 import EXTRACTED_PATH, data_file_path

print(f"Reading with Pandas {data_file_path}\n")
df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])

# Muestra el dataframe
print("Dataframe:")
print(df)
print()