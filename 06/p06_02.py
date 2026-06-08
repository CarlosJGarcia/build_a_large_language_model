# Procesa el fichero SMSSpamCollection.tsv
# Genera un dataset equilibrado (50% spam, 50% ham) y dividido en proporciones 70/10/20 Training/Validation/Test
# Guarda el resultado en tres ficheros .csv
# Reinach 08/Jun/2026

import pandas as pd
from pathlib import Path
from p06_01 import EXTRACTED_PATH, data_file_path

WRITE_PATH = "../data/processed/sms_spam_collection"
train_file_path = Path(WRITE_PATH) / "train.csv"
validation_file_path = Path(WRITE_PATH) / "validation.csv"
test_file_path = Path(WRITE_PATH) / "test.csv"

def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]     #1
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state=123
    )                                                 #2
    balanced_df = pd.concat([
        ham_subset, df[df["Label"] == "spam"]
    ])                                                #3
    return balanced_df
#1 Counts the instances of “spam”
#2 Randomly samples “ham” instances to match the number of “spam” instances
#3 Combines ham subset with “spam”


def random_split(df, train_frac, validation_frac):

    df = df.sample(
        frac=1, random_state=123
    ).reset_index(drop=True)                                    #1
    train_end = int(len(df) * train_frac)                       #2
    validation_end = train_end + int(len(df) * validation_frac) #3
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    return train_df, validation_df, test_df
#1 Shuffles the entire DataFrame
#2 Calculates split indices
#3 Splits the DataFrame


print(f"Reading with Pandas {data_file_path}\n")
df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])

# Muestra el dataframe
print("Dataframe:")
print(df)
print()

# Veces que aparece ham/spam
print(df["Label"].value_counts())
print()

# Creamos un sub-dataset equilibrado
balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())
print()

# Convierte las etiquetas de clase "ham" y "spam" (string) en enteros 0 y 1
# Es como tokenizar, pero solo hay dos tokens 0 y 1
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

# Test size is implied to be 0.2 as the remainder.
train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)                    

# Save the dataset as CSV (comma-separated value) files so we can reuse it later:
train_df.to_csv(train_file_path, index=None)
print(f"File saved as {train_file_path}")
validation_df.to_csv(validation_file_path, index=None)
print(f"File saved as {validation_file_path}")
test_df.to_csv(test_file_path, index=None)
print(f"File saved as {test_file_path}")
print()
