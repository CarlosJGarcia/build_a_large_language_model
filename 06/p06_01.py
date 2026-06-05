# Descarga el fichero SMSSpamCollection.tsv
# Modificado a partir del original, que no funciona detrás de FW, ya que este sustituye el certificado del site por el propio
# Reinach 05/Jun/2026

import os
import zipfile
import urllib.request
import ssl  # 1. Import the ssl module
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
ZIP_PATH = "sms_spam_collection.zip"
EXTRACTED_PATH = "../data/raw/sms_spam_collection"
data_file_path = Path(EXTRACTED_PATH) / "SMSSpamCollection.tsv"


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # 2. Create an unverified SSL context
    ssl_context = ssl._create_unverified_context()

    # 3. Pass the context parameter to urlopen
    with urllib.request.urlopen(url, context=ssl_context) as response:                                
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    # Unzips the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:                              
        zip_ref.extractall(extracted_path)

    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    # Adds .tsv file extension
    os.rename(original_file_path, data_file_path)                                
    print(f"File downloaded and saved as {data_file_path}")

    # Once I have extracted the data file, delete the downloaded .zip file 
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"Deleted temporary file: {zip_path}")

download_and_unzip_spam_data(URL, ZIP_PATH, EXTRACTED_PATH, data_file_path)
print()