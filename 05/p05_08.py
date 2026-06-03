import urllib.request

url = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch05/"
    "01_main-chapter-code/gpt_download.py"
)
filename = url.split('/')[-1]

print("Filename:", filename)
urllib.request.urlretrieve(url, filename)
print(f"{filename} descargado")