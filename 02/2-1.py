import re

# Old way - manual, risky
"""
f = open("file.txt", "r")
raw_text = f.read()
f.close()  # ← easy to forget! and won't run if an error occurs above
"""

# New way, with 'Context Manager' (with ... as)
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    print("Reading file")
    raw_text = f.read()

print("Total number of characters:", len(raw_text))
print(raw_text[:99])
print()

text = "Hello, world. This, is a test."
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)
print()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])
