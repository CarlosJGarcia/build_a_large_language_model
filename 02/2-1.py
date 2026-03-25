# Old way - manual, risky
"""
f = open("file.txt", "r")
raw_text = f.read()
f.close()  # ← easy to forget! and won't run if an error occurs above
"""

# New way, with 'Context Manager' (with ... as)
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of character:", len(raw_text))
print(raw_text[:99])
