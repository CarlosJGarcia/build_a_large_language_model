# Librería de Regular Expressions para el tokenizer
import re

# 1. Cargo el libro como una cadena de texto (variable raw_text)
# Abro y cargo el fichero de texto (libro) Manera 'antigua'. Old way - manual, risky
"""
f = open("file.txt", "r")
raw_text = f.read()
f.close()  # ← easy to forget! and won't run if an error occurs above
"""

# New way, with 'Python Context Manager' (with ... as). Una especie de try - catch
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    print("Reading file")
    raw_text = f.read()

print("Total number of characters:", len(raw_text))
print(raw_text[:99])
print()


# Tokenizador manual
text = "Hello, world. This, is a test."
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print("Text: ", text)
print("Result: ", result)
print()


# 2. Creo una lista de palabras y símbolos a partir del texto mediante una expresión regular (variable preprocessed)
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(f"Len libro: {len(raw_text)} (número de caracteres en la variable tipo text string)")
print("Tipo de preprocessed: ", type(preprocessed))
print(f"Len preprocessed: {len(preprocessed)} (número de items en la lista = número de tokens en el libro)")
print("Preprocessed: ", preprocessed[:15])
print()

# Creo una lista de tokens, ordenados alfabéticamente
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print(f"Len all_words: {vocab_size} (número de tokens en el diccionario)")


# 3. A partir de la lista de palabras, creo el vocabulario, una lista de palabras y símbolos únicos donde cada uno tiene un id (variable vocab)
# Muestro los 15 primeros tokens en el diccionario
vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 15:
        break
print()



# Añado dos palabras al vocabulario: <|endoftext|> para separar textos diferentes y <|junk|> para identificar con un token palabras que no estén en el dicc.
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}

print(len(vocab.items()))





# Clase Tokenizer
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab            #1
        self.int_to_str = {i:s for s,i in vocab.items()}        #2

    def encode(self, text):         #3
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):         #4
        text = " ".join([self.int_to_str[i] for i in ids]) 

        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)    #5
        return text 


# 4. Creo un objeto tokenizer a partir de la clase SimpleTokenizerV1 y del vocabulario
text = "It's the last he painted, you know, Mrs. Gisburn said with pardonable pride."
tokenizer = SimpleTokenizerV1(vocab)


print("Codificando.")
ids = tokenizer.encode(text)
print("Texto: ", text)
print("Codificado: ", ids)  
print("Decodificado: ", tokenizer.decode(ids))




