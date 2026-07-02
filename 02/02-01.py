# Tokenizer - Conceptos básicos

import re                        
from rich.console import Console

# Libro "The Verdict" short story, public domain and permitted for LLM training
BOOK_PATH = "../data/raw/the-verdict.txt"

# Clase Tokenizer v1
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


# Clase Tokenizer v2
# Si una palabra del texto no existe en el Vocabulario, le asigna el token <|unk|> (que debe estar definido en el Vocabulario
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int            #1
                        else "<|unk|>" for item in preprocessed]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)    #2
        return text



# Cargo el libro como una cadena de texto (variable raw_text)
with open(BOOK_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()

console = Console()
console.print(f"\nFile {BOOK_PATH} loaded", style="gold1", highlight=False)
print("Total number of characters:", len(raw_text))
print(raw_text[:99])
print()


# Tokenizador manual v1
text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)                              # Divide el string en una lista de palabras que incluyen los signos de puntuación
console.print(f"Tokenizer v1", style="gold1")
print("Text: ", text)
print("Result: ", result)
print()


# Tokenizador manual v2
console.print(f"Tokenizer v2", style="gold1")
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)             # Divide el string en una lista de palabras que incluyen los signos de puntuación
result = [item.strip() for item in result if item.strip()]     # Limpia la lista, quita palabras vacías y espacios delante/detrás
print("Text: ", text)
print("Result: ", result)
print()


# Aplico el tokenizador manual v2 al texto del libro
result = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
result = [item.strip() for item in result if item.strip()]
console.print(f"Tokenizer v2 aplicado al libro", style="gold1")
print(f"Número de tokens en el libro: {len(result)}")
print("Result: ", result[:15])


# Lista de tokens, ordenados alfabéticamente
all_tokens = sorted(set(result))
vocab_size = len(all_tokens)
print(f"Vocabulary size: {vocab_size} tokens")

# A partir de la lista de palabras, creo el vocabulario, una lista de palabras y signos únicos donde cada uno tiene un id 
# Vocabulary: En el conexto de IA, conjunto de palabras únicas (tokens), con un número único (entero 0, 1, 2 ...) asociado
# Dictionary: La estructura Python que se utiliza para guardar el Vocabulario. Es una lista de pares id-contenido
# Muestro los 15 primeros tokens en el diccionario
vocab = {token:integer for integer,token in enumerate(all_tokens)}
print("(Token, ID)")
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 15:
        break
print()


# Añado dos tokens: <|endoftext|> para separar textos diferentes y <|unk|> para identificar palabras que no estén en el vocabulario
console.print(f"Tokenizer v2 aplicado al libro + <|endoftext|> + <|junk|>", style="gold1", highlight=False)
all_tokens = sorted(list(set(result)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
vocab_size = len(all_tokens)
print(f"Vocabulary size: {vocab_size} tokens")

# Muestro los primeros 5 items del vocabulario
print("(Token, ID)")
items = list(vocab.items())
for i in range(5):
    print(items[i])
print("...")

# Muestro los últimos 5 items del vocabulario
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)
print()



# SimpleTokenizerV1
text = "It's the last he painted, you know, Mrs. Gisburn said with pardonable pride."
tokenizer = SimpleTokenizerV1(vocab)

console.print("Simple Tokenizer v1.", style="gold1")
ids = tokenizer.encode(text)
print("Texto: ", text)
print("Token IDs: ", ids)  
print("Decodificado: ", tokenizer.decode(ids))
print()


text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
tokenizer = SimpleTokenizerV2(vocab)
console.print("Simple Tokenizer v2.", style="gold1")
ids = tokenizer.encode(text)
print("Texto: ", text)
print("Token IDs: ", ids)
print("Decodificado: ", tokenizer.decode(ids))
print()