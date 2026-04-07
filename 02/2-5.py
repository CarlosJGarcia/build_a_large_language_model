# BPE (byte pair enconding) tokenization
# Kaiseraugst 07/Apr/2026

import tiktoken
from importlib.metadata import version

# Compruebo versión de tiktoker
print("tiktoken version:", version("tiktoken"))

# Creo un objeto tokenizer tipo GPT2
print("Creando objeto tokenizer - Tiktoker GPT2")
tokenizer = tiktoken.get_encoding("gpt2")
print()


# Prueba, incluyendo un marcador de cambio de fuente de datos "<|endoftext|>" y una palabra que no está en el diccionario "some-unknownPlace"
text = "Hello, do you like tea? <|endoftext|> In the sunlit terrace of some-unknownPlace."
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
strings = tokenizer.decode(integers)
print("Texto: ", text)
print("Tokens: ", integers)
print("Back to text: ", strings)
print()


# Try the BPE tokenizer from the tiktoken library on the unknown words “Akwirw ier” and print the individual token IDs. 
# Then, call the decode function on each of the resulting integers in this list to reproduce the mapping shown in figure 2.11. 
#Lastly, call the decode method on the token IDs to check whether it can reconstruct the original input, “Akwirw ier.”
text = "Akwirw ier"
integers = tokenizer.encode(text)
strings = tokenizer.decode(integers)
print("Texto: ", text)
print("Tokens: ", integers)
print("Back to text: ", strings)

