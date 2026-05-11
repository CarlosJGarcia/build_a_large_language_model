# BPE (byte pair enconding) tokenization - Tiktoken library
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
print()


# Tokenizar el libro "The Veredict"
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    print("Reading file")
    raw_text = f.read()

integers = tokenizer.encode(raw_text)
strings = tokenizer.decode(integers)
print("Texto: ", raw_text)
print("Tokens: ", integers)
print("Total number of tokens:", len(integers))
print("Back to text: ", strings)


#Quito los primeros 50 tokens. enc_sample = todo los tokens desde el número 51 al final
enc_sample = integers[50:]


# Creo los pares input-target
# x: input
# y: target
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")
print() 

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)
print()

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
print()