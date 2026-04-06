# Librería de BPE (byte pair enconding) tokenization

import tiktoken
from importlib.metadata import version

print("tiktoken version:", version("tiktoken"))
print()

# Creo un objeto tokenizer tipo GPT2
tokenizer = tiktoken.get_encoding("gpt2")

# Prueba
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

