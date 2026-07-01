python -c '
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.save_pretrained("../models/gsp-2/gsp2_355m_base_llama")
print("Tokenizer files saved!")
'
