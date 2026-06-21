** GSP-2: Generative Sequential Predictor v2 **

GSP-2 is a medium-scale Large Language Model (LLM) comprising approximately 400 million parameters, designed after the foundational GPT-2 architecture. It has been developed entirely from fundamental principles using the PyTorch deep learning framework. The model's data processing and training pipeline used the OpenWebText dataset for pre-training, the Tiktoken library for byte-pair encoding (BPE) and the AdamW optimizer for stable weight convergence. It integrates FlashAttention to ensure hardware-optimized, memory-efficient attention computations. This project was undertaken as a personal development exercise to explore and implement advanced neural network architectures. Just for fun.

Carlos García \
Kaiseraugst, May 2026


| File | Description | 
|---------|------------------|
| 02-04-openwebtext.py | Load openwebtext and tokenize | 
| 02-07.py | Tokenizing and embedding | 
| 04-06.py | GPT (Generative Pretrained Transformer) model architecture implementation, small size (124 M param)| 
| 04-06-cuda.py | 04-06 with CUDA instead of CPU | 
| 04-exercise-42-cuda.py | 04-06 in medium size (400 M param) instead of small |
| 05-04.py | Train the (small size) model with 30 books from Project Gutemberg sot learns words, syntax and builds phrases | 


| File | Description | 
|---------|------------------|
| p01_prepare_data.py | Load OpenWebText, tokenize and save. |
| p02_gpt_model.py | Library for model definition. Contains the classes of the GPT-2 architecture (MultiHeadAttention, TransformerBlock, GPTModel).|
| p03_train.py | Load the saved data from disk, import the GPTModel from p02_gpt_model.py, set up the PyTorch DataLoader, define the Optimizer, run the training loop on the GPU and save the model.|
| p04_inference.py | Load the model from disk and make some test inferences. |
| p05_prepare_data_fine.py | Prepare fine-tuning dataset. Load the Alpaca dataset from HF, filter out the items with extra inputs and save as local .json|
| p06_prepare_model_fine.py | Library for model fine-tuning. |
| p07_instruct_tune.py | Load the model, run the fine-tuning on the GPU and save the model. |
| p08_chat.py | Load the model. Interactive chat for testing the model. |