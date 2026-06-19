# GSP-2 Chat (p08_chat.py)

# Loads the trained model
# Interactive chat

# Basel 19/Jun/2026

import torch
import tiktoken
from rich.console import Console
from p07_instruct_tune import MODEL_PATH_SFT
from p02_gpt_model import GPTModel, GPT_CONFIG_355M
from p06_prepare_model_fine import format_input, generate, text_to_token_ids, token_ids_to_text

# Cargo el modelo
console = Console()
console.print(f"\nLoading model {MODEL_PATH_SFT}", style="gold1", highlight=False)
if torch.cuda.is_available():
        device = "cuda"
        console.print(f"Using GPU: {torch.cuda.get_device_name(0)}", style="bright_blue", highlight=False)
else:
        device = "cpu"
        console.print(f"CUDA not available. Using CPU.", style="gold1") 

model_state_dict = torch.load(MODEL_PATH_SFT, map_location=device, weights_only=True)

model = GPTModel(GPT_CONFIG_355M)
model.load_state_dict(model_state_dict)
model.eval()
model.to(device)
print("Model loaded")

# Initialize the Tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Interactive chat loop
console.print("Type your prompt and press Enter. Type 'quit' or 'exit' to stop.\n")

while True:
    try:
        # Get input from the user
        user_prompt = input("Prompt: ").strip()
        if not user_prompt:
            continue
        if user_prompt.lower() in ["quit", "exit"]:
            console.print("\nExiting chat.\n", style="gold1")
            break

        # Format the entry to match the Alpaca SFT template structure
        entry = {"instruction": user_prompt, "input": ""}
        formatted_prompt = format_input(entry)

        # Convert the formatted text into token IDs
        input_ids = text_to_token_ids(formatted_prompt, tokenizer).to(device)

        console.print("Answer: ", style="white", end="", highlight=False)

        # Generate tokens from the model
        with torch.no_grad():
            token_ids = generate(
                model=model,
                idx=input_ids,
                max_new_tokens=256,
                context_size=GPT_CONFIG_355M["context_length"],
                eos_id=50256  # <|endoftext|> token to stop generation early if the model finishes
            )
        
        # Convert token IDs back to a string
        generated_text = token_ids_to_text(token_ids, tokenizer)

        # Extract only the newly generated response (slice away the prompt structure)
        response_text = (
            generated_text[len(formatted_prompt):]
            .replace("### Response:", "")
            .strip()
        )
        
        console.print(response_text, style="white", highlight=False)
        print()

    except KeyboardInterrupt:
        console.print("\nExiting chat.\n", style="gold1")
        break