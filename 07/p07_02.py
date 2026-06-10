import json
import torch
import tiktoken
from rich.console import Console
from torch.utils.data import Dataset

FILE_PATH = "../data/raw/instruction-data.json"

# Format prompting function
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:         #1
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


# Collate function to implement the padding process (draft #1)
def custom_collate_draft_1(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)   #1
    inputs_lst = []

    for item in batch:                                      #2
        new_item = item.copy()
        new_item += [pad_token_id]


        padded = (
            new_item + [pad_token_id] * 
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])                 #3
        inputs_lst.append(inputs)

    inputs_tensor = torch.stack(inputs_lst).to(device)     #4
    return inputs_tensor
#1 Finds the longest sequence in the batch
#2 Pads and prepares inputs
#3 Removes extra padded token added earlier
#4 Converts the list of inputs to a tensor and transfers it to the target device

# Updated collate function (draft #2)
def custom_collate_draft_2(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] * 
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])     #1
        targets = torch.tensor(padded[1:])    #2
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


# Collate function (versión definitiva)
def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]


        padded = (                                    #1
            new_item + [pad_token_id] *          
            (batch_max_length - len(new_item))   
        )
        inputs = torch.tensor(padded[:-1])            #2
        targets = torch.tensor(padded[1:])            #3

        mask = targets == pad_token_id                #4
        indices = torch.nonzero(mask).squeeze()     
        if indices.numel() > 1:                     
            targets[indices[1:]] = ignore_index     

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]       #5
            targets = targets[:allowed_max_length]  

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor
#1 Pads sequences to max_length
#2 Truncates the last token for inputs
#3 Shifts +1 to the right for targets
#4 Replaces all but the first padding tokens in targets by ignore_index
#5 Optionally truncates to the maximum sequence length

# =========
# Execution
# =========
if __name__ == "__main__":  
    # Open the file and load its contents into the 'data' variable
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 50
    console = Console()
    console.print(f"\nData", style="gold1")
    print(f"Example entry:\n {data[50]}\n")

    console.print(f"Formatted", style="gold1")
    model_input = format_input(data[50])
    desired_response = f"\n\n### Response:\n{data[50]['output']}"
    print(model_input + desired_response)

    # 999 (No input)
    console.print(f"\nData", style="gold1")
    print(f"Example entry:\n {data[999]}\n")

    console.print(f"Formatted", style="gold1")
    model_input = format_input(data[999])
    desired_response = f"\n\n### Response:\n{data[999]['output']}"
    print(model_input + desired_response)
    print()

    # Divide the dataset into training, validation, and test
    train_portion = int(len(data) * 0.85)                     # 85% Training
    test_portion = int(len(data) * 0.1)                       # 10% Test
    val_portion = len(data) - train_portion - test_portion    #  5% Validation

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))
    print()

    # Creo un objeto tokenizer tipo GPT2
    console = Console()
    console.print(f"\nTokenizer - Tiktoken GPT2", style="gold1")
    tokenizer = tiktoken.get_encoding("gpt2")
    print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
    print()

    # Test the collate function. Three different inputs assembled into a batch
    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]
    batch = (
        inputs_1,
        inputs_2,
        inputs_3
    )
    print(custom_collate_draft_1(batch))
    print()

    console.print(f"---\n", style="gold1")

    # Test the updated collate function
    inputs, targets = custom_collate_draft_2(batch)
    print(inputs)
    print(targets)


    # Test the definitive collate function
    inputs, targets = custom_collate_fn(batch)
    print(inputs)
    print(targets)

    console.print(f"---\n", style="gold1")

    logits_1 = torch.tensor(
        [[-1.0, 1.0],                   # Predictions for 1st Token
        [-0.5, 1.5]]                   # Predictions for 2nd Token
    )
    targets_1 = torch.tensor([0, 1])    # Correct token indices to generate
    loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
    print("Loss value: ", loss_1)
    print()

    console.print(f"---\n", style="gold1")

    logits_2 = torch.tensor(
        [[-1.0, 1.0],
        [-0.5, 1.5],
        [-0.5, 1.5]]                    # New third token ID prediction
    )
    targets_2 = torch.tensor([0, 1, 1])
    loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
    print("Loss value: ", loss_2)
    print()

    console.print(f"---\n", style="gold1")

    # Replace the third target token ID with -100:
    targets_3 = torch.tensor([0, 1, -100])
    loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)
    print("Loss value: ",loss_3)
    print("loss_1 == loss_3:", loss_1 == loss_3)
    print()