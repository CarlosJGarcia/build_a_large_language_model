# Next Steps — Code Review Feedback

> Comprehensive review of the Build-a-LLM-from-Scratch project (GSP-2).
> Date: 2026-06-24

---

## Issue 1 (CRITICAL): Malformed `tokenizer.encode` calls

**Severity:** CRITICAL

Across many files, the `allowed_special` argument in `tokenizer.encode()` calls appears malformed:

```python
tokenizer.encode(text, allowed_special={'<'})
# or
tokenizer.encode(text, allowed_special={'""'})
```

Affected files:
- `05/p05_04.py` (lines 199, 390)
- `05/05-02.py` (line 207)
- `06/p06_01.py`
- `07/p07_01.py`
- `07/p07_02.py`
- `07/p07_03.py`
- `gsp-2/p03_train.py` (line 111)
- `gsp-2/p06_prepare_model_fine.py` (line 115)

This will likely cause a `SyntaxError` or `TypeError` at runtime if the files are not clean copies. **Verify every occurrence** has the correct call:

```python
tokenizer.encode(text, allowed_special={"<|endoftext|>"})
```

---

## Issue 2 (HIGH): Massive code duplication

**Severity:** HIGH

The classes `MultiHeadAttention`, `LayerNorm`, `GELU`, `FeedForward`, `TransformerBlock`, and `GPTModel` are copy-pasted into **12+ files**:

- `04/04-01.py`, `04/04-02.py`, `04/04-05.py`, `04/04-06.py`, `04/04-07.py`
- `05/05-02.py`, `05/p05_04.py`, `05/p05_05.py`
- `06/p06_01.py`
- `07/p07_01.py`, `07/p07_02.py`
- `gsp-2/p02_gpt_model.py`

**Risks:**
- Any fix (e.g. GELU device bug) must be manually repeated everywhere.
- Inconsistencies creep in between copies (e.g. GELU in `04/04-06-cuda.py` has `device=x.device`, but GELU in `05/05-02.py` does not).
- The GSP-2 project re-duplicates everything again.

**Fix:** Extract all shared classes into a single module (e.g. `gsp-2/gpt_model.py`) and import it from every script. Use `gsp-2/p02_gpt_model.py` as the canonical source since it has the most up-to-date version (FlashAttention, GELU fix).

---

## Issue 3 (HIGH): GELU device placement bug

**Severity:** HIGH

In many files (e.g. `05/05-02.py`, `06/p06_01.py`, `07/p07_01.py`), the GELU forward method is:

```python
def forward(self, x):
    return 0.5 * x * (1 + torch.tanh(
        torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
        (x + 0.044715 * torch.pow(x, 3))
    ))
```

The constant `torch.tensor(2.0 / torch.pi)` is created on CPU by default. When the model is on GPU, this causes a device mismatch error.

**Correct version** (present in `04/04-06-cuda.py`, `05/p05_04.py`, `gsp-2/p02_gpt_model.py`):

```python
def forward(self, x):
    return 0.5 * x * (1 + torch.tanh(
        torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device)) * 
        (x + 0.044715 * torch.pow(x, 3))
    ))
```

**Better fix:** Create the constant once at init time to avoid allocating it on every forward pass:

```python
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("gelu_const", torch.sqrt(torch.tensor(2.0 / torch.pi)))
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            self.gelu_const * (x + 0.044715 * torch.pow(x, 3))
        ))
```

---

## Issue 4 (MEDIUM): Typo — `CONTEXT_LENGHT` instead of `CONTEXT_LENGTH`

**Severity:** MEDIUM

The variable name `CONTEXT_LENGHT` (missing the second 'H') appears in:
- `04/04-01.py` line 9
- `04/04-03.py` line 10
- `04/04-04.py` line 9
- `04/04-05.py` line 8
- `05/p05_04.py` line 22
- `05/p05_05.py` line 12
- `05/p05_06.py` line 8
- `06/p06_01.py` line 68

In some files, this typo propagates into the config dict key:

```python
GPT_CONFIG_124M = {
    "context_length": CONTEXT_LENGHT,  # key is "context_length", var is "CONTEXT_LENGHT"
}
```

This creates an inconsistency where the config dict always has the correct key `"context_length"`, but the variable holding the value is misspelled. **Fix:** Rename to `CONTEXT_LENGTH` everywhere.

---

## Issue 5 (MEDIUM): Missing gradient clipping

**Severity:** MEDIUM

Both training loops (`gsp-2/p03_train.py` and `05/p05_04.py`) have no gradient clipping:

```python
loss.backward()
optimizer.step()
```

For a 355M model on OpenWebText, gradients can explode. Add after `loss.backward()` and before `optimizer.step()`:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

This is standard practice and was covered in the book's later chapters (Appendix D).

---

## Issue 6 (MEDIUM): Training only 1 epoch on OpenWebText

**Severity:** MEDIUM

In `gsp-2/p03_train.py`:

```python
NUM_EPOCHS = 1
```

With early stopping at `patience=5` and `eval_freq=500`, the early stopper will never trigger because there is only 1 epoch's worth of batches. OpenWebText is ~8GB of text — one pass is a very light pretraining run. The Gutenberg experiment used 5-10 epochs.

**Consider:**
- Increasing `NUM_EPOCHS` if you want a more thoroughly pretrained base model.
- Or explicitly documenting that the model is lightly pretrained and relies on fine-tuning for quality.
- If you keep 1 epoch, the early stopping logic is dead code and can be removed.

---

## Issue 7 (MEDIUM): Collate function edge case

**Severity:** MEDIUM

In `gsp-2/p06_prepare_model_fine.py`, `custom_collate_fn` lines ~189-191:

```python
indices = torch.nonzero(mask).squeeze()
if indices.numel() > 1:                     
    targets[indices[1:]] = ignore_index
```

If a sequence is exactly one token, `squeeze()` returns a 0-dim tensor. While `.numel()` and slicing still work in this case, the code is fragile. If `squeeze()` behavior changes or a zero-length sequence is encountered, it could break.

**Suggested fix:**

```python
if mask.sum() > 0:
    padded_indices = torch.nonzero(mask, as_tuple=True)[0]
    if len(padded_indices) > 1:
        targets[padded_indices[1:]] = ignore_index
```

This is more explicit about intent and safer.

---

## Issue 8 (MEDIUM): Inconsistent model config naming

**Severity:** MEDIUM

Inconsistent config dict naming across the codebase:

| File | Config Name | emb_dim | Correct? |
|------|------------|---------|----------|
| `04/04-01.py` | `GPT_CONFIG_124M` | 768 | Yes |
| `05/p05_04.py` | `GPT_CONFIG_124M` | 768, context=256 | Name OK, but context mismatch with real 124M |
| `gsp-2/p02_gpt_model.py` | `GPT_CONFIG_355M` | 1024 | Yes |
| `gsp-2/p06_prepare_model_fine.py` | `BASE_CONFIG` | 1024 | Ambiguous name |
| `gsp-2/p07_instruct_tune.py` | `BASE_CONFIG` | 1024 | Same |

The name `GPT_CONFIG_124M` in some files has `context_length=256`, which doesn't match the real GPT-2-124M spec (context=1024). And `GPT_CONFIG_124M` in `gsp-2/p02_gpt_model.py` correctly maps to 768 emb_dim.

**Fix:** Rename config dicts to `GPT_CONFIG_SMALL`, `GPT_CONFIG_MEDIUM` or document each config's parameters explicitly so the name reflects reality.

---

## Issue 9 (LOW): Path inconsistency — `ALPACA_FILTERED_PATH`

**Severity:** LOW

Two different paths for similar files:

- `gsp-2/p05_prepare_data_fine.py` saves the filtered Alpaca dataset to:
  `../data/processed/alpaca_filtered.json`
- `gsp-2/p07_instruct_tune.py` writes test results to:
  `../data/processed/gsp-2_test_result.json`

The naming is ambiguous. The first is the **input** (filtered Alpaca dataset). The second is the **output** (dataset with model responses appended). Consider renaming to make the purpose clear:

- `../data/processed/alpaca_filtered.json` → `../data/processed/alpaca_instruct.json`
- `../data/processed/gsp-2_test_result.json` → `../data/processed/alpaca_model_responses.json`

---

## Issue 10 (LOW): `load_weights_into_gpt` uses `np` without importing

**Severity:** LOW

In `gsp-2/p06_prepare_model_fine.py`, line 21:

```python
q_w, k_w, v_w = np.split(
    (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
```

`np` (numpy) is never imported in this file. The function is currently unused (the import is commented out at line 26), but if anyone uncomments it, it will fail with `NameError: name 'np' is not defined`.

Add at the top: `import numpy as np`.

---

## Issue 11 (LOW): Commented-out dead code everywhere

**Severity:** LOW

Every file has commented-out code blocks: original book code, alternate implementations, old approaches. Example in `gpt_download.py` lines 93-121 have the entire original `download_file` function commented out.

This fills files with noise and makes reading harder. **Fix:** Either delete old code or move it to separate branch files (e.g. `05/05-04-original.py`).

---

## Issue 12 (LOW): `download_and_load_gpt2` in p06_prepare_model_fine.py is a half-implemented wrapper

**Severity:** LOW

Lines 128-161 define a function that:
1. Hardcodes settings for the 355M model
2. Loads from `MODEL_PATH` (your local PyTorch checkpoint)
3. Recursively converts tensors to numpy

The docstring says "Loads a local PyTorch model instead of downloading the OpenAI TF checkpoint." Functionally, this is a no-op that just converts your own state dict to numpy so `load_weights_into_gpt` can use it. It only works with your 355M checkpoint.

**Fix:** Rename to `load_local_checkpoint_to_numpy` to be more honest about what it does, or integrate the conversion directly where needed.

---

## Issue 13 (LOW): `dialog/dialog.py` — fragile `sys.path` manipulation

**Severity:** LOW

Lines 13-22:

```python
import_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../05'))
if import_dir not in sys.path:
    sys.path.insert(0, import_dir)
```

This works when the script is run from its own directory but breaks if run from a different working directory. The `../05` relative path won't resolve correctly from elsewhere.

**Fix:** Use the project root as a base and construct absolute paths consistently. Better yet, rely on Python package imports once Issue 2 is fixed.

---

## Issue 14 (LOW): Training loss plot uses `torch.linspace`

**Severity:** LOW

In both training scripts:

```python
epochs_tensor = torch.linspace(0, NUM_EPOCHS, len(train_losses))
```

This creates an evenly spaced array of epoch numbers matching the number of evaluations. But if `eval_freq` changes or early stopping fires before the requested number of epochs, the mapping from array index to actual epoch number is inaccurate.

**Fix:** Track the actual epoch number at each evaluation point instead of using linspace:

```python
# In train_model_simple, when appending losses:
track_epochs.append(epoch)  # actual epoch number, not linspace index
```

---

## Issue 15 (LOW): No `__init__.py` files, no package structure

**Severity:** LOW

The project has no Python package structure. All scripts run as top-level files with `sys.path` manipulation. This is fine for a tutorial project but becomes unmanageable as the number of files grows.

**Suggested structure:**

```
gsp2/
    __init__.py
    model.py          # GPTModel, MultiHeadAttention, LayerNorm, GELU, FeedForward, TransformerBlock
    data.py           # Dataset, DataLoader helpers, collate functions
    train.py          # Training loop, evaluation, loss plotting
    inference.py      # Generation functions (greedy, temperature, top-k)
    fine_tune.py      # SFT pipeline (InstructionDataset, custom_collate_fn)
    utils.py          # File download, token conversion helpers
```

Each script in the project would then simply be a short entry point that imports from `gsp2`.

---

## Issue 16 (LOW): Mixed Spanish/English comments

**Severity:** LOW

The codebase has extensive Spanish comments mixed with English code and English variable names. Examples:

- `# Creo un objeto tokenizer tipo GPT2`
- `# Cargo el libro como una cadena de texto`
- `# Codificando v1`
- `# Guarda el modelo`

This is fine for a personal project, but if you ever share the code or contribute to others, English-only comments are standard in the Python community. If you prefer to keep Spanish, consider making it consistent (every comment, not some).

---

## Issue 17 (LOW): GSP-2 chat uses `input()` which blocks the thread

**Severity:** LOW

In `gsp-2/p08_chat.py`, the interactive chat uses Python's built-in `input()`:

```python
while True:
    user_prompt = input("Prompt: ").strip()
```

This blocks the main thread. With `torch.no_grad()` and GPU inference, this is fine for a simple REPL. But if you add streaming output or background tasks later, this will need to change. Consider `prompt_toolkit` or a simple FastAPI/Gradio interface as a next evolution.

---

## Issue 18 (LOW): No logging — all output is via `print()` / `console.print()`

**Severity:** LOW

All progress, error, and debug output goes through `print()` or `rich.console.Console.print()`. This means:

- No way to save training progress to a log file.
- No structured logging for monitoring.
- Cannot distinguish between INFO, WARNING, and ERROR levels.

**Fix:** Add Python's `logging` module with proper levels:

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log")]
)
```

---

## Issue 19 (LOW): Hardcoded device string `"cuda"` vs `torch.device("cuda")`

**Severity:** LOW

Throughout the codebase, device setup is inconsistent:

```python
# In some files:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# In others:
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Both work with `.to(device)` and `torch.load(..., map_location=device)`, but mixing approaches makes code less predictable. Pick one style and apply it everywhere.

---

## Issue 20 (LOW): `gsp-2/p03_train.py` imports from sibling file with relative import that may break

**Severity:** LOW

```python
from p02_gpt_model import GPTModel, GPT_CONFIG_355M
```

This works because `p03_train.py` is in the same directory as `p02_gpt_model.py`. But if you ever move files (e.g. into a `gsp2/` package per Issue 15), these imports will break.

**Fix:** Use relative imports once a package structure is in place:

```python
from .p02_gpt_model import GPTModel, GPT_CONFIG_355M
```

---

# Summary of Prioritized Actions

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| **P0** | 1 — Verify `allowed_special` in tokenizer calls | 10 min | CRITICAL — may prevent code from running |
| **P0** | 2 — Extract shared model code into single module | 2-3 hours | HIGH — eliminates all duplication |
| **P1** | 3 — Fix GELU device placement in all files | 30 min | HIGH — prevents CUDA errors |
| **P1** | 5 — Add gradient clipping to training loops | 5 min | MEDIUM — prevents gradient explosion |
| **P1** | 4 — Fix `CONTEXT_LENGHT` typo globally | 15 min | MEDIUM — improves readability |
| **P2** | 6 — Decide on NUM_EPOCHS for OpenWebText | 5 min | MEDIUM — clarify training intent |
| **P2** | 8 — Fix config naming inconsistencies | 20 min | MEDIUM — prevents confusion |
| **P2** | 9 — Rename ALPACA_FILTERED_PATH for clarity | 5 min | LOW — prevents file confusion |
| **P3** | 11 — Clean up dead code / commented blocks | 30 min | LOW — improves code readability |
| **P3** | 15 — Create package structure with `__init__.py` | 1 hour | LOW — future-proofing |
| **P3** | 14 — Fix epoch tracking in loss plots | 10 min | LOW — correctness |
| **P3** | 17 — Consider streaming chat interface | 2-3 hours | LOW — UX improvement |
| **P3** | 18 — Add proper logging | 30 min | LOW — observability |