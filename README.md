# TransformerLens-V

This repository is dedicated to analyzing transformer-based multimodal model in a interpretability perspective. Building on [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), we developed TransformerLens-V to facilitate interpert multi-modal models, such as LLaVA-NeXT and Chameleon. 

## Installation

Clone the source code from GitHub:
```bash
git clone https://github.com/saev-2025/TransformerLens-V.git
```
Create Environment:
```bash
pip install -r TransformerLens-V/requirements.txt
```

## Use

```python
from transformers import (
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)
from transformer_lens.HookedLlava import HookedLlava

MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"

processor = LlavaNextProcessor.from_pretrained(MODEL_NAME)
vision_model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float32, 
        low_cpu_mem_usage=True
)

hf_model=vision_model.language_model
model = HookedLlava.from_pretrained(
    MODEL_NAME, 
    hf_model=hf_model,
    torch_dtype=torch.float32, 
    low_cpu_mem_usage=True,
    device="cuda:2",
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    tokenizer=None,
    )
```