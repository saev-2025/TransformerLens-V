import sys
from tqdm import tqdm
import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)
from transformer_lens.HookedLlava import HookedLlava
MODEL_NAME = ""
MODEL_PATH=""
def load_models_and_processor(model_name,model_path):
    processor = LlavaNextProcessor.from_pretrained(model_name,model_path)
    vision_model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.float32, 
        low_cpu_mem_usage=True
    )
    print("Vision model loaded.")
    vision_tower = vision_model.vision_tower
    multi_modal_projector = vision_model.multi_modal_projector
    hook_language_model = HookedLlava.from_pretrained(
        model_name,
        hf_model=vision_model.language_model,
        device="cuda", 
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        tokenizer=None,
        dtype=torch.float32,
        vision_tower=vision_tower,
        multi_modal_projector=multi_modal_projector,
        
    )
    hook_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vision_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    hook_language_model = hook_language_model.to(hook_device)
    vision_model = vision_model.to(vision_device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return processor, vision_model, hook_language_model, tokenizer

def consistent_check(model, hf_model, tokenizer):
    prompts = [
        "The capital of Germany is",
        "2 * 42 = ", 
        "My favorite", 
        "aosetuhaosuh aostud aoestuaoentsudhasuh aos tasat naostutshaosuhtnaoe usaho uaotsnhuaosntuhaosntu haouaoshat u saotheu saonuh aoesntuhaosut aosu thaosu thaoustaho usaothusaothuao sutao sutaotduaoetudet uaosthuao uaostuaoeu aostouhsaonh aosnthuaoscnuhaoshkbaoesnit haosuhaoe uasotehusntaosn.p.uo ksoentudhao ustahoeuaso usant.hsa otuhaotsi aostuhs",
    ]
    
    model.eval()
    hf_model.eval()

    model_device = next(model.parameters()).device
    hf_model_device = next(hf_model.parameters()).device
    
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}")

        prompt_id = tokenizer.encode(prompt, return_tensors="pt").to(model_device)
        prompt_id_hf = tokenizer.encode(prompt, return_tensors="pt").to(hf_model_device)

        tl_logits = model(prompt_id).detach().cpu()
        hf_logits = hf_model(prompt_id_hf).logits.detach().cpu()

        if not torch.allclose(hf_logits, tl_logits, atol=1e-4, rtol=1e-2):
            print(f"Difference found in prompt {i}:")
            print(f"hf_logits: {hf_logits}")
            print(f"tl_logits: {tl_logits}")
            print(f"Difference: {hf_logits - tl_logits}")
            
            abs_diff = torch.max(torch.abs(hf_logits - tl_logits))
            rel_diff = torch.max(torch.abs((hf_logits - tl_logits) / (tl_logits + 1e-8)))
            print(f"Max absolute difference: {abs_diff.item()}")
            print(f"Max relative difference: {rel_diff.item()}")

            if not torch.allclose(hf_logits, tl_logits, atol=1e-3, rtol=1e-2):
                print(f"Larger difference persists for prompt {i}, investigate further.")
        
        assert torch.allclose(hf_logits, tl_logits, atol=1e-4, rtol=1e-2)

    print("Consistency check completed.")



def process_image_and_generate_response(processor, vision_model, image_path):

    image = Image.open(image_path)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is shown in this image?"},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    return inputs

def main():
    processor, vision_model, hook_language_model, tokenizer= load_models_and_processor(MODEL_NAME,MODEL_PATH)
    
    
    consistent_check(hook_language_model, vision_model.language_model, tokenizer)
    image_path = ""
    inputs = process_image_and_generate_response(processor, vision_model, image_path)
    input_text="The capital of America is"
    inputs=inputs.to("cuda:0")
    outputs = hook_language_model.generate(input_text)
    print(outputs)

if __name__ == "__main__":
    main()
