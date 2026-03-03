import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def generate_embeddings():
    text = "Hello, world! I am testing the predictive branch."
    model_path = "./fst_1_3B_local"
    save_path = "custom_virtual_embeddings.json"
    
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print("Loading and patching configuration...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.tie_word_embeddings = False # Fixing the author's config bug
    
    print("Loading model weights (from local disk)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        config=config, 
        trust_remote_code=True, 
        device_map="auto"
    )
    model.eval()

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    result_dict = {
        "input_text": text,
        "tokens": tokens,
        "token_ids": input_ids[0].tolist(),
        "virtual_embeddings_by_layer": {}
    }

    with torch.no_grad():
        e = model.get_input_embeddings()(input_ids)
        for i, block in enumerate(model.model.predictive_blocks):
            v_normed = block.norm_attn_v(e)
            virtual_emb = block.attn.v_proj(v_normed)
            result_dict["virtual_embeddings_by_layer"][f"layer_{i}"] = virtual_emb[0].tolist()
            print(f"Calculated virtual embeddings for Predictive Layer {i}")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
        
    print(f"\nSuccessfully saved virtual embeddings to {save_path}!")

if __name__ == "__main__":
    generate_embeddings()