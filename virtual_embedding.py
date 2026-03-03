# extracting V from model.py line 198 : v = self.norm_attn_v(e)

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def extract_normed_v_embeddings(
    text: str, 
    model_path: str = "./fst_1_3B_local", 
    save_path: str = "normed_v_embeddings.json"
):
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print("Loading config and model weights...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.tie_word_embeddings = False 
    
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
    
    print(f"\nInput text: '{text}'")
    print(f"Tokens: {tokens}")

    result_dict = {
        "input_text": text,
        "tokens": tokens,
        "token_ids": input_ids[0].tolist(),
        "normed_v_by_layer": {}
    }

    with torch.no_grad():
        # 获取原始输入 e (Token Embeddings)
        e = model.get_input_embeddings()(input_ids)
        
        # 遍历所有的 Predictive Blocks
        for i, block in enumerate(model.model.predictive_blocks):
            
            # 严格按照你的发现：只提取经过当前层 LayerNorm 的结果
            # 这里的 v 形状为 [batch_size=1, seq_len, hidden_size]
            v = block.norm_attn_v(e)
            
            # 转换为 Python 列表并保存
            result_dict["normed_v_by_layer"][f"layer_{i}"] = v[0].tolist()
            print(f"Extracted normed 'v' for Predictive Layer {i}")

    # 写入 JSON 文件
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
        
    print(f"\nSuccessfully saved normed 'v' embeddings to {save_path}!")

if __name__ == "__main__":
    custom_input = "Hello, world! I am testing the predictive branch."
    extract_normed_v_embeddings(text=custom_input)