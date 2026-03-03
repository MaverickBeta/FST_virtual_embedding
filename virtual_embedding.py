import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def extract_single_word_head_v(
    word: str, 
    model_path: str = "./fst_1_3B_local", 
    save_path: str = "single_word_heads_v.json"
):
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print("Loading config and model...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.tie_word_embeddings = False 
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        config=config, 
        trust_remote_code=True, 
        device_map="auto"
    )
    model.eval()

    # 输入单个单词
    inputs = tokenizer(word, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    print(f"\nTarget Word: '{word}'")
    print(f"Tokenized into: {tokens}")
    
    if len(tokens) > 1:
        print("⚠️ Warning: Your word was split into multiple tokens. We will only analyze the FIRST token.")

    # 准备保存的数据结构
    result_dict = {
        "word": word,
        "analyzed_token": tokens[0],
        "token_id": input_ids[0][0].item(),
        "heads_v_by_layer": {}
    }

    with torch.no_grad():
        # 获取输入
        e = model.get_input_embeddings()(input_ids)
        # 只取第一个 Token 的 Embedding
        # shape 变成 [1, 1, 2048]
        e_first_token = e[:, 0:1, :] 
        
        for i, block in enumerate(model.model.predictive_blocks):
            # 1. 过 LayerNorm
            v_normed = block.norm_attn_v(e_first_token)
            
            # 2. 过 V 的线性投影 (这一步才真正生成了该层专属的 V 特征)
            v_projected = block.attn.v_proj(v_normed) 
            
            # 3. 模拟 Attention 内部的分头逻辑 (重点！)
            B, T, hidden_size = v_projected.size()
            num_heads = block.attn.num_attention_heads
            head_dim = block.attn.head_dim
            
            # shape 转换: [1, 1, 2048] -> [1, 1, 32, 64] -> 去掉 batch 和 seq 维度 -> [32, 64]
            v_heads = v_projected.view(B, T, num_heads, head_dim)[0, 0]
            
            # 保存为一个包含 32 个列表的列表，每个列表长度为 64
            result_dict["heads_v_by_layer"][f"layer_{i}"] = v_heads.tolist()
            print(f"Extracted [32 heads x 64 dim] 'v' for Predictive Layer {i}")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
        
    print(f"\nSuccessfully saved multi-head 'v' to {save_path}!")

if __name__ == "__main__":
    # 输入一个你想探究的单词
    target_word = "apple" 
    extract_single_word_head_v(word=target_word)