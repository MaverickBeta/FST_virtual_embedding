import torch
import json
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def extract_pure_ov_embeddings(input_json: str, model_path: str, save_path: str):
    if not os.path.exists(input_json):
        print(f"❌ Error: {input_json} not found. Run extract_top_words.py first!")
        return

    with open(input_json, "r", encoding="utf-8") as f:
        word_dict = json.load(f)
        
    words = list(word_dict.keys())
    categories = list(word_dict.values())
    
    print(f"📦 Loaded {len(words)} words.")
    print(f"🧠 Loading tokenizer and model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.tie_word_embeddings = False  # 救命补丁
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=True, device_map="auto"
    )
    model.eval()

    base_model = getattr(model, "model", model)
    
    # ==========================================
    # 核心修复：精准架构对齐嗅探
    # ==========================================
    if hasattr(base_model, "predictive_blocks"):
        blocks = base_model.predictive_blocks
        model_type = "FST"
        print(f"✅ Detected FST architecture with {len(blocks)} Predictive Blocks.")
    elif hasattr(base_model, "blocks"):
        # 提取索引为 1, 3, 5... 的层 (即物理上的第 2, 4, 6... 层)
        blocks = base_model.blocks[1::2]
        model_type = "Standard"
        print(f"✅ Detected Standard Transformer. Aligned to extract {len(blocks)} layers (indices 1, 3, 5...).")
    else:
        raise ValueError("❌ Unknown architecture!")
    
    results = {
        "words": words,
        "categories": categories,
        "model_type": model_type,
        "ov_tensors": {}
    }

    print(f"\n🚀 Extracting 2048-D OV Circuit Embeddings...")

    with torch.no_grad():
        inputs = tokenizer(words, return_tensors="pt", padding=True, add_special_tokens=False)
        input_ids = inputs["input_ids"].to(model.device)
        
        # 获取最底层的绝对纯净 Embedding (E)
        E_raw = model.get_input_embeddings()(input_ids)[:, 0, :]
        
        for l_idx, block in enumerate(blocks):
            # 智能选择 LayerNorm
            if model_type == "FST":
                E_bar = block.norm_attn_v(E_raw)
            else:
                E_bar = block.norm_attn(E_raw)
            
            V_full = block.attn.v_proj(E_bar)
            
            num_heads = block.attn.num_attention_heads
            head_dim = block.attn.head_dim
            V_heads = V_full.view(len(words), num_heads, head_dim)
            W_O_full = block.attn.o_proj.weight
            
            for h_idx in range(num_heads):
                V_h = V_heads[:, h_idx, :]
                W_O_h = W_O_full[:, h_idx * head_dim : (h_idx + 1) * head_dim]
                
                OV_h = torch.matmul(V_h, W_O_h.t())
                
                # 【重要】：无论模型类型，我们都强制使用连续的 L00 - L11 作为 Key。
                # 这使得后续作图时，FST 和 Standard 能够自动一对一完美缝合！
                head_key = f"L{l_idx:02d}_H{h_idx:02d}"
                results["ov_tensors"][head_key] = OV_h.cpu().to(torch.float16)
                
            actual_layer = l_idx if model_type == "FST" else (l_idx * 2 + 1)
            print(f"  🔗 Processed Aligned Layer {l_idx:02d} (Physical Layer {actual_layer:02d}) - 32 Heads extracted")

    torch.save(results, save_path)
    print(f"\n🎉 Saved to {save_path} ({os.path.getsize(save_path) / (1024 * 1024):.2f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 2048-D OV Embeddings")
    
    # 【修改点 1】：默认去 data/ 找 JSON 词表
    parser.add_argument("--input_json", type=str, default="data/top_200_words.json")
    
    # 【修改点 2】：给个默认值指向 models/ 下的 FST 模型，省得每次手敲
    parser.add_argument("--model_path", type=str, default="models/fst_1_3B_local", help="Model path")
    
    # 【修改点 3】：默认保存到 data/ 文件夹
    parser.add_argument("--save_path", type=str, default="data/fst_ov.pt", help="Output .pt file")
    
    args = parser.parse_args()
    
    # 【安全补丁】：确保保存的父文件夹（比如 data/）存在
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
    extract_pure_ov_embeddings(args.input_json, args.model_path, args.save_path)