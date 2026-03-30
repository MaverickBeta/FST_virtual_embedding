import torch
import json
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def extract_v_embeddings(
    input_json: str, 
    model_path: str, 
    save_path: str
):
    if not os.path.exists(input_json):
        print(f"Error: {input_json} not found. Run extract_top_words.py first!")
        return

    with open(input_json, "r", encoding="utf-8") as f:
        word_dict = json.load(f)
    
    print(f"Loaded {len(word_dict)} words from {input_json}.")
    print(f"Loading tokenizer and model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.tie_word_embeddings = False 
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=True, device_map="auto"
    )
    model.eval()

    # ==========================================
    # 精确架构识别与拉链式重组 (Zip Interleaving)
    # ==========================================
    base_model = getattr(model, "model", model)
    blocks = []
    
    if hasattr(base_model, "feature_blocks") and hasattr(base_model, "predictive_blocks"):
        print("✅ Detected FST architecture: 'feature_blocks' & 'predictive_blocks'.")
        # 像拉链一样交替拼装 (Feature 0, Predictive 0, Feature 1, Predictive 1...)
        for f_block, p_block in zip(base_model.feature_blocks, base_model.predictive_blocks):
            blocks.append(f_block)
            blocks.append(p_block)
        print(f"🔗 Interleaved into {len(blocks)} unified layers.")
        arch_type = "FST"
        
    elif hasattr(base_model, "predictive_blocks"):
        # 兼容备用情况：只有 predictive_blocks
        print("✅ Detected FST (Predictive ONLY).")
        blocks = base_model.predictive_blocks
        arch_type = "FST_PRED_ONLY"
        
    elif hasattr(base_model, "blocks"):
        print("✅ Detected Standard Transformer architecture (`blocks`).")
        blocks = base_model.blocks
        print(f"🔗 Found {len(blocks)} standard layers.")
        arch_type = "STANDARD"
        
    else:
        raise ValueError("Unrecognized architecture! Check model structure.")

    print(f"\n🚀 Extracting vectors from {len(blocks)} layers and saving to {save_path}...")

    with open(save_path, "w", encoding="utf-8") as f_out:
        with torch.no_grad():
            for word, category in word_dict.items():
                inputs = tokenizer(word, return_tensors="pt")
                input_ids = inputs["input_ids"].to(model.device)
                tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                
                result_dict = {
                    "word": word,
                    "analyzed_token": tokens[0],
                    "token_id": input_ids[0][0].item(),
                    "category": category, 
                    "heads_v_by_layer": {}
                }

                e = model.get_input_embeddings()(input_ids)
                e_single_token = e[:, 0:1, :] 
                
                for i, block in enumerate(blocks):
                    layer_dict = {}
                    
                    # 智能探测 LayerNorm 路径（FST 通常是 norm_attn_v，Standard 是 norm_attn）
                    if hasattr(block, "norm_attn_v"):
                        v_normed = block.norm_attn_v(e_single_token)
                    elif hasattr(block, "norm_attn"):
                        v_normed = block.norm_attn(e_single_token)
                    else:
                        raise AttributeError(f"LayerNorm not found in block {i}!")
                    
                    # V 的线性映射
                    v_projected = block.attn.v_proj(v_normed)
                    
                    # 获取分头参数
                    num_heads = block.attn.num_attention_heads
                    head_dim = block.attn.head_dim
                    
                    B, T, _ = v_projected.size()
                    
                    # 魔法分头
                    v_heads = v_projected.view(B, T, num_heads, head_dim)[0, 0]
                    
                    for head_idx in range(num_heads):
                        layer_dict[f"head_{head_idx}"] = v_heads[head_idx].tolist()
                        
                    result_dict["heads_v_by_layer"][f"layer_{i}"] = layer_dict

                f_out.write(json.dumps(result_dict, ensure_ascii=False) + "\n")

    print(f"🎉 All done! Successfully processed {len(word_dict)} words across {len(blocks)} layers.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract virtual embeddings dynamically.")
    parser.add_argument("--input_json", type=str, default="top_200_words.json", help="Input JSON")
    parser.add_argument("--model_path", type=str, default="./fst_1_3B_local", help="Model path/ID")
    parser.add_argument("--save_path", type=str, default="structured_heads_v.jsonl", help="Output JSONL")
    args = parser.parse_args()
    
    extract_v_embeddings(input_json=args.input_json, model_path=args.model_path, save_path=args.save_path)