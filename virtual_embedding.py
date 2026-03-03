import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def extract_multiple_words_to_jsonl(
    words_list: list, 
    model_path: str = "./fst_1_3B_local", 
    save_path: str = "structured_heads_v.jsonl" # 后缀改为 .jsonl
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

    print(f"\nStarting extraction for {len(words_list)} words. Results will be saved line-by-line to {save_path}\n")

    # 使用 'w' 模式打开文件，每次处理完一个词就直接写进去一行
    with open(save_path, "w", encoding="utf-8") as f_out:
        
        with torch.no_grad():
            for word in words_list:
                inputs = tokenizer(word, return_tensors="pt")
                input_ids = inputs["input_ids"].to(model.device)
                tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                
                # 如果这个词被切分成了多个 token，打印个提示，但我们依然只取第一个
                if len(tokens) > 1:
                    print(f"⚠️ Warning: '{word}' was split into {tokens}. Analyzing ONLY the first token '{tokens[0]}'.")
                else:
                    print(f"Processing: '{word}' -> Token: '{tokens[0]}'")

                result_dict = {
                    "word": word,
                    "analyzed_token": tokens[0],
                    "token_id": input_ids[0][0].item(),
                    "heads_v_by_layer": {}
                }

                # 提取 embedding 并强制只取第一个 token 的切片
                e = model.get_input_embeddings()(input_ids)
                e_single_token = e[:, 0:1, :] 
                
                for i, block in enumerate(model.model.predictive_blocks):
                    layer_dict = {}
                    v_normed = block.norm_attn_v(e_single_token)
                    v_projected = block.attn.v_proj(v_normed) 
                    
                    B, T, _ = v_projected.size()
                    num_heads = block.attn.num_attention_heads
                    head_dim = block.attn.head_dim
                    
                    v_heads = v_projected.view(B, T, num_heads, head_dim)[0, 0]
                    
                    for head_idx in range(num_heads):
                        layer_dict[f"head_{head_idx}"] = v_heads[head_idx].tolist()
                        
                    result_dict["heads_v_by_layer"][f"layer_{i}"] = layer_dict

                # 将字典转为纯文本字符串，不要格式化(indent)，保证在一行内，并加上换行符
                f_out.write(json.dumps(result_dict, ensure_ascii=False) + "\n")

    print(f"\nAll done! Successfully saved {len(words_list)} words to {save_path}")

if __name__ == "__main__":
    # 在这里填入你想测试的任意多个词
    # 建议包含不同词性（名词、动词、形容词、功能词）来观察它们在空间中的差异
    target_words = [
        "apple", "pear", 
        "king", "queen", 
        "run", "think", 
        "happy", "sad", 
        "and", "the", "because",
        "Mike", "Steve", "name",
    ]
    
    extract_multiple_words_to_jsonl(words_list=target_words)