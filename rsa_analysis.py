import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 防死锁无头渲染
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import os
import argparse

def load_rdms_from_jsonl(jsonl_path: str):
    """从 JSONL 中提取并展平所有的关系矩阵 (RDM)"""
    data_cache = {}
    words_found = []
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            words_found.append(data["word"])
            
            for layer, heads_dict in data["heads_v_by_layer"].items():
                layer_idx = int(layer.split('_')[1])
                if layer_idx not in data_cache:
                    data_cache[layer_idx] = {}
                    
                for head, vec in heads_dict.items():
                    head_idx = int(head.split('_')[1])
                    if head_idx not in data_cache[layer_idx]:
                        data_cache[layer_idx][head_idx] = []
                    data_cache[layer_idx][head_idx].append(vec)
                    
    head_names = []
    rdm_vectors = []
    
    sorted_layers = sorted(data_cache.keys())
    for L in sorted_layers:
        sorted_heads = sorted(data_cache[L].keys())
        for H in sorted_heads:
            head_names.append(f"L{L:02d}_H{H:02d}")
            
            X = np.array(data_cache[L][H])
            sim_matrix = cosine_similarity(X)
            upper_tri_indices = np.triu_indices_from(sim_matrix, k=1)
            rdm_vectors.append(sim_matrix[upper_tri_indices])
            
    return head_names, np.array(rdm_vectors), words_found

def run_inter_model_rsa(model_a_path: str, model_b_path: str, output_dir: str):
    print(f"Loading Model A data from: {model_a_path}")
    heads_A, rdms_A, words_A = load_rdms_from_jsonl(model_a_path)
    
    print(f"Loading Model B data from: {model_b_path}")
    heads_B, rdms_B, words_B = load_rdms_from_jsonl(model_b_path)
    
    # 严格校验词表是否对齐，但放宽架构大小校验！
    assert words_A == words_B, "Error: JSONL files have different words or orders!"
    
    num_heads_A = len(heads_A)
    num_heads_B = len(heads_B)
    print(f"\nLoaded {len(words_A)} words.")
    print(f"Model A heads: {num_heads_A} | Model B heads: {num_heads_B}")
    print(f"Computing {num_heads_A}x{num_heads_B} Asymmetric Inter-Model RSA Matrix...")

    # 堆叠后计算全相关矩阵，然后切出右上角的 A vs B 区域
    combined_rdms = np.vstack([rdms_A, rdms_B])
    full_corr_matrix = np.corrcoef(combined_rdms)
    rsa_matrix = full_corr_matrix[:num_heads_A, num_heads_A:]

    print("\n========================================================")
    print("🎯 【架构映射验证】FST Predictive L[i] vs Standard L[2i+1]:")
    print("我们将抽出对应 Head 0 的分数作为代表来验证你的假说：")
    
    # 提取并验证用户假说：FST 的 i 层对应 Standard 的 2i+1 层
    # 这里我们只挑 Head 0 来展示
    for a_idx, head_a in enumerate(heads_A):
        if "_H00" in head_a:  # 只看 Head 0
            layer_i = int(head_a.split('_')[0].replace('L', ''))
            target_layer_b = 2 * layer_i + 1
            target_head_b_str = f"L{target_layer_b:02d}_H00"
            
            if target_head_b_str in heads_B:
                b_idx = heads_B.index(target_head_b_str)
                score = rsa_matrix[a_idx, b_idx]
                print(f"  FST [{head_a}] vs Standard [{target_head_b_str}] -> RSA Score: {score:.4f}")
            else:
                pass # 越界了就不打印

    print("\n🔥 【跨界融合】全局最相似的任意 Head Pairs (Top 10):")
    flat_indices = np.argsort(rsa_matrix, axis=None)[::-1][:10]
    for idx in flat_indices:
        r, c = np.unravel_index(idx, rsa_matrix.shape)
        print(f"  FST [{heads_A[r]}] & Standard [{heads_B[c]}] -> RSA Score: {rsa_matrix[r, c]:.4f}")
    print("========================================================\n")

    print("Plotting Asymmetric Inter-Model RSA Heatmap...")
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(20, 10))  # 调整为宽图适应长方形矩阵
    
    cax = ax.imshow(rsa_matrix, cmap='magma', interpolation='nearest', vmin=0, vmax=1, aspect='auto')
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Pearson Correlation")
    
    ax.set_title("Inter-Model RSA: FST (Predictive) vs Standard Transformer", fontsize=20, pad=20)
    ax.set_ylabel("FST Heads (Layer -> Head)", fontsize=14)
    ax.set_xlabel("Standard Transformer Heads (Layer -> Head)", fontsize=14)
    
    # 画辅助网格线
    heads_per_layer = 32
    for i in range(1, (num_heads_A // heads_per_layer)):
        ax.axhline(y=i * heads_per_layer - 0.5, color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    for i in range(1, (num_heads_B // heads_per_layer)):
        ax.axvline(x=i * heads_per_layer - 0.5, color='white', linestyle='-', linewidth=0.5, alpha=0.3)

    out_path = os.path.join(output_dir, "inter_model_rsa_heatmap_asymmetric.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🎉 Analysis complete! Download your heatmap: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Inter-Model RSA Analysis.")
    parser.add_argument("--model_a", type=str, default="fst_heads_v.jsonl", help="JSONL for Model A (e.g., FST)")
    parser.add_argument("--model_b", type=str, default="standard_heads_v.jsonl", help="JSONL for Model B (e.g., Standard)")
    parser.add_argument("--out_dir", type=str, default="rsa_results", help="Output directory for plots")
    
    args = parser.parse_args()
    
    run_inter_model_rsa(model_a_path=args.model_a, model_b_path=args.model_b, output_dir=args.out_dir)