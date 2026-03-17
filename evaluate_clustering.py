import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import os
import argparse

def load_data(jsonl_path: str):
    """提取每层的特征矩阵，并保持词汇顺序一致"""
    data_cache = {}
    words_found = []
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if len(words_found) < 200: # 假设总共200词，只记录一次顺序
                words_found.append(data["word"])
            
            for layer, heads_dict in data["heads_v_by_layer"].items():
                layer_idx = int(layer.split('_')[1])
                if layer_idx not in data_cache:
                    data_cache[layer_idx] = {}
                    
                for head, vec in heads_dict.items():
                    head_idx = int(head.split('_')[1])
                    if head_idx not in data_cache[layer_idx]:
                        data_cache[layer_idx][head_idx] = []
                    # 修复 Bug：这里必须是 append，把 200 个词的向量全部塞进去！
                    data_cache[layer_idx][head_idx].append(vec)
                    
    # 重组为矩阵格式 [200, 64]
    for L in data_cache:
        for H in data_cache[L]:
            data_cache[L][H] = np.array(data_cache[L][H])
            
    return data_cache, words_found

def build_ideal_rdm(words_found: list, json_labels_path: str):
    """构建人类视角的理想概念矩阵 (Target RDM)"""
    with open(json_labels_path, "r", encoding="utf-8") as f:
        labels_dict = json.load(f)
        
    n = len(words_found)
    ideal_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # 相同类别的词得 1.0，不同类别的词得 0.0
            if labels_dict.get(words_found[i]) == labels_dict.get(words_found[j]):
                ideal_matrix[i, j] = 1.0
            else:
                ideal_matrix[i, j] = 0.0
                
    # 提取上三角展平
    upper_tri_indices = np.triu_indices_from(ideal_matrix, k=1)
    return ideal_matrix[upper_tri_indices]

def get_layer_scores(data_cache, ideal_rdm_vector):
    """计算每一层中，表现最好的 Head 和该层的平均得分"""
    layer_max_scores = {}
    layer_mean_scores = {}
    
    for L in sorted(data_cache.keys()):
        head_scores = []
        for H in data_cache[L].keys():
            sim_matrix = cosine_similarity(data_cache[L][H])
            upper_tri_indices = np.triu_indices_from(sim_matrix, k=1)
            model_rdm_vector = sim_matrix[upper_tri_indices]
            
            # 计算 Pearson r
            corr = np.corrcoef(model_rdm_vector, ideal_rdm_vector)[0, 1]
            head_scores.append(corr)
            
        layer_max_scores[L] = np.max(head_scores)
        layer_mean_scores[L] = np.mean(head_scores)
        
    return layer_max_scores, layer_mean_scores

def run_evaluation(fst_path, standard_path, labels_path, out_dir):
    print("Loading data...")
    fst_cache, words_fst = load_data(fst_path)
    std_cache, words_std = load_data(standard_path)
    
    assert words_fst == words_std, "Word order mismatch between JSONL files!"
    
    print("Building Target Ideal RDM...")
    ideal_rdm_vector = build_ideal_rdm(words_fst, labels_path)
    
    print("Scoring FST model against Target RDM...")
    fst_max, fst_mean = get_layer_scores(fst_cache, ideal_rdm_vector)
    
    print("Scoring Standard Transformer against Target RDM...")
    std_max, std_mean = get_layer_scores(std_cache, ideal_rdm_vector)
    
    # ==========================================
    # 按照架构映射绘制对比折线图 (FST L[i] vs Std L[2i+1])
    # ==========================================
    print("\nPreparing Alignment Plot...")
    os.makedirs(out_dir, exist_ok=True)
    
    fst_layers = sorted(fst_max.keys())
    aligned_std_layers = [2 * i + 1 for i in fst_layers]
    
    # 过滤掉不存在的层 (防止越界)
    valid_fst_layers = []
    valid_std_layers = []
    plot_fst_max = []
    plot_std_max = []
    
    print("\n📊 对决分数 (Max Head RSA Score):")
    for fst_L, std_L in zip(fst_layers, aligned_std_layers):
        if std_L in std_max:
            valid_fst_layers.append(fst_L)
            valid_std_layers.append(std_L)
            plot_fst_max.append(fst_max[fst_L])
            plot_std_max.append(std_max[std_L])
            print(f"  FST Layer {fst_L:02d} ({fst_max[fst_L]:.4f})  vs  Standard Layer {std_L:02d} ({std_max[std_L]:.4f})")
            
    # 画图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x_axis = np.arange(len(valid_fst_layers))
    
    # 绘制表现最强 Head 的对比
    ax.plot(x_axis, plot_fst_max, marker='o', linewidth=3, markersize=8, color='#d62728', label='FST (Predictive Branch)')
    ax.plot(x_axis, plot_std_max, marker='s', linewidth=3, markersize=8, color='#1f77b4', label='Standard Transformer')
    
    # 设置 X 轴的对比标签
    x_labels = [f"FST L{f}\n(Std L{s})" for f, s in zip(valid_fst_layers, valid_std_layers)]
    ax.set_xticks(x_axis)
    ax.set_xticklabels(x_labels, fontsize=11)
    
    ax.set_xlabel("Architectural Alignment (Layer Pairs)", fontsize=14, labelpad=15)
    ax.set_ylabel("RSA Score with Target RDM (Clustering Purity)", fontsize=14, labelpad=15)
    ax.set_title("Clustering Quality Evolution: FST vs Standard Transformer", fontsize=18, pad=20)
    
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=12, loc='upper left')
    
    out_path = os.path.join(out_dir, "clustering_quality_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n🎉 Victory Check Complete! Chart saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fst", type=str, default="fst_heads_v.jsonl")
    parser.add_argument("--std", type=str, default="standard_heads_v.jsonl")
    parser.add_argument("--labels", type=str, default="fst_top_200.json", help="JSON dict with POS labels")
    parser.add_argument("--out", type=str, default="rsa_results")
    args = parser.parse_args()
    
    run_evaluation(args.fst, args.std, args.labels, args.out)