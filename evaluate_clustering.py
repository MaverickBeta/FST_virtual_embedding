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
            if len(words_found) < 200: 
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
            if labels_dict.get(words_found[i]) == labels_dict.get(words_found[j]):
                ideal_matrix[i, j] = 1.0
            else:
                ideal_matrix[i, j] = 0.0
                
    upper_tri_indices = np.triu_indices_from(ideal_matrix, k=1)
    return ideal_matrix[upper_tri_indices]

def get_all_scores_list(data_cache, ideal_rdm_vector):
    """底层提取器：计算每一层每一个 Head 的分数，直接返回列表字典"""
    layer_scores_list = {}
    for L in sorted(data_cache.keys()):
        head_scores = []
        for H in data_cache[L].keys():
            sim_matrix = cosine_similarity(data_cache[L][H])
            upper_tri_indices = np.triu_indices_from(sim_matrix, k=1)
            model_rdm_vector = sim_matrix[upper_tri_indices]
            
            # 计算 Pearson r
            corr = np.corrcoef(model_rdm_vector, ideal_rdm_vector)[0, 1]
            head_scores.append(corr)
            
        layer_scores_list[L] = head_scores
    return layer_scores_list

def run_evaluation(args):
    print("Loading data...")
    fst_cache, words_fst = load_data(args.fst)
    std_cache, words_std = load_data(args.std)
    
    assert words_fst == words_std, "Word order mismatch between JSONL files!"
    
    print("Building Target Ideal RDM...")
    ideal_rdm_vector = build_ideal_rdm(words_fst, args.labels)
    
    print("Scoring all heads for FST and Standard models...")
    fst_scores = get_all_scores_list(fst_cache, ideal_rdm_vector)
    std_scores = get_all_scores_list(std_cache, ideal_rdm_vector)
    
    print(f"\nPreparing Alignment Plot (Mode: {args.mode.upper()})...")
    os.makedirs(args.out, exist_ok=True)
    
    fst_layers = sorted(fst_scores.keys())
    valid_layers = [L for L in fst_layers if L in std_scores]
    
    fig, ax = plt.subplots(figsize=(20, 8))
    x_axis = np.arange(len(valid_layers))
    
    # 动态构建 X 轴标签
    def get_layer_label(idx):
        if idx % 2 == 0: return f"Feat {idx//2}\n(L{idx})"
        else: return f"Pred {idx//2}\n(L{idx})"
    x_labels = [get_layer_label(f) for f in valid_layers]
    ax.set_xticks(x_axis)
    ax.set_xticklabels(x_labels, fontsize=10)
    
    # 绘制垂直交替阴影带 (Predictive layers)
    for i in range(len(valid_layers)):
        if valid_layers[i] % 2 == 1:
            ax.axvspan(i - 0.5, i + 0.5, facecolor='gray', alpha=0.1)

    # ==========================================
    # 模式分流：根据 args.mode 渲染不同的图表
    # ==========================================
    if args.mode == "topk" or args.mode == "threshold":
        # 数据转换
        plot_fst_y = []
        plot_std_y = []
        
        for L in valid_layers:
            if args.mode == "topk":
                # 计算 Top-K 平均
                plot_fst_y.append(np.mean(sorted(fst_scores[L], reverse=True)[:args.top_k]))
                plot_std_y.append(np.mean(sorted(std_scores[L], reverse=True)[:args.top_k]))
                ylabel = f"RSA Score (Top-{args.top_k} Heads Mean)"
                title = f"Committee Purity: Top-{args.top_k} Experts Evolution"
                ylim_max = 1.0
            else:
                # 计算及格人数 (Threshold)
                plot_fst_y.append(sum(1 for s in fst_scores[L] if s >= args.threshold))
                plot_std_y.append(sum(1 for s in std_scores[L] if s >= args.threshold))
                ylabel = f"Number of 'Expert' Heads (r >= {args.threshold})"
                title = f"Expert Emergence: Heads Learning Human Semantics (Threshold {args.threshold})"
                ylim_max = 32 # 最多32个头

        # 拆分 FST 的特征层和预测层以赋予不同颜色
        feat_x = [i for i, L in enumerate(valid_layers) if L % 2 == 0]
        feat_y = [plot_fst_y[i] for i in feat_x]
        pred_x = [i for i, L in enumerate(valid_layers) if L % 2 == 1]
        pred_y = [plot_fst_y[i] for i in pred_x]
        
        # 绘图线与散点
        ax.plot(x_axis, plot_fst_y, linestyle='--', linewidth=2, color='#7f7f7f', alpha=0.6, zorder=1)
        ax.plot(feat_x, feat_y, marker='o', linestyle='', markersize=11, color='#d62728', label='FST (Feature Block)', zorder=3)
        ax.plot(pred_x, pred_y, marker='D', linestyle='', markersize=10, color='#ff7f0e', label='FST (Predictive Block)', zorder=3)
        ax.plot(x_axis, plot_std_y, marker='s', linestyle='-', linewidth=3, markersize=8, color='#1f77b4', label='Standard Transformer', zorder=2)
        
        ax.set_ylim(0.0, ylim_max)

    elif args.mode == "violin":
        # 数据准备
        std_data = [std_scores[L] for L in valid_layers]
        fst_data = [fst_scores[L] for L in valid_layers]
        
        # 绘制 Standard 小提琴图
        parts_std = ax.violinplot(std_data, positions=x_axis - 0.15, showmeans=True, widths=0.3)
        for pc in parts_std['bodies']: 
            pc.set_facecolor('#1f77b4'); pc.set_alpha(0.5)
        parts_std['cmeans'].set_color('#1f77b4')
        parts_std['cbars'].set_color('#1f77b4')
        parts_std['cmins'].set_color('#1f77b4')
        parts_std['cmaxes'].set_color('#1f77b4')
        
        # 绘制 FST 小提琴图
        parts_fst = ax.violinplot(fst_data, positions=x_axis + 0.15, showmeans=True, widths=0.3)
        for i, pc in enumerate(parts_fst['bodies']):
            if valid_layers[i] % 2 == 0:
                pc.set_facecolor('#d62728') # Feature 是红色
            else:
                pc.set_facecolor('#ff7f0e') # Predictive 是橙色
            pc.set_alpha(0.7)
        parts_fst['cmeans'].set_color('black')
        parts_fst['cbars'].set_color('black')
        parts_fst['cmins'].set_color('black')
        parts_fst['cmaxes'].set_color('black')

        # 添加图例占位符
        ax.plot([], [], color='#1f77b4', linewidth=8, alpha=0.5, label='Standard Transformer Distribution')
        ax.plot([], [], color='#d62728', linewidth=8, alpha=0.7, label='FST Feature Block Distribution')
        ax.plot([], [], color='#ff7f0e', linewidth=8, alpha=0.7, label='FST Predictive Block Distribution')

        ylabel = "RSA Score Distribution (All 32 Heads)"
        title = "Panoramic Distribution: Hero Heads vs Noise (Violin Plot)"
        ax.set_ylim(min(np.min(fst_data), np.min(std_data)) - 0.1, 1.0)

    # 统一设置图表修饰
    ax.set_xlabel("Architectural Alignment (Interleaved Blocks 0-23)", fontsize=14, labelpad=15)
    ax.set_ylabel(ylabel, fontsize=14, labelpad=15)
    ax.set_title(title, fontsize=18, pad=20)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.legend(fontsize=13, loc='upper left', framealpha=0.9)
    
    out_file = f"clustering_eval_{args.mode}.png"
    out_path = os.path.join(args.out, out_file)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n🎉 {args.mode.upper()} Analysis Complete! Chart saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fst", type=str, default="fst_heads_v.jsonl")
    parser.add_argument("--std", type=str, default="standard_heads_v.jsonl")
    parser.add_argument("--labels", type=str, default="fst_top_200.json")
    parser.add_argument("--out", type=str, default="rsa_results")
    
    # 核心模式控制参数
    parser.add_argument("--mode", type=str, choices=["topk", "violin", "threshold"], default="topk",
                        help="Choose evaluation plot type: topk, violin, or threshold")
    parser.add_argument("--top_k", type=int, default=5, 
                        help="Only valid for mode=topk: Number of top heads to average")
    parser.add_argument("--threshold", type=float, default=0.2, 
                        help="Only valid for mode=threshold: Correlation threshold to be considered an 'expert'")
    
    args = parser.parse_args()
    run_evaluation(args)