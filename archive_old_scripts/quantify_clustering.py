import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
import argparse
import os

def get_target_silhouette_score(X, categories, target_cat):
    """
    计算 '目标类 vs 其他类' 的二元轮廓系数。
    我们只关注 target_cat（比如 'Verb'）的平均得分。
    """
    # 构建二元标签：目标类为 1，非目标类为 0
    binary_labels = np.array([1 if cat == target_cat else 0 for cat in categories])
    
    # 只有一类（或者全是一个类）时无法算轮廓系数
    if len(np.unique(binary_labels)) < 2:
        return 0.0
        
    # 在 2048 维空间直接计算所有点的轮廓系数 (使用余弦距离)
    # 轮廓系数范围 [-1, 1]，越大越好
    sample_scores = silhouette_samples(X, binary_labels, metric='cosine')
    
    # 我们只关心目标类（比如动词）的得分均值！
    target_scores = sample_scores[binary_labels == 1]
    
    return np.mean(target_scores)

def quantify_layer_heads(fst_pt: str, std_pt: str, layer: int, target_cat: str, out_dir: str):
    print(f"📦 Loading 2048-D OV Tensors from {fst_pt} and {std_pt}...")
    fst_data = torch.load(fst_pt)
    std_data = torch.load(std_pt)
    
    categories = np.array(fst_data["categories"])
    
    if target_cat not in categories:
        print(f"❌ Error: Target category '{target_cat}' not found in data.")
        return

    os.makedirs(out_dir, exist_ok=True)
    std_physical_layer = layer * 2 + 1
    
    print(f"🚀 Quantifying '{target_cat}' clustering quality for Aligned Layer {layer:02d}...")
    
    fst_scores = []
    std_scores = []
    valid_heads = []
    
    for h in range(32):
        key = f"L{layer:02d}_H{h:02d}"
        if key not in fst_data["ov_tensors"] or key not in std_data["ov_tensors"]:
            continue
            
        fst_X = fst_data["ov_tensors"][key].float().numpy()
        std_X = std_data["ov_tensors"][key].float().numpy()
        
        # 计算 2048 维余弦空间的轮廓系数
        score_fst = get_target_silhouette_score(fst_X, categories, target_cat)
        score_std = get_target_silhouette_score(std_X, categories, target_cat)
        
        fst_scores.append(score_fst)
        std_scores.append(score_std)
        valid_heads.append(h)

    # =============== 打印排行榜 ===============
    print(f"\n🏆 Top 3 '{target_cat}' Experts in FST Model:")
    top_fst_idx = np.argsort(fst_scores)[::-1][:3]
    for idx in top_fst_idx:
        print(f"  🥇 Head {valid_heads[idx]:02d}: Score = {fst_scores[idx]:.4f}")
        
    print(f"\n🏆 Top 3 '{target_cat}' Experts in Standard Model:")
    top_std_idx = np.argsort(std_scores)[::-1][:3]
    for idx in top_std_idx:
        print(f"  🥇 Head {valid_heads[idx]:02d}: Score = {std_scores[idx]:.4f}")

    # =============== 绘制对比柱状图 ===============
    x = np.arange(len(valid_heads))
    width = 0.35

    fig, ax = plt.subplots(figsize=(18, 7))
    rects1 = ax.bar(x - width/2, fst_scores, width, label=f'FST (Predictive L{layer:02d})', color='#d62728', alpha=0.85)
    rects2 = ax.bar(x + width/2, std_scores, width, label=f'Standard (Physical L{std_physical_layer:02d})', color='#1f77b4', alpha=0.85)

    ax.set_ylabel(f'Silhouette Score (Target: {target_cat})', fontsize=14, labelpad=15)
    ax.set_xlabel('Attention Head Index (0-31)', fontsize=14, labelpad=15)
    ax.set_title(f"Quantitative Evaluation: '{target_cat}' Clustering Purity in 2048-D Space", fontsize=18, pad=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"H{h:02d}" for h in valid_heads], fontsize=10)
    ax.legend(fontsize=12, loc='upper left', frameon=True, framealpha=0.9)
    
    # 画一条 0 分基准线
    ax.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.5)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6, zorder=0)

    # 让负分和低分更显眼，设定一个合理的 y 轴范围
    y_min = min(min(fst_scores), min(std_scores), -0.05)
    y_max = max(max(fst_scores), max(std_scores), 0.1) + 0.05
    ax.set_ylim(y_min, y_max)

    out_path = os.path.join(out_dir, f"quantify_L{layer:02d}_{target_cat}.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"\n🎉 Chart saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fst_pt", type=str, default="fst_ov.pt")
    parser.add_argument("--std_pt", type=str, default="std_transformer_ov.pt")
    parser.add_argument("--layer", type=int, default=11, help="Which aligned layer to quantify (0-11)")
    parser.add_argument("--target_cat", type=str, default="Verb", choices=["Noun", "Verb", "Adjective", "Adverb"], help="Which category to focus on")
    parser.add_argument("--out_dir", type=str, default="quantify_results")
    
    args = parser.parse_args()
    quantify_layer_heads(args.fst_pt, args.std_pt, args.layer, args.target_cat, args.out_dir)