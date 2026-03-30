import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull
import argparse
import os

def draw_focus_hull(ax, points, color, alpha=0.2):
    """只为聚焦的目标类画多边形领地"""
    if len(points) >= 3:
        try:
            hull = ConvexHull(points)
            ax.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color, alpha=alpha)
            ax.plot(points[hull.vertices, 0], points[hull.vertices, 1], color=color, lw=2.5, alpha=alpha+0.5)
        except Exception:
            pass

def plot_binary_ax(ax, X_2d, words, categories, focus_cat, title):
    """二值化渲染：目标类高亮，非目标类变灰"""
    # 分离目标点和背景点
    is_target = (categories == focus_cat)
    target_points = X_2d[is_target]
    
    # 1. 先画背景点 (Non-Target) - 浅灰色，低透明度，置于底层
    for i, word in enumerate(words):
        if not is_target[i]:
            x, y = X_2d[i, 0], X_2d[i, 1]
            ax.scatter(x, y, color='#b0b0b0', s=30, alpha=0.3, zorder=1)
            # 背景词也可以稍微标注一下，但字体极小、极淡
            ax.text(x + 0.5, y + 0.5, word, fontsize=6, alpha=0.2, color='gray', zorder=2)

    # 2. 画出目标类的专属领地 (Convex Hull)
    draw_focus_hull(ax, target_points, color='#d62728', alpha=0.15) # 用强烈的红色标注领地
    
    # 3. 最后画目标点 (Target) - 鲜艳红色，大圆点，置于顶层
    for i, word in enumerate(words):
        if is_target[i]:
            x, y = X_2d[i, 0], X_2d[i, 1]
            ax.scatter(x, y, color='#d62728', s=80, alpha=0.9, edgecolors='white', linewidth=1, zorder=5)
            # 目标词清晰标注，加粗
            ax.text(x + 0.5, y + 0.5, word, fontsize=10, fontweight='bold', alpha=0.9, color='black', zorder=6)
            
    ax.set_title(title, fontsize=16, pad=15, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def run_binary_comparison(fst_pt: str, std_pt: str, layer: int, out_dir: str):
    print(f"📦 Loading OV Tensors from {fst_pt} and {std_pt}...")
    fst_data = torch.load(fst_pt)
    std_data = torch.load(std_pt)
    
    words = np.array(fst_data["words"])
    categories = np.array(fst_data["categories"])
    
    # 我们只关注最具代表性的三大词性
    focus_categories = ["Noun", "Verb", "Adjective"]
    
    os.makedirs(out_dir, exist_ok=True)
    std_physical_layer = layer * 2 + 1
    
    print(f"🚀 Generating One-vs-Rest comparisons for Layer {layer:02d}...")

    for h in range(32):
        key = f"L{layer:02d}_H{h:02d}"
        
        if key not in fst_data["ov_tensors"] or key not in std_data["ov_tensors"]:
            continue
            
        fst_X = fst_data["ov_tensors"][key].float().numpy()
        std_X = std_data["ov_tensors"][key].float().numpy()
        
        # 保证 FST 和 Standard 在同一次对比中，降维算法的随机种子一致
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
        fst_2d = tsne.fit_transform(fst_X)
        std_2d = tsne.fit_transform(std_X)
        
        # 为三大词性分别生成图纸
        for focus_cat in focus_categories:
            fig, axes = plt.subplots(1, 2, figsize=(20, 9))
            
            fst_title = f"FST Model (Predictive L{layer:02d}-H{h:02d})\nFocus: {focus_cat} vs Rest"
            std_title = f"Standard Model (L{std_physical_layer:02d}-H{h:02d})\nFocus: {focus_cat} vs Rest"
            
            plot_binary_ax(axes[0], fst_2d, words, categories, focus_cat, fst_title)
            plot_binary_ax(axes[1], std_2d, words, categories, focus_cat, std_title)
            
            # 极简图例
            legend_elements = [
                matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=12, label=f'Target: {focus_cat}'),
                matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#b0b0b0', markersize=10, label='Other Words')
            ]
            fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=14, frameon=False, bbox_to_anchor=(0.5, 0.02))
            plt.subplots_adjust(bottom=0.1)
            
            out_path = os.path.join(out_dir, f"L{layer:02d}_H{h:02d}_{focus_cat}_vs_Rest.png")
            fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
        print(f"  🎨 Rendered Head {h:02d} (Noun/Verb/Adjective focused)")

    print(f"\n🎉 Done! Visually isolated comparisons are in '{out_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fst_pt", type=str, default="fst_ov.pt")
    parser.add_argument("--std_pt", type=str, default="std_transformer_ov.pt")
    parser.add_argument("--layer", type=int, default=11, help="Which aligned layer to compare (0-11)")
    parser.add_argument("--out_dir", type=str, default="tsne_binary_focus")
    
    args = parser.parse_args()
    run_binary_comparison(args.fst_pt, args.std_pt, args.layer, args.out_dir)