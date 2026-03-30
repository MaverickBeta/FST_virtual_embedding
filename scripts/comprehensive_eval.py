import torch
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples
from scipy.spatial import ConvexHull
import argparse
import os
from tqdm import tqdm

# 彻底抛弃 pyplot 状态机，使用底层的 Agg 渲染引擎，杜绝任何内存泄漏！
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

COLOR_MAP = {
    "Noun": "#1f77b4", "Verb": "#d62728", "Adjective": "#2ca02c", 
    "Adverb": "#9467bd", "Other": "#7f7f7f"
}

def get_target_silhouette_score(X, categories, target_cat):
    binary_labels = np.array([1 if cat == target_cat else 0 for cat in categories])
    if len(np.unique(binary_labels)) < 2: return 0.0
    sample_scores = silhouette_samples(X, binary_labels, metric='cosine')
    return np.mean(sample_scores[binary_labels == 1])

def draw_focus_hull(ax, points, color, alpha=0.15):
    if len(points) >= 3:
        try:
            hull = ConvexHull(points)
            ax.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color, alpha=alpha)
            ax.plot(points[hull.vertices, 0], points[hull.vertices, 1], color=color, lw=2.5, alpha=alpha+0.5)
        except Exception: pass

def plot_binary_ax(ax, X_2d, words, categories, focus_cat, title):
    is_target = (categories == focus_cat)
    target_points = X_2d[is_target]
    
    for i, word in enumerate(words):
        if not is_target[i]:
            x, y = X_2d[i, 0], X_2d[i, 1]
            ax.scatter(x, y, color='#b0b0b0', s=30, alpha=0.3, zorder=1)
            ax.text(x + 0.5, y + 0.5, word, fontsize=6, alpha=0.2, color='gray', zorder=2)

    draw_focus_hull(ax, target_points, color=COLOR_MAP.get(focus_cat, '#d62728'), alpha=0.15)
    for i, word in enumerate(words):
        if is_target[i]:
            x, y = X_2d[i, 0], X_2d[i, 1]
            ax.scatter(x, y, color=COLOR_MAP.get(focus_cat, '#d62728'), s=80, alpha=0.9, edgecolors='white', linewidth=1, zorder=5)
            ax.text(x + 0.5, y + 0.5, word, fontsize=10, fontweight='bold', alpha=0.9, color='black', zorder=6)
            
    ax.set_title(title, fontsize=14, pad=15, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)

def run_cpu_stable_eval(fst_pt: str, std_pt: str, out_dir: str, draw_plots: bool, top_k_plots: int):
    print(f"📦 Loading OV Tensors from {fst_pt} and {std_pt}...")
    fst_data = torch.load(fst_pt)
    std_data = torch.load(std_pt)
    
    words = np.array(fst_data["words"])
    categories = np.array(fst_data["categories"])
    unique_cats = [c for c in set(categories) if c != "Other"]
    
    num_layers, num_heads = 12, 32
    table_records = []
    
    # 用字典保存每个 Tag 在每层的得分，方便稍后挑出 Top K 画图
    # 结构: scores_dict[cat][layer][head] = score
    scores_dict = {cat: {l: {} for l in range(num_layers)} for cat in unique_cats}
    std_scores_dict = {cat: {l: {} for l in range(num_layers)} for cat in unique_cats}

    print("\n🚀 Phase 1: Pure Mathematical Quantification (Extremely Fast)...")
    
    for layer in tqdm(range(num_layers), desc="Scoring Heads"):
        for h in range(num_heads):
            key = f"L{layer:02d}_H{h:02d}"
            if key not in fst_data["ov_tensors"] or key not in std_data["ov_tensors"]:
                continue
                
            fst_X = fst_data["ov_tensors"][key].float().numpy()
            std_X = std_data["ov_tensors"][key].float().numpy()
            
            for cat in unique_cats:
                s_fst = get_target_silhouette_score(fst_X, categories, cat)
                s_std = get_target_silhouette_score(std_X, categories, cat)
                scores_dict[cat][layer][h] = s_fst
                std_scores_dict[cat][layer][h] = s_std

    # 汇总 CSV 表格数据
    for layer in range(num_layers):
        for cat in unique_cats:
            fst_vals = list(scores_dict[cat][layer].values())
            std_vals = list(std_scores_dict[cat][layer].values())
            if not fst_vals: continue
            
            fst_mean = np.mean(fst_vals)
            std_mean = np.mean(std_vals)
            table_records.append({
                "Tag": cat, "Layer (FST)": f"L{layer:02d}",
                "FST Mean Score": fst_mean, "Std Mean Score": std_mean,
                "Diff (FST - Std)": fst_mean - std_mean
            })

    df = pd.DataFrame(table_records)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "global_clustering_scores.csv")
    df.to_csv(csv_path, index=False)
    
    # ==========================================
    # Phase 2: Targeted Memory-Safe Rendering
    # ==========================================
    if draw_plots:
        print("\n🎨 Phase 2: CPU Memory-Safe Sequential Rendering...")
        
        # 统计哪些头需要画图
        heads_to_plot = {l: set() for l in range(num_layers)}
        for cat in unique_cats:
            for layer in range(num_layers):
                head_scores = scores_dict[cat][layer]
                if not head_scores: continue
                # 选出该类目下该层得分最高的 K 个头
                top_h = sorted(head_scores.keys(), key=lambda k: head_scores[k], reverse=True)[:top_k_plots]
                heads_to_plot[layer].update(top_h)
        
        for layer in tqdm(range(num_layers), desc="Rendering by Layer"):
            std_physical_layer = layer * 2 + 1
            
            for h in heads_to_plot[layer]:
                key = f"L{layer:02d}_H{h:02d}"
                fst_X = fst_data["ov_tensors"][key].float().numpy()
                std_X = std_data["ov_tensors"][key].float().numpy()
                
                # 【巨型优化】：每个头只算 1 次 TSNE！
                tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
                fst_2d = tsne.fit_transform(fst_X)
                std_2d = tsne.fit_transform(std_X)
                
                # 为该头所属的 top_k Tag 画图
                for cat in unique_cats:
                    # 只有当这个头是这个 Tag 的 top_k 时才画（或者当 top_k 很大时全画）
                    if h in sorted(scores_dict[cat][layer].keys(), key=lambda k: scores_dict[cat][layer][k], reverse=True)[:top_k_plots]:
                        
                        plot_dir = os.path.join(out_dir, "plots", f"Layer_{layer:02d}", cat)
                        os.makedirs(plot_dir, exist_ok=True)
                        out_path = os.path.join(plot_dir, f"compare_H{h:02d}.png")
                        
                        # 【巨型优化】：使用无状态 Agg 引擎，绝不爆内存
                        fig = Figure(figsize=(18, 8))
                        canvas = FigureCanvas(fig)
                        axes = fig.subplots(1, 2)
                        
                        plot_binary_ax(axes[0], fst_2d, words, categories, cat, 
                                       f"FST Predictive L{layer:02d}-H{h:02d} (Score: {scores_dict[cat][layer][h]:.3f})")
                        plot_binary_ax(axes[1], std_2d, words, categories, cat, 
                                       f"Standard Physical L{std_physical_layer:02d}-H{h:02d} (Score: {std_scores_dict[cat][layer][h]:.3f})")
                        
                        fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
                        # 无需 close，变量失效后自动释放物理内存

    print("\n" + "="*80)
    for cat in unique_cats:
        cat_df = df[df["Tag"] == cat].copy()
        cat_df = cat_df.drop(columns=["Tag"]).set_index("Layer (FST)")
        cat_df["FST Mean Score"] = cat_df["FST Mean Score"].apply(lambda x: f"{x:.4f}")
        cat_df["Std Mean Score"] = cat_df["Std Mean Score"].apply(lambda x: f"{x:.4f}")
        cat_df["Diff (FST - Std)"] = cat_df["Diff (FST - Std)"].apply(lambda x: f"+{x:.4f} 🚀" if x > 0 else f"{x:.4f}")
        print(f"\n🏷️  TARGET CATEGORY: {cat.upper()}")
        print("-" * 55)
        print(cat_df.to_markdown())
        
    print("\n" + "="*80)
    print(f"✅ Safe & Stable processing complete. CSV at: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive Quantitative Evaluation")
    
    # 【修改点】：默认指向 data/ 文件夹中的最新文件
    parser.add_argument("--fst_pt", type=str, default="data/fst_ov.pt")
    parser.add_argument("--std_pt", type=str, default="data/std_transformer_ov.pt")
    parser.add_argument("--out_dir", type=str, default="comprehensive_results")
    
    parser.add_argument("--no_plots", action="store_true", help="Skip plotting")
    parser.add_argument("--top_k_plots", type=int, default=3, help="Top K heads to plot")
    args = parser.parse_args()
    
    run_cpu_stable_eval(args.fst_pt, args.std_pt, args.out_dir, not args.no_plots, args.top_k_plots)