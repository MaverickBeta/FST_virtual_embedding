import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_global_trends(csv_path: str, out_dir: str):
    print(f"📦 Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Error: {csv_path} not found.")
        return

    # 设置学术风极简主题
    sns.set_theme(style="whitegrid", context="paper")
    
    unique_tags = df["Tag"].unique()
    
    # 获取全局 Y 轴的最大最小值，为了让 4 张图放在同一个绝对尺度下比较
    global_y_min = min(df["FST Mean Score"].min(), df["Std Mean Score"].min()) - 0.01
    global_y_max = max(df["FST Mean Score"].max(), df["Std Mean Score"].max()) + 0.02
    
    # 创建 2x2 的画板
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, tag in enumerate(unique_tags):
        ax = axes[idx]
        subset = df[df["Tag"] == tag].copy()
        
        # 提取 X 轴和 Y 轴数据
        layers = subset["Layer (FST)"].tolist()
        fst_scores = subset["FST Mean Score"].tolist()
        std_scores = subset["Std Mean Score"].tolist()
        
        # 画折线图 (带数据点 Marker)
        ax.plot(layers, fst_scores, marker='o', markersize=8, linewidth=3, 
                color='#d62728', label='FST (Predictive)', alpha=0.9)
        ax.plot(layers, std_scores, marker='s', markersize=8, linewidth=3, 
                color='#1f77b4', label='Standard', alpha=0.9)
        
        # 图表修饰
        ax.set_title(f"Target: {tag}", fontsize=16, fontweight='bold', pad=10)
        ax.set_xlabel("Model Layer (Aligned)", fontsize=12)
        ax.set_ylabel("Mean Silhouette Score", fontsize=12)
        
        # 统一 Y 轴刻度范围
        ax.set_ylim(global_y_min, global_y_max)
        
        # 添加 0 分参考线
        ax.axhline(0, color='black', linewidth=1.5, linestyle='--', alpha=0.6)
        
        # 隐藏右边和上边的边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 旋转 X 轴标签防止拥挤
        ax.tick_params(axis='x', rotation=45)
        ax.legend(fontsize=11, loc='upper left', frameon=True)

    # 调整子图间距
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle("Evolution of Pure Concept Clustering Across Layers (FST vs Standard)", 
                 fontsize=22, fontweight='bold', y=0.96)
    
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "layer_trend_comparison.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"🎉 Trend plot saved to: {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="comprehensive_results/global_clustering_scores.csv")
    parser.add_argument("--out_dir", type=str, default="comprehensive_results")
    args = parser.parse_args()
    
    plot_global_trends(args.csv, args.out_dir)