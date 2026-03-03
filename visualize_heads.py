import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def batch_visualize_all_heads(
    jsonl_path: str = "structured_heads_v.jsonl", 
    output_dir: str = "pca_results"
):
    print(f"Loading data from {jsonl_path}...")
    
    # 自动建立总输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义你的词汇分组
    word_groups = {
        "king": "Royalty", "queen": "Royalty",
        "run": "Verbs", "think": "Verbs",
        "happy": "Emotions", "sad": "Emotions",
        "and": "Function", "the": "Function", "because": "Function",
    }
    
    # 为不同组分配颜色
    color_map = {
        "Fruits": "red", "Royalty": "gold", "Verbs": "blue",
        "Emotions": "purple", "Function": "gray", "Names": "green"
    }

    # ==========================================
    # 1. 读取并重组数据
    # ==========================================
    data_cache = {}
    words_found = []
    groups_found = []
    
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                word = data["word"]
                words_found.append(word)
                groups_found.append(word_groups.get(word, "Other"))
                
                for layer, heads_dict in data["heads_v_by_layer"].items():
                    if layer not in data_cache:
                        data_cache[layer] = {}
                    for head, vec in heads_dict.items():
                        if head not in data_cache[layer]:
                            data_cache[layer][head] = []
                        data_cache[layer][head].append(vec)
    except FileNotFoundError:
        print(f"Error: Could not find {jsonl_path}.")
        return

    num_words = len(words_found)
    print(f"Successfully loaded {num_words} words. Starting batch PCA plotting...")

    # ==========================================
    # 2. 遍历绘图并按 Head 分文件夹保存
    # ==========================================
    for layer in data_cache.keys():
        for head in data_cache[layer].keys():
            # --- 【关键修改点 1: 创建 Head 子文件夹】 ---
            # 为每个 head 创建独立文件夹，例如 pca_results/head_0
            head_dir = os.path.join(output_dir, f"head_{head}")
            os.makedirs(head_dir, exist_ok=True)
            
            vectors = data_cache[layer][head]
            X = np.array(vectors)
            
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(X)
            explained_variance = pca.explained_variance_ratio_
            
            plt.figure(figsize=(10, 8))
            
            for i, word in enumerate(words_found):
                x_coord = X_reduced[i, 0]
                y_coord = X_reduced[i, 1]
                group = groups_found[i]
                color = color_map.get(group, "black")
                
                plt.scatter(x_coord, y_coord, color=color, s=120, alpha=0.8, edgecolors='w')
                plt.text(x_coord + 0.02, y_coord + 0.02, word, fontsize=13, color='black', weight='bold')

            plt.title(f"PCA of Head {head} - {layer}", fontsize=16, pad=15)
            plt.xlabel(f"PC1 ({explained_variance[0]*100:.1f}%)")
            plt.ylabel(f"PC2 ({explained_variance[1]*100:.1f}%)")
            plt.grid(True, linestyle='--', alpha=0.5)
            
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=l) 
                       for l, c in color_map.items()]
            plt.legend(handles=handles, title="Semantic Groups", loc='best')

            # --- 【关键修改点 2: 修改保存路径】 ---
            # 文件名改为按 layer 排序，存入对应的 head 文件夹
            # 使用 zfill(2) 可以让 layer_2 排在 layer_10 前面，方便文件夹内自动排序
            layer_str = str(layer).zfill(2) 
            output_img = os.path.join(head_dir, f"layer_{layer_str}_{head}.png")
            
            plt.savefig(output_img, dpi=150, bbox_inches='tight')
            plt.close()
            
        print(f"✅ Processed all heads for {layer}.")
        
    print(f"\n🎉 Task complete! Images organized by Head in: '{output_dir}/'")

if __name__ == "__main__":
    batch_visualize_all_heads()