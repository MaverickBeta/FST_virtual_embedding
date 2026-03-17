import os
# 【核弹级防死锁补丁】：必须在导入 numpy 和 sklearn 之前执行！
# 强行关闭底层 BLAS/OpenMP 的多线程，防止它们在多进程环境下互相锁死
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import numpy as np
import multiprocessing as mp
import matplotlib
# 强制无头渲染
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from concurrent.futures import ProcessPoolExecutor, as_completed

# 调色板
COLOR_MAP = {
    "Noun": "blue", "Verb": "red", "Adjective": "green", 
    "Adverb": "purple", "Other": "gray", "Top_Words": "blue"
}

def plot_single_head_task(args):
    """独立的子进程工作函数"""
    # 为了保证子进程环境纯净，在进程内部再次声明后端
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    layer, head, vectors, words, groups, head_dir = args
    
    X = np.array(vectors)
    
    # t-SNE 降维
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_reduced = tsne.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for i, word in enumerate(words):
        x_coord = X_reduced[i, 0]
        y_coord = X_reduced[i, 1]
        group = groups[i]
        color = COLOR_MAP.get(group, "black")
        
        ax.scatter(x_coord, y_coord, color=color, s=80, alpha=0.6, edgecolors='w')
        ax.text(x_coord + 0.01, y_coord + 0.01, word, fontsize=8, alpha=0.8, color='black')

    ax.set_title(f"t-SNE of {head.capitalize()} - {layer.capitalize()}", fontsize=16, pad=15)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.grid(True, linestyle='--', alpha=0.3)
    
    unique_groups = set(groups)
    handles = [matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_MAP.get(l, "black"), markersize=10, label=l) 
               for l in unique_groups]
    ax.legend(handles=handles, title="POS Tags", loc='best')

    layer_num = layer.split('_')[1].zfill(2)
    head_num = head.split('_')[1].zfill(2)
    output_img = os.path.join(head_dir, f"layer_{layer_num}_head_{head_num}.png")
    
    fig.savefig(output_img, dpi=150, bbox_inches='tight')
    plt.close(fig) # 极其重要：释放内存
    
    return f"Layer {layer_num} Head {head_num}"

def batch_visualize_parallel(
    jsonl_path: str = "structured_heads_v.jsonl", 
    output_dir: str = "tsne_results"
):
    print(f"Loading data from {jsonl_path}...")
    os.makedirs(output_dir, exist_ok=True)
    
    data_cache = {}
    words_found = []
    groups_found = []
    
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                words_found.append(data["word"])
                groups_found.append(data.get("category", "Other"))
                
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

    # 构建任务列表
    tasks = []
    for layer in data_cache.keys():
        for head in data_cache[layer].keys():
            head_dir = os.path.join(output_dir, f"head_{head.split('_')[1].zfill(2)}")
            os.makedirs(head_dir, exist_ok=True)
            vectors = data_cache[layer][head]
            tasks.append((layer, head, vectors, words_found, groups_found, head_dir))

    total_tasks = len(tasks)
    print(f"Loaded {len(words_found)} words. Total tasks: {total_tasks}. Firing up CPU cores...")

    # 利用所有的 16 个核心！
    max_cores = 16 
    
    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        futures = [executor.submit(plot_single_head_task, task) for task in tasks]
        
        # 实时高频更新进度
        completed = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                completed += 1
                # 每次完成都打印，让你安心！
                print(f"[{completed}/{total_tasks}] ✅ Saved: {result}", flush=True)
            except Exception as e:
                print(f"❌ Task failed: {e}")

    print(f"\n🎉 Task complete! All t-SNE images organized in: '{output_dir}/'")

if __name__ == "__main__":
    # 强制在 Linux 下使用安全的 spawn 模式启动多进程
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    batch_visualize_parallel()