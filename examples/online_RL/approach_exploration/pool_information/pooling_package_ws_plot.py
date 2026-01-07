import json
import numpy as np
import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import os

# ===== 1. 工具函数与清洗逻辑 =====
def clean_answer(ans: str) -> str:
    if not ans: return ans
    ans = re.sub(r"\\(?:,|;|:|!|quad|qquad|enspace|,| )", "", ans)
    ans = re.sub(r"\s+", "", ans)
    ans = ans.replace(r"\dfrac", r"\frac")
    ans = re.sub(r"\b[a-zA-Z]\s*=", "", ans)
    ans = re.sub(r",+", ",", ans)
    ans = ans.strip(",")
    return ans

def load_ground_truth(path):
    gt = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                item = json.loads(line)
                gt[i] = clean_answer(item.get("answer", "")) #
    return gt

# ===== 2. 核心分析逻辑：支持参数化筛选 =====
def get_ranks_with_params(traces, gt_val, percentile, num_calibration=256):
    if len(traces) < num_calibration:
        return {"MaxC": 999, "SumC": 999}
    
    # 计算动态阈值 s
    calibration_traces = traces[:num_calibration]
    lowest_confs = [min(t['group_confidence']) for t in calibration_traces if t.get('group_confidence')]
    if not lowest_confs: return {"MaxC": 999, "SumC": 999}
    
    s = np.percentile(lowest_confs, percentile) # 根据输入参数确定强度

    # 模拟 Online Screening 过滤
    # 这里包含去噪逻辑：剔除 min_conf < s 的低质量路径
    good_traces = [t for t in traces if min(t.get('group_confidence', [0])) >= s]
    
    if not good_traces:
        return {"MaxC": 999, "SumC": 999}

    # 聚合统计：仅保留 MaxC 和 SumC
    stats = defaultdict(lambda: {'max_conf': -1.0, 'sum_conf': 0.0})
    for t in good_traces:
        ans = clean_answer(t.get("answer", ""))
        if not ans: continue
        conf = min(t.get('group_confidence', [0]))
        
        stats[ans]['max_conf'] = max(stats[ans]['max_conf'], conf)
        stats[ans]['sum_conf'] += conf

    unique_answers = list(stats.keys())
    
    # 排序：MaxC 倾向于发现高质量单门路径，SumC 是频次与质量的折中
    maxc_rank = sorted(unique_answers, key=lambda x: stats[x]['max_conf'], reverse=True)
    sumc_rank = sorted(unique_answers, key=lambda x: stats[x]['sum_conf'], reverse=True)

    def get_pos(rank_list, val):
        try: return rank_list.index(val) + 1
        except ValueError: return 999

    return {
        "MaxC": get_pos(maxc_rank, gt_val),
        "SumC": get_pos(sumc_rank, gt_val)
    }

# ===== 3. 热力图生成函数 =====
def plot_heatmaps(results_df, orioutdir, dataname):
    metrics = ["MaxC", "SumC"]
    for m in metrics:
        # 准备透视表：行为 Percentile，列为 N
        pivot_data = results_df[results_df['Metric'] == m].pivot(
            index="Percentile", columns="N", values="Recall"
        )
        
        # 绘图设置
        plt.clf()
        sns.heatmap(pivot_data, annot=True, fmt=".2%", cmap="YlGnBu", cbar_kws={'label': 'Recall'})
        plt.title(f"Recall @ N Analysis ({m} Metric)\n(Efficiency vs. Performance)")
        plt.xlabel("Candidate Pool Size (N)")
        plt.ylabel("Online Screening Percentile (P)")
        
        # 拼接保存路径
        outdir=os.path.join(orioutdir, f"{m.lower()}")
        os.makedirs(outdir, exist_ok=True)
        file_name = f"heatmap_{dataname}.png"
        plt.savefig(os.path.join(outdir, file_name), bbox_inches='tight')
        print(f"Heatmap for {m} saved to {file_name}")

# ===== 4. 主程序：网格扫描 =====
def main():
    parser = argparse.ArgumentParser(description="Grid Sweep for Pooling Parameters")
    parser.add_argument("--Ninit", type=int, default=16, help="Warmup")
    args = parser.parse_args()
    Ninit = args.Ninit
    
    datanames = ["aime_2024", "aime_2025", "brumo_2025", "hmmt_2025"]  
    for dataname in datanames:
        gt_path = Path(f"/home/yz54720/Projects/Method/deepconf/data/raw/{dataname}.jsonl")

        gt = load_ground_truth(gt_path)
        trace_path = Path(f"/home/yz54720/Projects/Method/deepconf/data/processed/{dataname}/traces")
        
        # 定义扫描范围
        percentiles = [0, 10, 30, 50, 70, 80, 90]
        Ns = [2, 3, 4, 5, 6, 7]
        metrics = ["MaxC", "SumC"]

        # 预加载所有数据到内存以加快扫描速度
        all_data = {}
        for qid in gt.keys():
            file = trace_path / f"{dataname}_{qid}_full.jsonl"
            if file.exists():
                with open(file, "r") as f:
                    all_data[qid] = [json.loads(line) for line in f if line.strip()]

        sweep_results = []

        print("Starting Grid Sweep...")
        for p in percentiles:
            # 对每个百分位运行一次全量分析
            hits = defaultdict(lambda: defaultdict(int)) # metric -> n -> count
            total_q = 0
            
            for qid, traces in all_data.items():
                total_q += 1
                ranks = get_ranks_with_params(traces, gt[qid], p, num_calibration=Ninit)
                
                for m in metrics:
                    r = ranks[m]
                    for n in Ns:
                        if r <= n:
                            hits[m][n] += 1
            
            # 记录结果
            for m in metrics:
                for n in Ns:
                    sweep_results.append({
                        "Percentile": p,
                        "N": n,
                        "Metric": m,
                        "Recall": hits[m][n] / total_q if total_q > 0 else 0
                    })
            print(f"Completed analysis for Percentile P={p}")

        # 绘图与展示
        df = pd.DataFrame(sweep_results)
        outdir = Path(f"/home/yz54720/Projects/Method/deepconf/data/calibration/pooling_package_plots/Ninit_{Ninit}")  
        plot_heatmaps(df, outdir, dataname)
        
        # 打印简表供快速参考
        print("\nSummary Results (MaxC):")
        print(df[df['Metric'] == 'MaxC'].pivot(index="Percentile", columns="N", values="Recall"))

if __name__ == "__main__":
    main()