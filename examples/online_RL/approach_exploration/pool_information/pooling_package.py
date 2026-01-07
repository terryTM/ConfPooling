import json
import numpy as np
import argparse
import re
from pathlib import Path
from collections import defaultdict, Counter

# ===== 1. 真值加载与清洗 (引用自 voting.py) =====
def clean_answer(ans: str) -> str:
    if not ans: return ""
    return re.sub(r"\s+", "", str(ans))
def clean_latex_answer(ans: str) -> str:
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
                # 使用与 voting.py 一致的逻辑
                gt[i] = clean_latex_answer(item.get("answer", ""))
    return gt

# ===== 2. 筛选逻辑 (引用自 pool_information.py) =====
def get_predicted_good(traces, num_calibration=64):
    if len(traces) < num_calibration:
        return traces
    
    # 计算阈值 s (90th percentile - 严格模式)
    calibration_traces = traces[:num_calibration]
    lowest_confs = [min(t['group_confidence']) for t in calibration_traces if t.get('group_confidence')]
    if not lowest_confs: return traces
    s = np.percentile(lowest_confs, 90)
    # s = min(lowest_confs)

    predicted_good = []
    # 筛选后续 traces
    for trace in traces[num_calibration:]:
        conf_curve = trace.get('group_confidence', [])
        if not any(v < s for v in conf_curve):
            predicted_good.append(trace)
    # 保留 calibration 中达标的
    for trace in calibration_traces:
        if min(trace.get('group_confidence', [0])) >= s:
            predicted_good.append(trace)
    return predicted_good

# ===== 3. 分析核心函数 =====
def analyze_question(qid, traces, gt_val):
    # good_traces = get_predicted_good(traces)      
    good_traces = traces  # 不进行筛选，使用全部数据进行投票
    if not good_traces:
        return None

    # 聚合统计
    stats = defaultdict(lambda: {'count': 0, 'max_conf': -1.0, 'sum_conf': 0.0})
    for t in good_traces:
        ans = clean_latex_answer(t.get("answer", ""))
        if not ans: continue
        conf = min(t.get('group_confidence', [0]))
        
        stats[ans]['count'] += 1
        stats[ans]['max_conf'] = max(stats[ans]['max_conf'], conf)
        stats[ans]['sum_conf'] += conf

    unique_answers = list(stats.keys())
    
    # 三种排序方法
    ranks = {
        "Freq": sorted(unique_answers, key=lambda x: stats[x]['count'], reverse=True),
        "MaxC": sorted(unique_answers, key=lambda x: stats[x]['max_conf'], reverse=True),
        "SumC": sorted(unique_answers, key=lambda x: stats[x]['sum_conf'], reverse=True)
    }

    res = {"unique_count": len(unique_answers), "ranks": {}}
    for method, sorted_list in ranks.items():
        try:
            pos = sorted_list.index(gt_val) + 1
        except ValueError:
            pos = 999 # 未包含
        res["ranks"][method] = pos
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, required=True)
    args = parser.parse_args()

    gt_path = Path(f"/home/yz54720/Projects/Method/deepconf/data/raw/{args.dataname}.jsonl")
    gt = load_ground_truth(gt_path)
    trace_path = Path(f"/home/yz54720/Projects/Method/deepconf/data/processed/{args.dataname}/traces")
    
    summary = defaultdict(lambda: defaultdict(int)) # method -> N -> count
    total_valid_q = 0
    Ns = [5,6,7,8,9,10]

    print(f"{'QID':<6} | {'TotalAns':<10} | {'Freq Rank':<10} | {'MaxC Rank':<10} | {'SumC Rank':<10}")
    print("-" * 60)

    for qid, gt_val in gt.items():
        file = trace_path / f"{args.dataname}_{qid}_full.jsonl"
        if not file.exists(): continue
        
        with open(file, "r") as f:
            traces = [json.loads(line) for line in f if line.strip()]
        
        result = analyze_question(qid, traces, gt_val)
        if not result: continue
        
        total_valid_q += 1
        print(f"{qid:<6} | {result['unique_count']:<10} | {result['ranks']['Freq']:<10} | {result['ranks']['MaxC']:<10} | {result['ranks']['SumC']:<10}")
        
        for method in ["Freq", "MaxC", "SumC"]:
            r = result["ranks"][method]
            for n in Ns:
                if r <= n:
                    summary[method][n] += 1

    print("\n" + "="*40)
    print(f"SUMMARY (Total Questions: {total_valid_q})")
    print(f"{'Method':<10} | " + " | ".join([f"Top {n}" for n in Ns]))
    for method in ["Freq", "MaxC", "SumC"]:
        row = [f"{summary[method][n]/total_valid_q:.2%}" for n in Ns]
        print(f"{method:<10} | " + " | ".join(row))
    print("="*40)

if __name__ == "__main__":
    main()