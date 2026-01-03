import json
import random
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ===== 1. 工具函数 (与你之前的代码完全一致) =====

def load_traces(file_path):
    traces = []
    if not Path(file_path).exists(): return traces
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try: traces.append(json.loads(line))
                except: continue
    return traces

def load_ground_truth(path):
    gt = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                item = json.loads(line)
                gt[i] = str(item.get("answer", "")).strip()
    return gt

def clean_latex_answer(ans: str) -> str:
    if not ans: return ans
    ans = re.sub(r"\\(?:,|;|:|!|quad|qquad|enspace|,| )", "", ans)
    ans = re.sub(r"\s+", "", ans)
    ans = ans.replace(r"\dfrac", r"\frac")
    ans = re.sub(r"\b[a-zA-Z]\s*=", "", ans)
    ans = re.sub(r",+", ",", ans)
    return ans.strip(",")

# ===== 2. Online 筛选逻辑 (Top 10% 策略) =====

def analyze_qid(qid, traces, ground_truth, num_calibration=16, seed=13):
    random.seed(seed)
    if len(traces) < num_calibration: return None
    
    # 阈值计算 (与 DeepConf-low 逻辑一致)
    calibration_traces = random.sample(traces, num_calibration)
    lowest_confs = [min(t['group_confidence']) for t in calibration_traces if t.get('group_confidence')]
    if not lowest_confs: return None
    s_strict = np.percentile(lowest_confs, 90) # 90分位点作为 Top 10% 的生死线
    
    # 筛选
    good_traces = []
    bad_traces = []
    for t in traces:
        is_bad = any(c < s_strict for c in t.get('group_confidence', []))
        if is_bad: bad_traces.append(t)
        else: good_traces.append(t)
        
    # 计算 MV 结果 (全量 256 条，不加权)
    all_ans = [clean_latex_answer(t.get('answer')) for t in traces]
    mv_voted = Counter(all_ans).most_common(1)[0][0] if all_ans else None
    mv_correct = (mv_voted == ground_truth)
    
    # 计算 Online Baseline 结果 (Survivors, 加权投票)
    online_score = {}
    for t in good_traces:
        ans = clean_latex_answer(t.get('answer'))
        conf = np.min(t.get('group_confidence', np.nan))
        online_score[ans] = online_score.get(ans, 0) + conf
    online_voted = max(online_score, key=online_score.get) if online_score else None
    online_correct = (online_voted == ground_truth)
    
    return {
        'mv_correct': mv_correct,
        'online_correct': online_correct,
        'good': good_traces,
        'bad': bad_traces,
        'threshold': s_strict,
        'mv_voted': mv_voted,
        'online_voted': online_voted
    }

# ===== 3. 主循环与绘图 =====

def main():
    dataname = "brumo_2025"
    base_dir = Path("/home/yz54720/Projects/Method/deepconf/data")
    traces_dir = base_dir / "processed" / dataname / "traces"
    gt_path = base_dir / "raw" / f"{dataname}.jsonl"
    
    gt = load_ground_truth(gt_path)
    print(f"Starting analysis for {dataname}...")

    for qid in range(len(gt)):
        trace_file = next(traces_dir.glob(f"{dataname}_{qid}_*.jsonl"), None)
        traces = load_traces(trace_file)
        if not traces: continue
        
        res = analyze_qid(qid, traces, gt[qid])
        if not res: continue
        
        # 找出 MV 对但 Online 错的题目
        if res['mv_correct'] and not res['online_correct']:
            print(f"Found Target QID: {qid} (MV Correct, Online Wrong)")
            
            fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
            
            # 左图: 被筛掉的轨迹 (Bad)
            for t in res['bad']:
                color = 'tab:green' if clean_latex_answer(t.get('answer')) == gt[qid] else 'tab:red'
                axes[0].plot(t['group_confidence'], color=color, alpha=0.15, linewidth=1)
            axes[0].axhline(y=res['threshold'], color='blue', linestyle='--', label=f'Threshold: {res["threshold"]:.2f}')
            axes[0].set_title(f"QID {qid}: Screened Out (n={len(res['bad'])})\n(Dropped Correct Traces are in GREEN)")
            axes[0].set_ylabel("Confidence")
            
            # 右图: 留下的轨迹 (Survivors/Good)
            for t in res['good']:
                color = 'tab:green' if clean_latex_answer(t.get('answer')) == gt[qid] else 'tab:red'
                axes[1].plot(t['group_confidence'], color=color, alpha=0.4, linewidth=2)
            axes[1].axhline(y=res['threshold'], color='blue', linestyle='--', label=f'Threshold')
            axes[1].set_title(f"QID {qid}: Not Screened Out (n={len(res['good'])})\n(Voted Wrong: {res['online_voted']})")
            
            # 图例
            legend_elements = [
                Line2D([0], [0], color='tab:green', lw=2, label='Correct Trace'),
                Line2D([0], [0], color='tab:red', lw=2, label='Incorrect Trace'),
                Line2D([0], [0], color='blue', linestyle='--', label='Early-stop Threshold')
            ]
            axes[1].legend(handles=legend_elements, loc='lower right')
            
            plt.suptitle(f"Analysis of QID {qid} in BRUMO 2025: Why Online Baseline Failed", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f"brumo_failure_analysis_qid_{qid}.png", dpi=200)
            print(f"Saved plot: brumo_failure_analysis_qid_{qid}.png")
            plt.close()

if __name__ == "__main__":
    main()