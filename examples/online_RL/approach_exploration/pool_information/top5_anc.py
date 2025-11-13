import os
import re
import json
import pandas as pd
from pathlib import Path
from collections import Counter
import random
import numpy as np

# ======= 配置 =======
ROOT = Path("~/Projects/Method/deepconf/data/processed").expanduser()
OUTPUT_CSV = ROOT / "summary_top5_answers.csv"
DATASETS = ["aime_2025", "brumo_2025", "aime_2024", 'hmmt_2025']  # 👈 可以自由加减

# ======= 工具函数 =======
def load_traces(file_path):
    traces = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return traces
def clean_latex_answer(ans: str) -> str:
    """清洗 LaTeX 表达式：去空格、去 a= 等、标准化 \\dfrac"""
    if not ans:
        return ans

    # 去掉 LaTeX 空格命令（\ , \quad, \qquad 等）
    ans = re.sub(r"\\(?:,|;|:|!|quad|qquad|enspace|,| )", "", ans)
    
    # 去掉所有普通空格
    ans = re.sub(r"\s+", "", ans)
    
    # 将 \dfrac 转为 \frac
    ans = ans.replace(r"\dfrac", r"\frac")
    
    # 去掉形如 a=、x=、y= 等赋值
    ans = re.sub(r"\b[a-zA-Z]\s*=", "", ans)
    
    # 去掉可能的多余逗号空位，比如 "2,,3" → "2,3"
    ans = re.sub(r",+", ",", ans)
    
    # 去掉首尾逗号
    ans = ans.strip(",")
    
    return ans
def get_top5_answers(traces):
    """统计每题 top5 答案"""
    answers = [clean_latex_answer(t.get("answer")) for t in traces if t.get("answer")]
    if not answers:
        return {}, 0

    counts = Counter(answers)
    if not counts:
        return {}, 0

    top_counts_sorted = sorted(set(counts.values()), reverse=True)
    threshold_values = top_counts_sorted[:min(5, len(top_counts_sorted))]
    top5_anc = [(ans, cnt) for ans, cnt in counts.items() if cnt in threshold_values]
    return top5_anc, len(answers)

# ======= 主逻辑 =======
def main():
    records = []

    for data_name in DATASETS:
        trace_dir = ROOT / data_name / "traces"
        if not trace_dir.exists():
            print(f"⚠️ Skipping {data_name}, path not found.")
            continue

        for file in sorted(trace_dir.glob(f"{data_name}_*_full.jsonl")):
            qid_match = re.search(rf"{data_name}_(\d+)_", file.name)
            if not qid_match:
                continue
            qid = int(qid_match.group(1))

            traces = load_traces(file)
            if not traces:
                continue
            NUM_CALIBRATION_TRACES = 16
            USE_LOW_THRESHOLD = False  # True: 10% percentile (lenient), False: 90% percentile (strict)
            random.seed(13)

            # --- Calculate Threshold ---
            s = None
            if len(traces) >= NUM_CALIBRATION_TRACES:
                calibration_traces = random.sample(traces, NUM_CALIBRATION_TRACES)
                lowest_confs = [min(t['group_confidence']) for t in calibration_traces if t['group_confidence']]
                if lowest_confs:
                    s_high = np.percentile(lowest_confs, 10)
                    s_low = np.percentile(lowest_confs, 90)
                    s = s_high if USE_LOW_THRESHOLD else s_low
                    
            if s is not None:

                predicted_good = []  # ✅ 收集未被截断的 trace
                predicted_bad = []

                for trace in traces:
                    actual_is_correct = trace['is_correct']
                    conf_curve = trace['group_confidence']

                    stop_indices = np.where(np.array(conf_curve) < s)[0] if conf_curve else []
                    predicted_as_bad = len(stop_indices) > 0

                    # ✅ 保存分类结果
                    if predicted_as_bad:
                        predicted_bad.append(trace)
                    else:
                        predicted_good.append(trace)
            top5_anc, total = get_top5_answers(predicted_good)
            if not top5_anc:
                continue

            records.append({
                "dataset": data_name,
                "question_id": qid,
                "top5_anc": top5_anc,
            })
            print(f"Processed {data_name} QID {qid}")

    # ======= 输出结果 =======
    df = pd.DataFrame(records).sort_values(["dataset", "question_id"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved summary to: {OUTPUT_CSV}")
    print(df.head(10))

if __name__ == "__main__":
    main()
