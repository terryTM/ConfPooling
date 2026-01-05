import json
import random
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import re
import argparse

# ===== 工具函数 (保持与原代码一致) =====

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
    ans = ans.strip(",")
    return ans

def data_loading_all(base_dir, dataname):
    """加载所有原始 traces，不进行置信度筛选"""
    results = defaultdict(list)
    # 匹配文件名如 aime_2024_0_...jsonl
    for jsonl_path in base_dir.glob(f"{dataname}_*.jsonl"):
        qid_match = re.search(rf"{dataname}_(\d+)_", jsonl_path.name)
        if not qid_match:
            continue
        qid = int(qid_match.group(1))
        traces = load_traces(jsonl_path)
        for item in traces:
            ans = clean_latex_answer(item.get("answer"))
            if ans:
                results[qid].append(ans)
    return results

# ===== 核心投票逻辑 =====

def main():
    parser = argparse.ArgumentParser(description="Run standard Majority Voting baseline.")
    # parser.add_argument("--dataname", type=str, required=True, help="e.g., aime_2024")
    parser.add_argument("--budget", type=int, default=256, help="Number of traces to sample per question")
    args = parser.parse_args()

    # ===== 路径配置 =====
    all_dataname=["aime_2024", "aime_2025", "brumo_2025", "hmmt_2025"]
    for dataname in all_dataname:
        BASE_PROJECT_DIR = Path("/home/yz54720/Projects/Method/deepconf/data") 
        TRACES_DIR = BASE_PROJECT_DIR / "processed" / dataname / "traces"
        RAW_DATA_PATH = BASE_PROJECT_DIR / "raw" / f"{dataname}.jsonl"

        print(f"--- Running MV for {dataname} (Budget N={args.budget}) ---")

        # 1. 加载真值
        gt = load_ground_truth(RAW_DATA_PATH)
        
        # 2. 加载所有轨迹 (不做筛选)
        all_traces = data_loading_all(TRACES_DIR, dataname)
        
        correct_cnt = 0
        total_questions = len(gt)
        random.seed(42) # 保证采样可重复

        for qid in range(total_questions):
            ground_truth = gt.get(qid)
            traces = all_traces.get(qid, [])
            
            if not traces:
                continue

            # 3. 按照 Budget N 进行采样
            # 如果原始 trace 不足 N，则取全部；如果超过 N，则随机抽取 N 个
            sample_size = min(len(traces), args.budget)
            sampled_answers = random.sample(traces, sample_size)
            
            # 4. 执行不加权的 Majority Voting
            if sampled_answers:
                counts = Counter(sampled_answers)
                # 获取票数最多的答案
                voted_answer = counts.most_common(1)[0][0]
                
                if voted_answer == ground_truth:
                    correct_cnt += 1
                else:
                    # 可选：打印错误详情
                    # print(f"QID {qid} Wrong: GT={ground_truth!r}, Voted={voted_answer!r}")
                    pass

        accuracy = correct_cnt / total_questions
        print(f"\nFinal Result for {dataname}:")
        print(f"Accuracy: {correct_cnt}/{total_questions} = {accuracy:.4f}")

if __name__ == "__main__":
    main()