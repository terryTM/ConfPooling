import json
import random
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import re
import numpy as np
import argparse

# ===== 工具函数 ====
import ast
# ===== online screening ======

def online_screening(traces, num_calibration=64, use_low_threshold=False):
   # True: 10% percentile (lenient), False: 90% percentile (strict)
    # TODO 要把这里改成pooling中同样的online集合来投票
    random.seed(13)

    # --- Calculate Threshold ---
    s = None
    if len(traces) >= num_calibration:
        calibration_traces = traces[:num_calibration]
        lowest_confs = [min(t['group_confidence']) for t in calibration_traces if t['group_confidence']]
        if lowest_confs:
            s_high = np.percentile(lowest_confs, 10)
            s_low = np.percentile(lowest_confs, 90)
            s = s_high if use_low_threshold else s_low

    if s is not None:

        predicted_good = []  # ✅ 收集未被截断的 trace
        predicted_bad = []

        for trace in traces[num_calibration:]:
            actual_is_correct = trace['is_correct']
            conf_curve = trace['group_confidence']

            stop_indices = np.where(np.array(conf_curve) < s)[0] if conf_curve else []
            predicted_as_bad = len(stop_indices) > 0

            # ✅ 保存分类结果
            if predicted_as_bad:
                predicted_bad.append(trace)
            else:
                predicted_good.append(trace)
    for trace in traces[:num_calibration]:
        if min(trace['group_confidence']) > s:
            predicted_good.append(trace)
    return predicted_good
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
    """只有pooling_information中提取followup和base_answer时使用，traces中的answer已经给定，是cleaned的"""
    """最理想的是统一适用data_creation中的extract函数"""
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
def extract_boxed_answer(text: str):
    """提取文本中最后一个 \\boxed{...} 的内容（支持任意嵌套 {}）并清洗"""
    if not text:
        return None

    results = []
    pos = 0
    while True:
        start = text.find(r'\boxed{', pos)
        if start == -1:
            break  # 没有更多了

        i = start + len(r'\boxed{')
        depth = 1
        content_chars = []

        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == '{':
                depth += 1
                content_chars.append(ch)
            elif ch == '}':
                depth -= 1
                if depth > 0:
                    content_chars.append(ch)
            else:
                content_chars.append(ch)
            i += 1

        if depth == 0:
            results.append(''.join(content_chars))
            pos = i  # 从上次结束后继续查找
        else:
            break  # 不完整，结束循环

    if not results:
        return None

    # 返回最后一个
    return clean_latex_answer(results[-1])
def data_loading(base_dir, dataname, is_followup=False):
    results = defaultdict(list)
    for jsonl_path in base_dir.glob(f"{dataname}_*.jsonl"):
        qid_match = re.search(rf"{dataname}_(\d+)_", jsonl_path.name)
        if not qid_match:
            continue
        qid = int(qid_match.group(1))
        traces = load_traces(jsonl_path)
        if is_followup:
            for item in traces:
                base_ans = clean_latex_answer(item.get("base_answer"))
                ans = extract_boxed_answer(item.get("trace_2", "")) 
                conf = np.min(item.get("group_confidences_2", np.nan))
                if ans and (base_ans is not None):
                    results[qid].append((base_ans, ans, conf))
        else:
            predicted_good = online_screening(traces)
            for item in predicted_good:
                ans = clean_latex_answer(item.get("answer"))
                # 算一个trace level的confidence
                conf = np.min(item.get("group_confidence", np.nan))
                if ans and (ans is not None):
                    results[qid].append((ans, conf))
    return results
def load_ground_truth(path):
    gt = {}
    for i, item in enumerate(load_traces(path)):
        gt[i] = str(item.get("answer", "")).strip()
        # 去掉每个字符串中的所有空格
        gt[i] = re.sub(r"\s+", "", gt[i])
    return gt

# ===== 主函数 =====
def main():
    parser = argparse.ArgumentParser(description="Generate trace dataset with full precision confidence.")
    parser.add_argument("--dataname", type=str, required=True)
    parser.add_argument("--version", type=str, default="3_newprompt")
    parser.add_argument("--ifcnt", type=str, default="False")
    args = parser.parse_args()
    print(f"dataname: {args.dataname}, version: {args.version}, ifcnt: {args.ifcnt}")
    # ===== 路径配置 =====
    POOLING_DATA_DIR = Path(f"/home/yz54720/Projects/Method/deepconf/data/processed/{args.dataname}/pool_information_v{args.version}")
    TRACES_DIR = Path(f"/home/yz54720/Projects/Method/deepconf/data/processed/{args.dataname}/traces")
    DATA_PATH = Path(f"/home/yz54720/Projects/Method/deepconf/data/raw/{args.dataname}.jsonl")
    OUTPUT_PATH = POOLING_DATA_DIR / "question_level_voting_summary.csv"
    gt = load_ground_truth(DATA_PATH)
    # loading traces, online screened
    traces_data = data_loading(TRACES_DIR, args.dataname, is_followup=False)
    # loading followup results
    followup_data = data_loading(POOLING_DATA_DIR, args.dataname, is_followup=True)
    # voting by qid
    baseline_correct_cnt = 0
    followup_correct_cnt = 0
    for qid in gt.keys():
        ground_truth = gt[qid]
        traces = traces_data.get(qid, [])
        followups = followup_data.get(qid, [])
        
        # traces voting
        trace_score_c = {}
        for ans, conf in traces:
            trace_score_c[ans] = trace_score_c.get(ans, 0) + conf

        # voting result
        # 去掉none再投票
        trace_score_c.pop(None, None)
        if trace_score_c:
            baseline_voted_answer = max(trace_score_c, key=trace_score_c.get)
        else:
            baseline_voted_answer = None
        if baseline_voted_answer == ground_truth:
            baseline_correct_cnt += 1
        # count answers
        counts = Counter([ans for ans, _ in traces])
        top_counts_sorted = sorted(set(counts.values()), reverse=True)
        threshold_values = top_counts_sorted[:min(5, len(top_counts_sorted))]
        top5_anc = {ans: cnt for ans, cnt in counts.items() if cnt in threshold_values}

        # followup voting
        followup_score_c = {}
        if gt[qid] not in [base_ans for base_ans, _, _ in followups]:
            print(f"⚠️ QID {qid}: ground truth {gt[qid]!r} not in follow-up base answers.")
        for base_ans, ans, conf in followups:
            if base_ans in top5_anc:
                cnt = top5_anc[base_ans]
            else: 
                cnt = 1
                print(f"⚠️ QID {qid}: follow-up base answer {base_ans!r} not in top5 of traces.")
            followup_score_c[ans] = followup_score_c.get(ans, 0) + conf * (cnt if args.ifcnt == "True" else 1)
        # voting result
        if followup_score_c:
            voted_answer = max(followup_score_c, key=followup_score_c.get)
        else:
            voted_answer = None
        if voted_answer == ground_truth:
            followup_correct_cnt += 1
        if (baseline_voted_answer != ground_truth) or (voted_answer != ground_truth):
            print(f"❌ QID {qid}: GT={ground_truth!r}, Baseline Vote={baseline_voted_answer!r}, Follow-up Vote={voted_answer!r}")
            # details
            for base_ans, ans, conf in followups:
                print(f"    Base Ans: {base_ans!r}, Follow-up Ans: {ans!r}, Conf: {conf:.4f}, Weight: {top5_anc.get(base_ans, 1)}")
            print(f"followup_score_c: {followup_score_c}")
    total_questions = len(gt)
    print(f"Baseline Accuracy: {baseline_correct_cnt}/{total_questions} = {baseline_correct_cnt/total_questions:.4f}")
    print(f"Follow-up Accuracy: {followup_correct_cnt}/{total_questions} = {followup_correct_cnt/total_questions:.4f}")

if __name__ == "__main__":
    main()
