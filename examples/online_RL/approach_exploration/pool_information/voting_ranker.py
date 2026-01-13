import json
import random
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import re
import numpy as np
import argparse

# ===== 工具函数 ====

def clean_latex_answer(ans: str) -> str:
    if not ans: return ans
    ans = re.sub(r"\\(?:,|;|:|!|quad|qquad|enspace|,| )", "", ans)
    ans = re.sub(r"\s+", "", ans)
    ans = ans.replace(r"\dfrac", r"\frac")
    ans = re.sub(r"\b[a-zA-Z]\s*=", "", ans)
    ans = re.sub(r",+", ",", ans)
    ans = ans.strip(",")
    return ans

def extract_boxed_answer(text: str):
    """提取最后一个普通的 \boxed{ans}"""
    if not text: return None
    results = re.findall(r'\\boxed\{(.*)\}', text)
    if not results: return None
    return clean_latex_answer(results[-1])

def extract_ranking_boxed(text: str) -> list:
    """
    提取 \boxed{1:Ans1 > 2:Ans2 > ...} 中的排名
    返回: ['Ans1', 'Ans2', 'Ans3', 'Ans4']
    """
    if not text: return []
    # 匹配最后一个 boxed 内容
    match = re.search(r'\\boxed\{([^{}]*1:.*)\}', text)
    if not match:
        # 如果没找到排名格式，退而求其次找普通 boxed
        single = extract_boxed_answer(text)
        return [single] if single else []
    
    content = match.group(1)
    # 使用正则匹配所有数字标记后的内容，分割符为 >
    # 匹配模式: 1:XXX 或 2:XXX
    items = re.split(r'\s*>\s*', content)
    ranking = []
    for item in items:
        # 去掉 '1:', '2:' 等前缀
        clean_item = re.sub(r'^\d+\s*:\s*', '', item).strip()
        if clean_item:
            ranking.append(clean_latex_answer(clean_item))
    return ranking

# ===== 投票算法 =====

def borda_voting(rankings_with_weights):
    """
    Borda Count: 排名第1得 N 分，第2得 N-1 分...
    rankings_with_weights: List of (ranking_list, weight)
    """
    scores = defaultdict(float)
    for ranking, weight in rankings_with_weights:
        n = len(ranking)
        for i, ans in enumerate(ranking):
            # 分数 = 权重 * (总人数 - 排名索引)
            # 例如 4人排名，第一名(i=0)得 4分
            scores[ans] += weight * (n - i)
    return max(scores, key=scores.get) if scores else None

def recursive_voting(rankings_with_weights):
    """
    Recursive (Instant-Runoff Voting): 每一轮消除得票最低的，直到剩下一个。
    """
    valid_rankings = [(r, w) for r, w in rankings_with_weights if r]
    if not valid_rankings: return None

    # 获取所有出现过的候选人
    candidates = set()
    for r, _ in valid_rankings: candidates.update(r)

    while len(candidates) > 1:
        # 统计当前每一张选票的第一志愿
        round_counts = defaultdict(float)
        for r, w in valid_rankings:
            if r: round_counts[r[0]] += w
        
        # 找到当前得票最低的候选人
        if not round_counts: break
        # 在当前候选人名单中，没出现在 round_counts 里的得票为0
        cand_scores = {c: round_counts.get(c, 0) for c in candidates}
        loser = min(cand_scores, key=cand_scores.get)
        
        # 消除这位失败者
        candidates.remove(loser)
        new_valid_rankings = []
        for r, w in valid_rankings:
            new_r = [c for c in r if c != loser]
            if new_r: new_valid_rankings.append((new_r, w))
        valid_rankings = new_valid_rankings
        
        if not valid_rankings: break

    return list(candidates)[0] if candidates else None

# ===== 数据处理 =====

def online_screening(traces, num_calibration=64, use_low_threshold=False):
    random.seed(13)
    s = None
    if len(traces) >= num_calibration:
        calibration_traces = traces[:num_calibration]
        lowest_confs = [min(t['group_confidence']) for t in calibration_traces if t.get('group_confidence')]
        if lowest_confs:
            s_high = np.percentile(lowest_confs, 10)
            s_low = np.percentile(lowest_confs, 90)
            s = s_high if use_low_threshold else s_low

    predicted_good = []
    if s is not None:
        for trace in traces[num_calibration:]:
            conf_curve = trace.get('group_confidence', [])
            if not conf_curve or min(conf_curve) >= s:
                predicted_good.append(trace)
    
    for trace in traces[:num_calibration]:
        if trace.get('group_confidence') and min(trace['group_confidence']) > (s or 0):
            predicted_good.append(trace)
    return predicted_good

def load_traces(file_path):
    traces = []
    if not file_path.exists(): return traces
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try: traces.append(json.loads(line))
                except: continue
    return traces

def data_loading(base_dir, dataname, is_followup=False):
    results = defaultdict(list)
    for jsonl_path in base_dir.glob(f"{dataname}_*.jsonl"):
        qid_match = re.search(rf"{dataname}_(\d+)_", jsonl_path.name)
        if not qid_match: continue
        qid = int(qid_match.group(1))
        traces = load_traces(jsonl_path)
        
        if is_followup:
            is_topn_true_count = sum(item.get("is_topn", False) for item in traces)
            for item in traces:
                base_ans = clean_latex_answer(item.get("base_answer"))
                # ✅ 提取排名列表
                ranking = extract_ranking_boxed(item.get("trace_2", "")) 
                conf = np.min(item.get("group_confidences_2", np.nan))
                is_topn = item.get("is_topn", False)
                if is_topn_true_count <= 5: is_topn = True
                
                if ranking and (base_ans is not None) and is_topn:
                    results[qid].append((base_ans, ranking, conf))
        else:
            predicted_good = online_screening(traces)
            for item in predicted_good:
                ans = clean_latex_answer(item.get("answer"))
                conf = np.min(item.get("group_confidence", np.nan))
                if ans: results[qid].append((ans, conf))
    return results

def load_ground_truth(path):
    gt = {}
    traces = load_traces(path)
    for i, item in enumerate(traces):
        gt[i] = clean_latex_answer(str(item.get("answer", "")))
    return gt

# ===== 主函数 =====

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, required=True)
    parser.add_argument("--version", type=str, default="4")
    parser.add_argument("--ifcnt", type=str, default="False")
    parser.add_argument("--method", type=str, choices=["borda", "recursive"], default="borda")
    args = parser.parse_args()

    POOLING_DATA_DIR = Path(f"/home/yz54720/Projects/Method/deepconf/data/processed/{args.dataname}/pool_information_v{args.version}")
    TRACES_DIR = Path(f"/home/yz54720/Projects/Method/deepconf/data/processed/{args.dataname}/traces")
    DATA_PATH = Path(f"/home/yz54720/Projects/Method/deepconf/data/raw/{args.dataname}.jsonl")
    
    gt = load_ground_truth(DATA_PATH)
    traces_data = data_loading(TRACES_DIR, args.dataname, is_followup=False)
    followup_data = data_loading(POOLING_DATA_DIR, args.dataname, is_followup=True)

    baseline_correct_cnt = 0
    followup_correct_cnt = 0

    for qid in gt.keys():
        ground_truth = gt[qid]
        traces = traces_data.get(qid, [])
        followups = followup_data.get(qid, [])

        # 1. Baseline Voting (Sum of Confs)
        trace_score_c = defaultdict(float)
        for ans, conf in traces:
            trace_score_c[ans] += conf
        baseline_voted_answer = max(trace_score_c, key=trace_score_c.get) if trace_score_c else None
        if baseline_voted_answer == ground_truth: baseline_correct_cnt += 1

        # 2. Prep Weights for Followup
        counts = Counter([ans for ans, _ in traces])
        top_counts_sorted = sorted(set(counts.values()), reverse=True)
        threshold_values = top_counts_sorted[:min(5, len(top_counts_sorted))]
        top5_anc = {ans: cnt for ans, cnt in counts.items() if cnt in threshold_values}

        # 3. Followup Voting (Ranked)
        rankings_with_weights = []
        for base_ans, ranking, conf in followups:
            cnt = top5_anc.get(base_ans, 1)
            # 最终权重 = 置信度 * (计数权重 if True else 1)
            weight = conf * (cnt if args.ifcnt == "True" else 1)
            rankings_with_weights.append((ranking, weight))

        if args.method == "borda":
            voted_answer = borda_voting(rankings_with_weights)
        else:
            voted_answer = recursive_voting(rankings_with_weights)

        if voted_answer == ground_truth: followup_correct_cnt += 1
        if (baseline_voted_answer != ground_truth) or (voted_answer != ground_truth):
            print(f"❌ QID {qid}: GT={ground_truth!r}, Baseline Vote={baseline_voted_answer!r}, Follow-up Vote={voted_answer!r}")
            # details
            for base_ans, ans, conf in followups:
                print(f"    Base Ans: {base_ans!r}, Follow-up Ans: {ans!r}, Conf: {conf:.4f}, Weight: {top5_anc.get(base_ans, 1)}")


    total = len(gt)
    print(f"\nMethod: {args.method.upper()}")
    print(f"Baseline: {baseline_correct_cnt}/{total} = {baseline_correct_cnt/total:.4f}")
    print(f"Follow-up: {followup_correct_cnt}/{total} = {followup_correct_cnt/total:.4f}")

if __name__ == "__main__":
    main()