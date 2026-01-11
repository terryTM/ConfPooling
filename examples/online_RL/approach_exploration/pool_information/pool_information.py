import argparse
from deepconf import DeepThinkLLM
from vllm import SamplingParams, LLM

# 假设 deep_llm 实例和第一轮的 traces 数据已经准备好
# deep_llm = DeepThinkLLM(model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
# traces = load_concatenated_json('trace_data/aime_2025_0_full.jsonl')['traces']
import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
from typing import List, Dict, Any
random.seed(13)

######### LOAD DATA #########
# --- 1. 配置 ---
# --- 2. 将工作目录设置为项目根目录 (如果需要) ---
os.chdir(os.path.expanduser('~/Projects/Method/deepconf'))
def clean_answer(ans):
    if not ans: return ans
    ans = re.sub(r"\\(?:,|;|:|!|quad|qquad|enspace|,| )", "", ans)
    ans = re.sub(r"\s+", "", ans)
    ans = ans.replace(r"\dfrac", r"\frac")
    ans = re.sub(r"\b[a-zA-Z]\s*=", "", ans)
    ans = re.sub(r",+", ",", ans)
    ans = ans.strip(",")
    return ans
def load_concatenated_json(file_path):
    """
    Reads a file containing one or more concatenated JSON objects
    and merges them into a single, valid data structure.
    """
    decoder = json.JSONDecoder()
    data_parts = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        idx = 0
        while idx < len(content):
            while idx < len(content) and content[idx].isspace():
                idx += 1
            if idx == len(content):
                break
            try:
                obj, end = decoder.raw_decode(content, idx)
                data_parts.append(obj)
                idx = end
            except json.JSONDecodeError:
                break

    if not data_parts:
        return None

    # Merge the parts
    final_data = {k: v for k, v in data_parts[0].items() if k != 'traces'}
    all_traces = [trace for part in data_parts for trace in part.get('traces', [])]
    
    final_data['traces'] = all_traces
    final_data['num_traces'] = len(all_traces)
    
    return final_data

def build_follow_up_question(current_answer, current_max_conf, top_n_data, max_length, llm):
    """
    根据用户逻辑优化的双路径 Prompt。
    Peer Review: 极简逻辑验证。
    Correction Needed: 强化纠偏压力。
    """
    is_in_top_n = current_answer in top_n_data
    
    if is_in_top_n:
        # Peer Review: Top 4 内部对比
        candidates = {ans: conf for ans, conf in top_n_data.items() if ans != current_answer}
        protocol_type = "PEER_REVIEW"
    else:
        # Correction Needed: 与 Top 3 进行对比
        top_3_ans = list(top_n_data.keys())[:3]
        candidates = {ans: top_n_data[ans] for ans in top_3_ans}
        protocol_type = "CORRECTION_NEEDED"

    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1])

    while True:
        cand_text = "\n".join([
            f"{i}. Candidate {ans} (Confidence: {conf:.4f})" 
            for i, (ans, conf) in enumerate(sorted_candidates, 1)
        ])

        if protocol_type == "PEER_REVIEW":
            # 极简模式：去掉对 Confidence 的重复说明，专注逻辑
            prompt_body = f"""
Previously, you concluded: {current_answer} (Confidence: {current_max_conf:.4f})

Other independent reasoning paths produced the following high-confidence candidates:
{cand_text}

### EVALUATION PROTOCOL:
1. **Divergence Analysis**: Identify the specific logical junction where your reasoning differs from these alternatives.
2. **Objective Verification**: Evaluate each path based on its mathematical and logical soundness.
3. **Final Decision**: Maintain your conclusion if the logic remains robust, or pivot if you find a definitive flaw. Avoid changing your answer solely for the sake of alignment.
"""
        else:
            # 强化模式：明确指出当前答案极大概率错误，正确答案在列表中
            # 这里的 N 使用 len(candidates) 动态显示
            prompt_body = f"""
Previously, you concluded: {current_answer} (Confidence: {current_max_conf:.4f})

**URGENT DIAGNOSTIC REQUIRED:** Our multi-path analysis indicates your previous answer is highly likely to be incorrect, as it failed to reach the confidence threshold of the top candidates. **The correct solution is almost certainly contained within the alternatives listed below:**
{cand_text}

### CORRECTION PROTOCOL:
1. **Flaw Identification**: Treat your previous reasoning as having a confirmed logical derailment. Your task is to find that specific error by contrasting it with the paths above.
2. **Path Reconstruction**: These candidates represent the most stable and consistent reasoning found across multiple independent trials. Re-verify them to identify which one represents the objective truth.
3. **Decisive Pivot**: Use the candidates above as your primary reference to correct your reasoning and provide the valid final answer.
"""

        full_prompt = prompt_body.strip() + "\n\nFinal decision format: **FINAL ANSWER: \\boxed{{X}}**"
        
        # Token 检查与动态截断
        if len(llm.tokenizer.tokenize(full_prompt)) <= max_length:
            return full_prompt
        
        if sorted_candidates:
            sorted_candidates.pop(0) # 优先删除置信度较低的干扰项
        else:
            return f"Re-verify: {current_answer}\nFINAL ANSWER: \\boxed{{X}}"


def calculate_token_confs_from_logprobs(logprobs: List[Dict[int, Any]]) -> List[float]:
    """
    Calculates token confidences from raw vLLM logprobs with full precision.
    """
    token_confs = []
    if not logprobs:
        return token_confs
        
    for step_logprobs in logprobs:
        if not step_logprobs:
            continue
        chosen_token_logprob = -float('inf')
        for logprob_obj in step_logprobs.values():
            chosen_token_logprob = max(chosen_token_logprob, logprob_obj.logprob)
        
        confidence = np.exp(chosen_token_logprob)
        token_confs.append(confidence)
        
    return token_confs


def compute_group_confidence_full_precision(confs: List[float], group_size: int) -> List[float]:
    """Computes sliding window mean confidence with full floating-point precision."""
    if not confs: return [0.0]
    if len(confs) < group_size: return [sum(confs) / len(confs)]
    
    sliding_means = []
    current_sum = sum(confs[:group_size])
    sliding_means.append(current_sum / group_size)
    
    for i in range(len(confs) - group_size):
        current_sum = current_sum - confs[i] + confs[i + group_size]
        sliding_means.append(current_sum / group_size)
        
    return sliding_means
def split_thinking_and_answer(text):
    """
    拆分 <think> ... </think> 结构，返回 (reasoning, answer)，
    并去掉末尾的 <｜end▁of▁sentence｜>。
    """
    # 匹配 <think> ... </think>
    match = re.search(r"<think>(.*?)</think>(.*)", text, flags=re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
        answer = match.group(2).strip()
    else:
        reasoning = ""
        answer = text.strip()

    # 去除结尾的特殊符号（包含可能的空格或换行）
    answer = re.sub(r"<\s*[\|｜]\s*end▁of▁sentence\s*[\|｜]\s*>", "", answer, flags=re.IGNORECASE).strip()

    return reasoning, answer
def parse_args():
    parser = argparse.ArgumentParser(description="Run follow-up self-check experiment")

    parser.add_argument("--model", type=str, default="deepseek-r1-qwen-8b",
                        help="LLM model name to use.")
    parser.add_argument("--data_name", type=str, default="aime_2025")
    parser.add_argument("--question_id", type=int, required=True,
                        help="AIME problem ID (for logging).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output traces.")
    parser.add_argument("--window_size", type=int, default=2048,
                        help="Window size for computing group confidence.")

    return parser.parse_args()
# --- 4. Load the Target File ---
def main():
    args = parse_args()
    import json
    import numpy as np

    path = f"data/processed/{args.data_name}/traces/{args.data_name}_{args.question_id}_full.jsonl" 
    traces = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError: continue

    print(f"[INFO] 成功加载 {len(traces)} 条记录")
    question = traces[0].get("question", "Question text not found.")

    ######### SCREENING & MaxC POOLING #########
    NUM_CALIBRATION_TRACES = 64
    PERCENTILE = 90 # 推荐使用 80，兼顾去噪与召回
    
    # 预计算每条 trace 的 min_conf
    for t in traces:
        t['min_conf'] = min(t['group_confidence']) if t.get('group_confidence') else 0.0
        t['answer'] = clean_answer(t.get("answer", ""))

    # 计算筛选阈值 s
    s = 0.0
    if len(traces) >= NUM_CALIBRATION_TRACES:
        calibration_confs = [t['min_conf'] for t in traces[:NUM_CALIBRATION_TRACES]]
        s = np.percentile(calibration_confs, PERCENTILE)
        print(f"--- Screening Active: s = {s:.4f} ---")

    # 过滤并按答案聚合 MaxC
    predicted_good = [t for t in traces if t['min_conf'] >= s and t.get("answer") is not None]
    
    answer_max_confs = {}
    for t in predicted_good:
        ans = t['answer']
        conf = t['min_conf']
        if ans not in answer_max_confs or conf > answer_max_confs[ans]:
            answer_max_confs[ans] = conf

    # 选出 Top 4 作为竞争池 (N=4)
    top_n_answers = sorted(answer_max_confs.keys(), key=lambda x: answer_max_confs[x], reverse=True)[:4]
    top_n_data = {ans: answer_max_confs[ans] for ans in top_n_answers}

    ########### INITIALIZE LLM & FOLLOW-UP (Batch Mode) ###########
    deep_llm = DeepThinkLLM(model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    
    prompts_to_run = []
    params_to_run = []
    metadata_list = []

    if predicted_good:
        print(f"\n[INFO] Preparing {len(predicted_good)} tasks for Batch Generation...")
        
        for base_trace in predicted_good:
            current_answer = base_trace.get("answer")
            current_max_conf = answer_max_confs[current_answer]
            trace_1_string = base_trace.get("text", "")
            
            # 计算动态截断长度，防止总长度溢出
            max_length = 131072 - len(deep_llm.tokenizer.tokenize(trace_1_string)) - len(deep_llm.tokenizer.tokenize(question)) - 500
            
            follow_up_question = build_follow_up_question(
                current_answer, current_max_conf, top_n_data, max_length, deep_llm
            )

            messages_turn_2 = [
                {"role": "system", "content": "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是2025年5月28日，星期一。\n"},
                {"role": "user", "content": question},
                {"role": "assistant", "content": trace_1_string},
                {"role": "user", "content": follow_up_question}
            ]

            prompt_2 = deep_llm.tokenizer.apply_chat_template(messages_turn_2, tokenize=False, add_generation_prompt=True)
            
            prompts_to_run.append(prompt_2)
            params_to_run.append(SamplingParams(temperature=0.6, max_tokens=64000, top_p=0.95, logprobs=20))
            metadata_list.append({
                "base_trace_id": base_trace.get("trace_id"),
                "base_answer": current_answer,
                "current_max_conf": current_max_conf,
                "is_topn": current_answer in top_n_answers
            })

        # --- 批量生成核心逻辑 ---
        # 针对 131k context，BATCH_SIZE 建议设为 16 以平衡吞吐量与显存风险
        BATCH_SIZE = 16 
        all_traces_2 = []
        
        for i in range(0, len(prompts_to_run), BATCH_SIZE):
            # 这里的切片会自动处理末尾不足 BATCH_SIZE 的情况
            chunk_prompts = prompts_to_run[i : i + BATCH_SIZE]
            chunk_params = params_to_run[i : i + BATCH_SIZE]
            chunk_meta = metadata_list[i : i + BATCH_SIZE]
            
            current_batch_count = len(chunk_prompts)
            print(f"  - [Batch] Processing {i} to {i + current_batch_count} (Current Chunk Size: {current_batch_count})...")
            
            # 批量喂给 vLLM
            vllm_outputs = deep_llm.llm.generate(chunk_prompts, chunk_params)
            
            for idx, output in enumerate(vllm_outputs):
                meta = chunk_meta[idx]
                trace_2_text = output.outputs[0].text
                raw_logprobs = output.outputs[0].logprobs
                
                token_confidences = calculate_token_confs_from_logprobs(raw_logprobs)
                group_confidences_2 = compute_group_confidence_full_precision(token_confidences, args.window_size)

                all_traces_2.append({
                    "base_trace_id": meta["base_trace_id"],
                    "base_answer": meta["base_answer"],
                    "current_max_conf": meta["current_max_conf"],
                    "is_topn": meta["is_topn"],
                    "trace_2": trace_2_text, 
                    "group_confidences_2": group_confidences_2,
                    "trace2_token_length": len(deep_llm.tokenizer.tokenize(trace_2_text))
                })

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = f'{args.output_dir}/{args.data_name}_{args.question_id}_deepconflow_self_check.jsonl'
    with open(out_file, 'w', encoding='utf-8') as f:
        for item in all_traces_2:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
if __name__ == '__main__':
    main()