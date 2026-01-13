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

def build_follow_up_question(current_answer, top_n_data):
    """
    根据 Ranked Voting 逻辑优化的追问 Prompt。
    top_n_data: Dict[str, dict] -> {answer: {"max_conf": float, "summary": str}}
    """
    # 获取 Top 4 列表
    top_4_list = list(top_n_data.keys())
    is_in_top_n = current_answer in top_4_list

    # 构造统一的候选答案展示文本（包含推理过程）
    candidate_display_text = ""
    for i, ans in enumerate(top_4_list, 1):
        info = top_n_data[ans]
        candidate_display_text += f"\n[Candidate {i}]: {ans} (Confidence: {info['max_conf']:.4f})\n"
        candidate_display_text += f"Reasoning Summary: {info['summary']}\n"

    if is_in_top_n:
        # 路径 A: 内部成员 (PEER REVIEW)
        prompt_body = f"""
Previously, you concluded: {current_answer} 

Other independent reasoning paths produced the following high-confidence candidates:
{candidate_display_text}

### EVALUATION PROTOCOL:
1. **Divergence Analysis**: Identify the specific logical junction where your reasoning differs from these alternatives.
2. **Objective Verification**: Evaluate each path based on its mathematical and logical soundness.
3. **Final Decision**: Maintain your conclusion if the logic remains robust, or pivot if you find a definitive flaw. Avoid changing your answer solely for the sake of alignment.
"""
    else:
        # 场景 B：当前 Trace 不在精英组 (External Arbitration)
        prompt_body = f"""
Previously, you concluded: {current_answer} 

Other independent reasoning paths produced the following high-confidence candidates:
{candidate_display_text}

### EVALUATION PROTOCOL:
1. **Divergence Analysis**: Identify the specific logical junction where your reasoning differs from these alternatives.
2. **Objective Verification**: Evaluate each path based on its mathematical and logical soundness.
3. **Final Decision**: Maintain your conclusion if the logic remains robust, or pivot if you find a definitive flaw. Avoid changing your answer solely for the sake of alignment.
"""

    # 统一的输出格式要求：强制使用 \boxed 排序

    full_prompt = prompt_body.strip() + "\n\nPut your final answer within \\boxed{}"
    return full_prompt

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
        summary = match.group(2).strip()
    else:
        reasoning = ""
        summary = text.strip()

    # 去除结尾的特殊符号（包含可能的空格或换行）
    summary = re.sub(r"<\s*[\|｜]\s*end▁of▁sentence\s*[\|｜]\s*>", "", summary, flags=re.IGNORECASE).strip()

    return reasoning, summary
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
        
        answer_best_info = {}
        for t in predicted_good:
            ans = t['answer']
            conf = t['min_conf']
            # 如果该答案还没记录，或者当前 trace 置信度更高，则更新
            if ans not in answer_best_info or conf > answer_best_info[ans]['max_conf']:
                # 提取推理部分 (reasoning) 作为 summary
                _, summary = split_thinking_and_answer(t.get("text", ""))
                answer_best_info[ans] = {
                    'max_conf': conf,
                    'summary': summary 
                }

        # 2. 确定最终的 Top 4 字典（所有 base trace 共用这套数据）
        top_n_answers = sorted(answer_best_info.keys(), key=lambda x: answer_best_info[x]['max_conf'], reverse=True)[:4]
        top_n_data = {ans: answer_best_info[ans] for ans in top_n_answers}

        # --- 进入循环 ---
        for base_trace in predicted_good:
            current_answer = base_trace.get("answer")
            trace_1_string = base_trace.get("text", "")
            
            # 调用下面补全的函数
            follow_up_question = build_follow_up_question(current_answer, top_n_data)

            messages_turn_2 = [
                {"role": "system", "content": "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是2025年5月28日，星期一。\n"},
                {"role": "user", "content": question},
                {"role": "assistant", "content": trace_1_string},
                {"role": "user", "content": follow_up_question}
            ]

            prompt_2 = deep_llm.tokenizer.apply_chat_template(messages_turn_2, tokenize=False, add_generation_prompt=True)
            
            prompts_to_run.append(prompt_2)
            params_to_run.append(SamplingParams(temperature=0.6, max_tokens=24000, top_p=0.95, logprobs=20))
            metadata_list.append({
                "base_trace_id": base_trace.get("trace_id"),
                "base_answer": current_answer,
                "is_topn": current_answer in top_n_answers
            })

        # --- 批量生成核心逻辑 ---
        # 针对 131k context，BATCH_SIZE 建议设为 16 以平衡吞吐量与显存风险
        BATCH_SIZE = 32
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