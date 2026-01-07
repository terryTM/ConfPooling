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

    path = f"data/processed/{args.data_name}/traces/{args.data_name}_{args.question_id}_full.jsonl" 
    traces = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[ERROR] 第 {i} 行解析失败: {e}")
                break

    print(f"[INFO] 成功加载 {len(traces)} 条记录")

    question = traces[0].get("question", "Question text not found.")

    ######### SCREENING #########

    # --- Early Stopping Simulation Configuration ---
    NUM_CALIBRATION_TRACES = 64
    USE_LOW_THRESHOLD = False  # True: 10% percentile (lenient), False: 90% percentile (strict)
    random.seed(13)

    # --- Calculate Threshold ---
    s = None
    predicted_good = [] 
    if len(traces) >= NUM_CALIBRATION_TRACES:
        calibration_traces = traces[:NUM_CALIBRATION_TRACES]
        lowest_confs = [min(t['group_confidence']) for t in calibration_traces if t['group_confidence']]
        if lowest_confs:
            s_high = np.percentile(lowest_confs, 10)
            s_low = np.percentile(lowest_confs, 90) # bigger
            s = s_high if USE_LOW_THRESHOLD else s_low
            print(f"--- Early Stopping Simulation ---")
            print(f"Threshold computed from {len(lowest_confs)} calibration samples.")
            print(f"  - High Threshold (10th percentile): {s_high:.4f}") 
            print(f"  - Low Threshold (90th percentile): {s_low:.4f}")
            print(f"--- Active Threshold s = {s:.4f} ---")
            # 保留lowest_confs大于s的trace
            
    # --- Plotting and Confusion Matrix Calculation ---
    # TODO 修改一下，使得budget逻辑是对齐的（即最后T中有budget条），最好加入consensus
    if s is not None:

        remaining_filtered_traces = []
        remaining_traces = traces[NUM_CALIBRATION_TRACES:]
        for trace in remaining_traces:
            conf_curve = trace['group_confidence']

            stop_indices = np.where(np.array(conf_curve) < s)[0] if conf_curve else []
            predicted_as_bad = len(stop_indices) > 0

            # ✅ 保存分类结果
            if predicted_as_bad:
                pass
            else:
                remaining_filtered_traces.append(trace)
 
        predicted_good = remaining_filtered_traces.copy()
        # 保留calibration中lowest_confs大于s的trace
        for trace in calibration_traces:
            if min(trace['group_confidence']) >= s:
                predicted_good.append(trace)
        # 建议修改点：去掉answer为None的trace
        predicted_good = [t for t in predicted_good if t.get("answer") is not None]
        # --- ✅ Show Predicted Good Answers ---
        print("\n--- Answers from Predicted Good Traces ---")
        good_answers = [t["answer"] for t in predicted_good if t.get("answer") is not None]
        if good_answers:
            # 只包含非 None 答案
            from collections import Counter
            counts = Counter(good_answers)
            for ans, cnt in counts.most_common(15):  # 只打印前 15 个最常见的
                print(f"{ans!r:40s}  →  {cnt} traces")
            # 查询traces数量中最大的五个数
            answer_number_sorted = sorted(set(counts.values()), reverse=True)
            print(answer_number_sorted[:min(5, len(answer_number_sorted))])
            top5_anc = {ans:cnt for ans, cnt in counts.items() if cnt in answer_number_sorted[:min(5, len(answer_number_sorted))]}
            top5_answers = list(top5_anc.keys())
            # filtered_traces = [t for t in predicted_good if t.get("answer") in top5_answers]

            # print(f"\n✅ Delete {len(predicted_good) - len(filtered_traces)} traces not belonging to top 5 answers, deleted answers are: {set(t.get('answer') for t in predicted_good) - set(top5_answers)}")
            # print(f"top5_answers: {top5_answers}")
        else:
            print("No predicted good traces found.")
        
    ########### INITIALIZE DEEP THINK LLM ###########
    # TODO: 不能同时启用两个，不然显存不够
    # TODO：如果不做online的筛选，直接用raw的前多少个来做pooling会如何
    deep_llm = DeepThinkLLM(model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    # --- 使用已有的 JSONL 数据进行追问 ---
    # 1. 提取第一轮的上下文
    # 假设我们选择第一条 trace (trace_id: 0) 作为上下文

    all_traces_2=[]
    if good_answers:

        print("\n=== Generating Self-Check Prompts for Each Candidate Answer ===")

        # 对所有 predicted good traces 进行追问
        for base_trace in predicted_good:
            current_answer = base_trace.get("answer")
            current_answer_number = counts[current_answer]
            trace_1_string = base_trace.get("text", "(Reasoning trace missing...)")
            # 其他每个答案都抽一个trace，并且从他们的回答中用split_thinking_and_answer提取回答的部分，返回的结果是{}
            other_answers = [ans for ans in top5_answers if ans != current_answer]
            other_answers_text = {}
            for ans in other_answers:
                ans_trace = random.choice([t for t in predicted_good if t.get("answer") == ans])
                # ans_text = deep_llm.tokenizer.convert_tokens_to_string(ans_trace.get("tokens", "(Reasoning trace missing...)"))
                ans_text = ans_trace.get("text", "(Reasoning trace missing...)")
                _, ans_only = split_thinking_and_answer(ans_text)
                other_answers_text[ans] = ans_only

            # 3. 准备我们的追问
            max_length = 131072 - len(deep_llm.tokenizer.tokenize(trace_1_string)) - len(deep_llm.tokenizer.tokenize(question))
            follow_up_question = build_follow_up_question(current_answer, current_answer_number, other_answers_text, top5_anc, max_length, deep_llm)
            # 4. 构建包含完整历史的消息列表
            #    [user_q1, assistant_a1, user_q2]
            messages_turn_2 = [
                {"role": "system", "content": "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是2025年5月28日，星期一。\n"},
                {"role": "user", "content": question},
                {"role": "assistant", "content": trace_1_string}, # 使用我们刚刚转换好的字符串
                {"role": "user", "content": follow_up_question}
            ]

            # 5. 应用聊天模板，生成包含上下文的第二轮 prompt
            prompt_2 = deep_llm.tokenizer.apply_chat_template(
                messages_turn_2,
                tokenize=False,
                add_generation_prompt=True
            )

            # 6. 调用 vLLM 获取追问的回答
            sampling_params = SamplingParams(temperature=0.6, max_tokens=64000, top_p=0.95, top_k=0, logprobs=20)
            outputs_2 = deep_llm.generate(prompt_2, sampling_params)
            trace_2 = outputs_2[0].outputs[0].text
            # 这里要改，logprobs什么都没有
            raw_logprobs = outputs_2[0].outputs[0].logprobs
            token_confidences = calculate_token_confs_from_logprobs(raw_logprobs)
            group_confidences = compute_group_confidence_full_precision(token_confidences, args.window_size)
            all_traces_2.append({
                "base_trace_id": base_trace["trace_id"],
                "base_summary": trace_1_string, 
                "base_answer": current_answer,
                "other_answers": other_answers,
                "trace_2": trace_2, 
                "group_confidences_2": group_confidences, 
                "trace2_token_length": len(deep_llm.tokenizer.tokenize(trace_2))
            })
    # write to file
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f'{args.output_dir}/{args.data_name}_{args.question_id}_deepconflow_self_check.jsonl', 'w', encoding='utf-8') as f:
        for item in all_traces_2:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
if __name__ == '__main__':
    main()