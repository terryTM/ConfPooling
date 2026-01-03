"""
Generate trace-confidence dataset for RL training (per-question mode).

This script runs the core vLLM engine directly to generate multiple traces,
ensuring that all confidence calculations start from the raw, unprocessed
log probabilities, guaranteeing full floating-point precision.

After generation, it prints a summary of the top 5 most frequent answers.

Usage example (per question on SLURM):
  srun python generate_trace_dataset.py --dataset aime25.jsonl --qid $SLURM_ARRAY_TASK_ID --budget 256 --output_path ./trace_data/aime25_${SLURM_ARRAY_TASK_ID}.jsonl
"""

import os
import json
import argparse
import numpy as np
import time
import copy
from vllm import SamplingParams
from deepconf import DeepThinkLLM
from dynasor.core.evaluator import math_equal
from typing import List, Dict, Any
from collections import Counter

# -------------------------------
#  Helper functions
# -------------------------------

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

def quick_parse(text: str) -> str:
    """Simplify LaTeX-style answers."""
    if not isinstance(text, str): return ""
    # Corrected the logic for finding the closing brace
    while '\\text{' in text:
        start = text.find('\\text{')
        end = text.find('}', start)
        if start == -1 or end == -1: break
        text = text[:start] + text[start + 6:end] + text[end + 1:]
    return text

def extract_answer(text: str) -> str:
    """Extracts the answer from the full text, compatible with deepconf."""
    if not isinstance(text, str): return None
    if "boxed" in text:
        ans = text.split("boxed")[-1]
        if len(ans) == 0: return ""
        if ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{": stack += 1; a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0: break
                    a += c
                else: a += c
            return a.strip()
        else:
            return ans.split("$")[0].strip()
    return None


def equal_func(answer: str, ground_truth: str) -> bool:
    """Check correctness."""
    if answer is None or ground_truth is None: return False
    answer = quick_parse(str(answer))
    ground_truth = str(ground_truth)
    if (len(answer) == 1 and answer.isalpha() and 
        len(ground_truth) == 1 and ground_truth.isalpha()):
        return answer.lower() == ground_truth.lower()
    else:
        try: return math_equal(answer, ground_truth)
        except Exception: return answer.strip() == ground_truth.strip()

def prepare_prompt(question: str, tokenizer, model_type: str = "deepseek", **kwargs):
    """Prepare prompt."""
    if model_type == "deepseek":
        messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": question}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        messages = [{"role": "user", "content": question}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)

# -------------------------------
#  Main logic
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate trace dataset with full precision confidence.")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--qid", type=int, required=True)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--window_size", type=int, default=2048)
    parser.add_argument("--max_tokens", type=int, default=64000)
    parser.add_argument("--model_type", type=str, default="deepseek", choices=["gpt", "deepseek"])
    parser.add_argument("--reasoning_effort", type=str, default="high")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    print(f"[INFO] Loading dataset from {args.dataset}")
    try:
        with open(args.dataset, "r", encoding="utf-8") as f: data = [json.loads(line) for line in f]
    except Exception as e:
        print(f"[ERROR] Could not read dataset file: {e}"); return

    if not (0 <= args.qid < len(data)):
        print(f"[ERROR] Invalid qid={args.qid}. Dataset has {len(data)} entries."); return

    question_entry = data[args.qid]
    question, ground_truth = question_entry["question"], str(question_entry.get("answer", "")).strip()
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    question_id_str = f"{dataset_name}_{args.qid:03d}"

    print(f"[INFO] Processing QID={args.qid} ({question_id_str}): {question[:80]}...")
    
    deep_llm = DeepThinkLLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size)
    
    prompt = prepare_prompt(question, deep_llm.tokenizer, args.model_type, reasoning_effort=args.reasoning_effort)
    
    base_sampling_params = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        max_tokens=args.max_tokens, logprobs=20,
    )
    
    sampling_params_list = []
    base_seed = int(time.time())
    for i in range(args.budget):
        params = copy.deepcopy(base_sampling_params)
        params.seed = base_seed + i
        sampling_params_list.append(params)

    print(f"[INFO] Generating {args.budget} traces directly with vLLM engine...")
    vllm_outputs = deep_llm.llm.generate([prompt] * args.budget, sampling_params_list)

    print("[INFO] Processing generated traces with full precision...")
    processed_traces = []
    all_answers = []
    
    for i, request_output in enumerate(vllm_outputs):
        try:
            # 检查这个 RequestOutput 是否真的包含结果
            if not request_output.outputs:
                print(f"[WARNING] Empty output for trace {i}, skipping...")
                continue
            
            # 从 RequestOutput 中提取出那唯一的一条 trace
            trace_output = request_output.outputs[0]
            raw_logprobs = trace_output.logprobs
            token_confidences = calculate_token_confs_from_logprobs(raw_logprobs)
            group_confidences = compute_group_confidence_full_precision(token_confidences, args.window_size)
            
            token_ids = trace_output.token_ids
            # 使用 extract_answer 提取答案，以保持与 deepconf 库一致性
            answer_text = extract_answer(trace_output.text)
            all_answers.append(str(answer_text)) # For statistics
            is_correct = equal_func(answer_text, ground_truth)

            processed_traces.append({
                "trace_id": len(processed_traces),  # 使用连续的ID
                # "tokens": deep_llm.tokenizer.convert_ids_to_tokens(token_ids),
                "text": trace_output.text,
                "group_confidence": group_confidences,
                # "trace_confidence" field is now removed
                "answer": answer_text,
                "is_correct": is_correct
            })
        except Exception as e:
            print(f"[ERROR] Failed to process trace {i}: {e}")
            continue

    # --- 新增：在控制台输出答案分布统计 ---
    print("\n--- Answer Distribution (Top 5) ---")
    if all_answers:
        total_answers = len(all_answers)
        answer_counts = Counter(all_answers)
        for answer, count in answer_counts.most_common(5):
            percentage = (count / total_answers) * 100
            print(f"  - Answer: {answer if answer is not None else 'None'}")
            print(f"    Count: {count}/{total_answers} ({percentage:.1f}%)")
    else:
        print("  No answers were extracted.")
    print("-------------------------------------\n")


    # output_data = {
    #     "question_id": question_id_str, "question": question,
    #     "ground_truth": ground_truth, "num_traces": len(processed_traces),
    #     "traces": processed_traces,
    # }

    # output_path = args.output_file
    # try:
    #     with open(output_path, "w", encoding="utf-8") as f: f.write(json.dumps(output_data))
    #     print(f"[SUCCESS] Saved {len(processed_traces)} traces to {output_path}")
    # except IOError as e:
    #     print(f"[ERROR] Failed to write to output file: {e}")

    # --- Save in standard JSONL format ---
    try:
        with open(args.output_path, "w", encoding="utf-8") as f:
            for trace in processed_traces:
                record = {
                    "question_id": question_id_str,
                    "question": question,
                    "ground_truth": ground_truth,
                    "trace_id": trace["trace_id"],
                    "text": trace["text"],
                    "group_confidence": trace["group_confidence"],
                    "answer": trace["answer"],
                    "is_correct": trace["is_correct"]
                }
                f.write(json.dumps(record) + "\n")
        print(f"[SUCCESS] Wrote {len(processed_traces)} traces (JSONL) to {args.output_path}")
    except IOError as e:
        print(f"[ERROR] Failed to write to output file: {e}")

if __name__ == "__main__":
    main()