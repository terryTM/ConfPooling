"""
Modal Phase 1: Generate trace-confidence dataset matching data_creation.py format.

This script generates traces in JSONL format compatible with the RL training pipeline.
Each line contains one trace with question metadata, text, confidence scores, and answer.

Usage:
    # Generate traces for a single question
    modal run modal_phase1.py --dataset data/raw/aime_2024.jsonl --qid 0 --budget 256

    # Generate traces for a range of questions
    modal run modal_phase1.py --dataset data/raw/aime_2024.jsonl --start-qid 0 --end-qid 10 --budget 256
"""
import modal
import json
import os
import copy
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter

# Define Modal app
app = modal.App("phase1-trace-generation")

gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.13.0",
        "huggingface_hub[hf_transfer]==0.36.0",
    )
    .env({
        # Hopper (H200) friendly defaults; avoid Blackwell-only MoE kernels.
        "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8": "0",
        # Enable fast HuggingFace downloads
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
    .add_local_dir(
        "deepconf",
        "/root/deepconf",
        copy=True
    )
    .add_local_dir(
        "data",
        "/root/data",
        copy=True
    )
)


models_volume = modal.Volume.from_name("models-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("results-cache", create_if_missing=True)
GPU_CONFIG = "H200:1"


# -------------------------------
#  Helper functions (from data_creation.py)
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
    if not confs:
        return [0.0]
    if len(confs) < group_size:
        return [sum(confs) / len(confs)]

    sliding_means = []
    current_sum = sum(confs[:group_size])
    sliding_means.append(round(current_sum / group_size, 2))

    for i in range(len(confs) - group_size):
        current_sum = current_sum - confs[i] + confs[i + group_size]
        sliding_means.append(round(current_sum / group_size, 2))

    return sliding_means


def compute_deepconf_group_confidence(raw_logprobs: List[Dict[int, Any]], window_size: int = 2048) -> List[float]:
    """
    Compute Group Confidence per DeepConf paper Appendix G.

    Logic:
    1. Token Confidence: Exclude the highest probability token, compute negative mean log prob of rest.
    2. Group Confidence: Sliding window mean. For short traces (< window_size), uses full trace as single window.

    Args:
        raw_logprobs: vLLM generated trace_output.logprobs (List of dicts).
        window_size: Sliding window size, paper default is 2048.

    Returns:
        List[float]: Average confidence for each complete sliding window.
                     For short traces, returns single-element list with mean of all tokens.
    """
    token_confs = []

    # Step 1: Compute Token Confidence for each step
    for step_logprobs_dict in raw_logprobs:
        if not step_logprobs_dict:
            continue

        # Extract logprob values for all candidate tokens at this step
        all_lps = sorted([lp.logprob for lp in step_logprobs_dict.values()], reverse=True)

        # Exclude logprobs[0] (highest probability sampled token)
        # Compute negative mean log prob of remaining candidates
        if len(all_lps) > 1:
            new_conf = -sum(all_lps[1:]) / len(all_lps[1:])
        else:
            new_conf = 0.0

        token_confs.append(new_conf)

    # Step 2: Compute sliding window mean (Group Confidence)
    if not token_confs:
        return [0.0]

    # For short traces, use entire trace as a single window
    if len(token_confs) < window_size:
        return [sum(token_confs) / len(token_confs)]

    # Standard sliding window for long traces
    group_confs = []
    current_sum = sum(token_confs[:window_size])
    group_confs.append(current_sum / window_size)

    for i in range(len(token_confs) - window_size):
        current_sum = current_sum - token_confs[i] + token_confs[i + window_size]
        group_confs.append(current_sum / window_size)

    return group_confs


def quick_parse(text: str) -> str:
    """Simplify LaTeX-style answers."""
    if not isinstance(text, str):
        return ""
    while '\\text{' in text:
        start = text.find('\\text{')
        end = text.find('}', start)
        if start == -1 or end == -1:
            break
        text = text[:start] + text[start + 6:end] + text[end + 1:]
    return text


def extract_answer(text: str) -> Optional[str]:
    """Extracts the answer from the full text, compatible with deepconf."""
    if not isinstance(text, str):
        return None
    if "boxed" in text:
        ans = text.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        if ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
            return a.strip()
        else:
            return ans.split("$")[0].strip()
    return None


def equal_func(answer: str, ground_truth: str) -> bool:
    """Check correctness."""
    if answer is None or ground_truth is None:
        return False
    answer = quick_parse(str(answer))
    ground_truth = str(ground_truth)
    if (len(answer) == 1 and answer.isalpha() and
        len(ground_truth) == 1 and ground_truth.isalpha()):
        return answer.lower() == ground_truth.lower()
    else:
        return answer.strip() == ground_truth.strip()


@app.cls(
    image=gpu_image,
    gpu=GPU_CONFIG,
    timeout=86000,
    volumes={"/models": models_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class Phase1TraceGenerator:
    """Modal class for generating Phase 1 traces in data_creation.py format"""

    # Use modal.parameter() instead of __init__ (Modal's new recommended pattern)
    model_name: str = modal.parameter(default="openai/gpt-oss-120b")

    @modal.enter()
    def setup(self):
        """Initialize vLLM model on Modal container startup"""
        import sys
        sys.path.insert(0, "/root")

        from vllm import LLM
        from transformers import AutoTokenizer

        print(f"Loading model: {self.model_name}")
        start_time = time.time()

        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=1,
            enable_prefix_caching=True,
            trust_remote_code=True,
            download_dir="/models",
            gpu_memory_utilization=0.95,
            max_model_len=131072,  # 130k context
            kv_cache_dtype="fp8_e4m3",  # FP8 KV cache for memory efficiency
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f}s")

    def _prepare_prompt(self, question: str) -> str:
        """Prepare prompt using model's chat template without forcing high reasoning mode."""
        messages = [{"role": "user", "content": question}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    @modal.method()
    def generate_traces(
        self,
        question: str,
        ground_truth: str,
        qid: int,
        dataset_name: str,
        budget: int = 256,
        window_size: int = 2048,
        max_tokens: int = 131072,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 40,
    ) -> List[Dict[str, Any]]:
        """
        Generate traces for a single question in data_creation.py format.

        Returns a list of trace records, each ready to be written as a JSONL line.
        """
        from vllm import SamplingParams

        question_id_str = f"{dataset_name}_{qid:03d}"
        print(f"\n=== Generating {budget} traces for QID {qid} ({question_id_str}) ===")
        print(f"Question: {question[:80]}...")

        prompt = self._prepare_prompt(question)

        # Create sampling params with different seeds for each trace
        base_seed = int(time.time())
        sampling_params_list = []

        for i in range(budget):
            params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                logprobs=20,
                seed=base_seed + i,
            )
            sampling_params_list.append(params)

        print(f"[INFO] Generating {budget} traces with vLLM engine...")
        start_time = time.time()
        vllm_outputs = self.llm.generate([prompt] * budget, sampling_params_list)
        generation_time = time.time() - start_time
        print(f"[INFO] Generation completed in {generation_time:.2f}s")

        print("[INFO] Processing generated traces...")
        processed_traces = []
        all_answers = []

        for i, request_output in enumerate(vllm_outputs):
            try:
                if not request_output.outputs:
                    print(f"[WARNING] Empty output for trace {i}, skipping...")
                    continue

                trace_output = request_output.outputs[0]
                raw_logprobs = trace_output.logprobs

                # Compute confidence scores using both methods
                group_confidences = compute_deepconf_group_confidence(raw_logprobs, window_size)
                token_confidences = calculate_token_confs_from_logprobs(raw_logprobs)
                old_group_confidences = compute_group_confidence_full_precision(token_confidences, window_size)

                # Extract answer
                answer_text = extract_answer(trace_output.text)
                all_answers.append(str(answer_text))
                is_correct = equal_func(answer_text, ground_truth)

                # Build record in data_creation.py format
                record = {
                    "question_id": question_id_str,
                    "question": question,
                    "ground_truth": ground_truth,
                    "trace_id": len(processed_traces),
                    "text": trace_output.text,
                    "group_confidence": group_confidences,
                    "answer": answer_text,
                    "is_correct": is_correct,
                    "old_group_confidence": old_group_confidences,
                }
                processed_traces.append(record)

            except Exception as e:
                print(f"[ERROR] Failed to process trace {i}: {e}")
                continue

        # Print answer distribution
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

        return processed_traces


@app.function(
    image=gpu_image,
    # No GPU needed - this function only orchestrates, the GPU work happens in Phase1TraceGenerator
    timeout=86000,  # 24 hours
    volumes={"/results": results_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_phase1(
    dataset_path: str,
    qid: Optional[int] = None,
    start_qid: Optional[int] = None,
    end_qid: Optional[int] = None,
    budget: int = 256,
    window_size: int = 2048,
    max_tokens: int = 131072,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 40,
    model_name: str = "openai/gpt-oss-120b",
    output_dir: str = "/results/traces_jsonl",
) -> str:
    """
    Run Phase 1 trace generation.

    Can run for a single qid or a range of qids.
    Outputs JSONL files matching data_creation.py format.
    """
    from pathlib import Path
    from datetime import datetime

    # Load dataset (prepend /root/ for Modal mounts)
    if not dataset_path.startswith('/'):
        dataset_path = f"/root/{dataset_path}"

    print(f"[INFO] Loading dataset from {dataset_path}")
    with open(dataset_path, 'r') as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

    # Determine which questions to process
    if qid is not None:
        qids = [qid]
    elif start_qid is not None and end_qid is not None:
        qids = list(range(start_qid, end_qid))
    else:
        qids = list(range(len(data)))

    print(f"[INFO] Processing QIDs: {qids}")
    print(f"[INFO] Budget: {budget}, Window: {window_size}, Max tokens: {max_tokens}")
    print(f"[INFO] Sampling: temp={temperature}, top_p={top_p}, top_k={top_k}")

    # Reload volume to see latest files (for resume after preemption)
    results_volume.reload()

    # Create generator
    generator = Phase1TraceGenerator(model_name=model_name)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_files = []

    skipped_existing = 0
    for q in qids:
        if not (0 <= q < len(data)):
            print(f"[ERROR] Invalid qid={q}. Dataset has {len(data)} entries. Skipping.")
            continue

        # Check if output file already exists - skip if so
        output_path = f"{output_dir}/{dataset_name}_{q:03d}.jsonl"
        if os.path.exists(output_path):
            skipped_existing += 1
            print(f"[SKIP] Output already exists: {output_path}")
            output_files.append(output_path)
            continue

        question_entry = data[q]
        question = question_entry["question"]
        ground_truth = str(question_entry.get("answer", "")).strip()

        # Generate traces
        traces = generator.generate_traces.remote(
            question=question,
            ground_truth=ground_truth,
            qid=q,
            dataset_name=dataset_name,
            budget=budget,
            window_size=window_size,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        # Write to JSONL file (output_path already defined above)
        with open(output_path, "w", encoding="utf-8") as f:
            for trace in traces:
                f.write(json.dumps(trace) + "\n")

        # Commit to volume after each question to persist progress
        results_volume.commit()
        print(f"[SUCCESS] Wrote {len(traces)} traces to {output_path} (committed to volume)")
        output_files.append(output_path)

    if skipped_existing > 0:
        print(f"\n[INFO] Skipped {skipped_existing} questions (already completed)")
    return f"Generated traces for {len(output_files)} questions ({skipped_existing} skipped as already done): {output_files}"


@app.local_entrypoint()
def main(
    dataset: str = "data/raw/aime_2024.jsonl",
    qid: Optional[int] = None,
    start_qid: Optional[int] = None,
    end_qid: Optional[int] = None,
    budget: int = 256,
    window_size: int = 2048,
    max_tokens: int = 131072,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 40,
    model: str = "openai/gpt-oss-120b",
    output_dir: str = "/results/traces_jsonl",
):
    """
    Phase 1: Generate trace-confidence dataset matching data_creation.py format.

    Examples:
        # Single question
        modal run modal_phase1.py --dataset data/raw/aime_2024.jsonl --qid 0 --budget 256

        # Range of questions
        modal run modal_phase1.py --dataset data/raw/aime_2024.jsonl --start-qid 0 --end-qid 10

        # Custom parameters
        modal run modal_phase1.py --dataset data/raw/aime_2024.jsonl --qid 0 --temperature 1.0 --top-p 1.0 --top-k 40
    """
    print("=" * 80)
    print("Phase 1: Trace Generation")
    print("=" * 80)
    print(f"Dataset: {dataset}")
    print(f"Model: {model}")
    print(f"Budget: {budget}")
    print(f"Sampling: temp={temperature}, top_p={top_p}, top_k={top_k}")
    print(f"Max tokens: {max_tokens}")
    print(f"Output dir: {output_dir}")
    print("=" * 80)

    result = run_phase1.remote(
        dataset_path=dataset,
        qid=qid,
        start_qid=start_qid,
        end_qid=end_qid,
        budget=budget,
        window_size=window_size,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        model_name=model,
        output_dir=output_dir,
    )

    print("\n" + "=" * 80)
    print("Result:", result)
    print("=" * 80)
