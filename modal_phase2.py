"""
Modal Phase 2: Pool Information Self-Verification

This script takes Phase 1 traces and runs a follow-up "self-check" phase where the model
re-evaluates its answer after seeing what other high-confidence traces concluded.

The key idea is "wisdom of crowds" verification - showing the model high-confidence
alternatives to see if it changes its mind. Incorrect traces will more likely switch
to correct answers when confronted with confident alternatives.

Usage:
    # Run phase 2 on a single question's traces
    modal run modal_phase2.py --traces-file /results/traces_jsonl/aime_2024_000.jsonl

    # Run phase 2 on multiple questions
    modal run modal_phase2.py --traces-dir /results/traces_jsonl --dataset-prefix aime_2024 --start-qid 0 --end-qid 10
"""
import modal
import json
import os
import re
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter

# Define Modal app
app = modal.App("phase2-pool-information")

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
#  Helper functions
# -------------------------------

def clean_answer(ans: str) -> str:
    """Clean LaTeX formatting from answer for comparison."""
    if not ans:
        return ans
    ans = re.sub(r"\\(?:,|;|:|!|quad|qquad|enspace|,| )", "", ans)
    ans = re.sub(r"\s+", "", ans)
    ans = ans.replace(r"\dfrac", r"\frac")
    ans = ans.replace(r"\tfrac", r"\frac")
    ans = ans.replace(r"\displaystyle", "")
    ans = re.sub(r"\b[a-zA-Z]\s*=", "", ans)
    ans = re.sub(r",+", ",", ans)
    ans = ans.strip(",")
    return ans


def extract_answer(text: str) -> Optional[str]:
    """Extract boxed answer from text."""
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


def calculate_token_confs_from_logprobs(logprobs: List[Dict[int, Any]]) -> List[float]:
    """Calculate token confidences from raw vLLM logprobs."""
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


def compute_group_confidence(confs: List[float], group_size: int) -> List[float]:
    """Compute sliding window mean confidence."""
    if not confs:
        return [0.0]
    if len(confs) < group_size:
        return [sum(confs) / len(confs)]

    sliding_means = []
    current_sum = sum(confs[:group_size])
    sliding_means.append(current_sum / group_size)

    for i in range(len(confs) - group_size):
        current_sum = current_sum - confs[i] + confs[i + group_size]
        sliding_means.append(current_sum / group_size)

    return sliding_means


def get_trace_confidence(trace: Dict[str, Any]) -> float:
    """
    Get a single confidence value for a trace, handling short traces.

    For traces with group_confidence data:
        - Returns min(group_confidence)
    For short traces (< window_size tokens):
        - Computes mean of old_group_confidence if available
        - Falls back to 0.0 if no confidence data exists

    This ensures all traces get a usable confidence score for filtering/ranking.
    """
    # First try: use existing group_confidence (DeepConf method)
    gc = trace.get('group_confidence', [])
    if gc:
        return min(gc)

    # Second try: use old_group_confidence (standard method, handles short traces)
    old_gc = trace.get('old_group_confidence', [])
    if old_gc:
        return min(old_gc)

    # No precomputed confidence available
    return 0.0


def build_follow_up_prompt(current_answer: str, top_n_data: Dict[str, Dict], if_sf: bool = False) -> str:
    """
    Build follow-up prompt for self-verification.

    Args:
        current_answer: The answer from the trace being verified
        top_n_data: Dict mapping answers to {"score": float, "metric_name": str, "summary": str}
        if_sf: If True, use naive self-refinement (no pooling context)
    """
    if if_sf:
        # BASELINE: NAIVE SELF-REFINEMENT (no candidates shown)
        prompt_body = f"""
Previously, you concluded the answer was: {current_answer}

### SELF-CORRECTION PROTOCOL:
1. **Critical Review**: Carefully re-examine your previous reasoning step-by-step.
2. **Error Detection**: Check for potential calculation slips, misinterpretations of the question, or logical gaps.
3. **Verification**: Independently verify your conclusion. If you find a flaw, correct your path.
"""
    else:
        # ADVANCED: POOL INFORMATION BASED FOLLOW-UP
        top_4_list = list(top_n_data.keys())
        is_in_top_n = current_answer in top_4_list

        # Build candidate display text with reasoning summaries
        candidate_display_text = ""
        for i, ans in enumerate(top_4_list, 1):
            info = top_n_data[ans]
            candidate_display_text += f"\n[Candidate {i}]: {ans} ({info['metric_name']}: {info['score']:.4f})\n"
            candidate_display_text += f"Reasoning Summary: {info['summary']}\n"

       
        #print(candidate_display_text)
        if is_in_top_n:
            # Path A: In top-n (PEER REVIEW)
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
            # Path B: Not in top-n (EXTERNAL ARBITRATION - more aggressive)
            prompt_body = f"""
Previously, you concluded the answer was: {current_answer}

**URGENT DIAGNOSTIC REQUIRED:** Our multi-path analysis indicates your previous answer is highly likely to be incorrect, as it failed to align with the high-confidence consensus found in multiple independent reasoning paths. **The correct solution is almost certainly contained within the alternatives listed below:**
{candidate_display_text}

### CORRECTION PROTOCOL:
1. **Flaw Identification**: Treat your previous reasoning as having a confirmed logical derailment. Your task is to find that specific error by contrasting it with the paths above.
2. **Path Reconstruction**: These candidates represent the most stable and consistent reasoning found across multiple independent trials. Re-verify them to identify which one represents the objective truth.
3. **Decisive Pivot**: Use the candidates above as your primary reference to correct your reasoning and provide the valid final answer.
"""

    full_prompt = prompt_body.strip() + "\n\nPut your final answer within \\boxed{}"
    return full_prompt


def split_harmony_content(text: str) -> tuple:
    """
    Split GPT-OSS output by 'assistantfinal' marker.
    Returns (reasoning, summary).
    """
    marker = "assistantfinal"
    if marker in text.lower():
        parts = re.split(marker, text, flags=re.IGNORECASE)
        summary = parts[-1].strip()
        reasoning = parts[0].strip()
        return reasoning, summary
    return "", text.strip()


def get_reasoning_summary(text: str, max_chars: int = 500) -> str:
    """Extract a summary of the reasoning from the trace text (for GPT-OSS)."""
    if not text:
        return ""

    # For GPT-OSS, use assistantfinal marker
    _, summary = split_harmony_content(text)
    if summary and summary != text.strip():
        return summary[:max_chars]

    # Fallback: Try to get the conclusion/final part of reasoning
    conclusion_markers = [
        "therefore", "thus", "hence", "so the answer",
        "the answer is", "we get", "we have", "finally"
    ]

    text_lower = text.lower()
    best_pos = len(text)

    for marker in conclusion_markers:
        pos = text_lower.rfind(marker)
        if pos != -1 and pos < best_pos:
            best_pos = pos

    # Get text from the conclusion marker onwards, or last max_chars
    if best_pos < len(text) - 50:
        summary = text[best_pos:best_pos + max_chars]
    else:
        summary = text[-max_chars:]

    return summary.strip()


@app.cls(
    image=gpu_image,
    gpu=GPU_CONFIG,
    timeout=86000,
    volumes={"/models": models_volume, "/results": results_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class Phase2PoolVerifier:
    """Modal class for Phase 2 pool information verification."""

    model_name: str = modal.parameter(default="openai/gpt-oss-120b")

    @modal.enter()
    def setup(self):
        """Initialize vLLM model on Modal container startup."""
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
            max_model_len=131072,
            kv_cache_dtype="fp8_e4m3",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f}s")

    @modal.method()
    def run_pool_verification(
        self,
        traces: List[Dict[str, Any]],
        question: str,
        ground_truth: str,
        question_id: str,
        budget: int = 256,
        num_calibration: int = 32,
        percentile: float = 80.0,
        top_n: int = 4,
        agg_method: str = "maxc",
        if_sf: bool = False,
        window_size: int = 2048,
        max_tokens: int = 64000,
        temperature: float = 0.6,
        top_p: float = 0.95,
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """
        Run pool information verification on Phase 1 traces.

        Args:
            traces: Phase 1 traces
            budget: Max number of traces to use (128 or 256)
            num_calibration: Number of traces for calibration
            percentile: Screening percentile (30 or 80)
            agg_method: Aggregation method - "maxc", "sumc", or "meanc"
            if_sf: If True, use naive self-refinement (no pooling)

        Returns list of Phase 2 trace records with follow-up responses.
        """
        from vllm import SamplingParams

        # Limit traces to budget
        traces = traces[:budget]

        print(f"\n=== Phase 2: Pool Verification for {question_id} ===")
        print(f"Input traces: {len(traces)} (budget={budget})")
        print(f"Settings: num_calibration={num_calibration}, percentile={percentile}, top_n={top_n}, agg_method={agg_method}, if_SF={if_sf}")

        # Step 1: Compute min_conf for each trace and clean answers
        for t in traces:
            t['min_conf'] = get_trace_confidence(t)
            raw_answer = t.get('answer', '')
            t['clean_answer'] = clean_answer(raw_answer) if raw_answer else None

        # Step 2: Compute screening threshold
        screening_threshold = 0.0
        if len(traces) >= num_calibration:
            calibration_confs = [t['min_conf'] for t in traces[:num_calibration]]
            screening_threshold = np.percentile(calibration_confs, percentile)
            print(f"Screening threshold (p{percentile}): {screening_threshold:.4f}")

        # Step 3: Filter high-confidence traces
        filtered_traces = [
            t for t in traces
            if t['min_conf'] >= screening_threshold and t.get('clean_answer')
        ]
        print(f"Filtered traces (conf >= {screening_threshold:.4f}): {len(filtered_traces)}")

        if not filtered_traces:
            print("[WARNING] No traces passed filtering. Returning empty results.")
            return []

        # Step 4: Group by answer and aggregate confidence scores
        answer_stats = {}
        for t in filtered_traces:
            ans = t['clean_answer']
            conf = t['min_conf']
            if ans not in answer_stats:
                answer_stats[ans] = {'confs': [], 'best_conf': 0, 'best_trace': None}
            answer_stats[ans]['confs'].append(conf)
            if conf > answer_stats[ans]['best_conf']:
                answer_stats[ans]['best_conf'] = conf
                answer_stats[ans]['best_trace'] = t

        # Compute aggregated scores based on agg_method
        label_map = {"maxc": "Max Confidence", "sumc": "Cumulative Confidence", "meanc": "Mean Confidence"}

        for ans, stats in answer_stats.items():
            confs = stats['confs']
            if agg_method == "maxc":
                stats['agg_score'] = max(confs)
            elif agg_method == "sumc":
                stats['agg_score'] = sum(confs)
            elif agg_method == "meanc":
                stats['agg_score'] = sum(confs) / len(confs)

        # Step 5: Select top N answers by aggregated score
        top_n_answers = sorted(
            answer_stats.keys(),
            key=lambda x: answer_stats[x]['agg_score'],
            reverse=True
        )[:top_n]

        # Build top_n_data with format matching pool_information.py
        top_n_data = {}
        for ans in top_n_answers:
            stats = answer_stats[ans]
            best_trace = stats['best_trace']
            summary = get_reasoning_summary(best_trace.get('text', '')) if best_trace else ''
            top_n_data[ans] = {
                'score': stats['agg_score'],
                'metric_name': label_map[agg_method],
                'summary': summary
            }

        print(f"\nTop {top_n} candidate answers ({label_map[agg_method]}):")
        for i, ans in enumerate(top_n_answers, 1):
            info = top_n_data[ans]
            print(f"  {i}. {ans[:50]}... ({info['metric_name']}: {info['score']:.4f})")

        # Step 6: Build prompts for all filtered traces
        # Max model length is 131072 tokens - skip prompts that exceed this
        MAX_MODEL_LEN = 131072

        prompts = []
        params_list = []
        metadata_list = []
        skipped_count = 0

        for trace in filtered_traces:
            current_answer = trace['clean_answer']
            trace_text = trace.get('text', '')

            follow_up = build_follow_up_prompt(current_answer, top_n_data, if_sf=if_sf)

            # Build multi-turn conversation
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": trace_text},
                {"role": "user", "content": follow_up}
            ]

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort="high",
            )

            # Check prompt length - skip if it exceeds max model length
            prompt_tokens = len(self.tokenizer.tokenize(prompt))
            if prompt_tokens > MAX_MODEL_LEN:
                skipped_count += 1
                continue

            prompts.append(prompt)
            params_list.append(SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                logprobs=20,
            ))
            metadata_list.append({
                "base_trace_id": trace.get("trace_id"),
                "base_answer": trace.get("answer"),
                "clean_base_answer": current_answer,
                "base_min_conf": trace['min_conf'],
                "is_in_top_n": current_answer in top_n_answers,
            })

        if skipped_count > 0:
            print(f"[WARNING] Skipped {skipped_count} traces due to prompt length > {MAX_MODEL_LEN} tokens")

        print(f"\n[INFO] Running {len(prompts)} follow-up generations in batches of {batch_size}...")

        # Step 7: Batch generation
        all_results = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_params = params_list[i:i + batch_size]
            batch_meta = metadata_list[i:i + batch_size]

            print(f"  Batch {i//batch_size + 1}: traces {i} to {i + len(batch_prompts)}")

            start_time = time.time()
            vllm_outputs = self.llm.generate(batch_prompts, batch_params)
            batch_time = time.time() - start_time
            print(f"    Completed in {batch_time:.1f}s")

            for idx, output in enumerate(vllm_outputs):
                meta = batch_meta[idx]

                if not output.outputs:
                    continue

                trace_2_text = output.outputs[0].text
                raw_logprobs = output.outputs[0].logprobs

                # Compute confidence for phase 2 response
                token_confs = calculate_token_confs_from_logprobs(raw_logprobs)
                group_confs_2 = compute_group_confidence(token_confs, window_size)

                # Extract new answer
                new_answer = extract_answer(trace_2_text)
                clean_new_answer = clean_answer(new_answer) if new_answer else None

                # Check if answer changed
                answer_changed = clean_new_answer != meta['clean_base_answer']

                # Check correctness
                is_correct_before = clean_answer(ground_truth) == meta['clean_base_answer']
                is_correct_after = clean_answer(ground_truth) == clean_new_answer

                # Count tokens in trace_2
                trace2_token_length = len(self.tokenizer.tokenize(trace_2_text))

                # Output format matching pool_information.py exactly
                result = {
                    "base_trace_id": meta["base_trace_id"],
                    "base_answer": meta["base_answer"],
                    "is_topn": meta["is_in_top_n"],
                    "trace_2": trace_2_text,
                    "group_confidences_2": group_confs_2,
                    "trace2_token_length": trace2_token_length,
                }
                all_results.append(result)

        # Print summary
        print(f"\n--- Phase 2 Summary ---")
        print(f"Total processed: {len(all_results)}")
        if all_results:
            avg_tokens = sum(r['trace2_token_length'] for r in all_results) / len(all_results)
            print(f"Avg trace2 token length: {avg_tokens:.0f}")
        print("-" * 40)

        return all_results


@app.function(
    image=gpu_image,
    timeout=86000,
    volumes={"/results": results_volume},
)
def run_phase2(
    traces_file: Optional[str] = None,
    traces_dir: Optional[str] = None,
    dataset_prefix: Optional[str] = None,
    qid: Optional[int] = None,
    start_qid: Optional[int] = None,
    end_qid: Optional[int] = None,
    budget: int = 256,
    num_calibration: int = 32,
    percentile: float = 80.0,
    top_n: int = 4,
    agg_method: str = "maxc",
    if_sf: bool = False,
    window_size: int = 2048,
    max_tokens: int = 64000,
    temperature: float = 0.6,
    top_p: float = 0.95,
    batch_size: int = 32,
    model_name: str = "openai/gpt-oss-120b",
    output_dir: str = "/results/phase2_jsonl",
) -> str:
    """
    Run Phase 2 pool verification.

    Can process a single file or multiple files from a directory.
    """
    # Reload volume to see latest files
    results_volume.reload()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Determine which files to process
    files_to_process = []

    if traces_file:
        files_to_process.append(traces_file)
    elif traces_dir and dataset_prefix:
        if qid is not None:
            files_to_process.append(f"{traces_dir}/{dataset_prefix}_{qid:03d}.jsonl")
        elif start_qid is not None or end_qid is not None:
            # Find all matching files first
            import glob
            pattern = f"{traces_dir}/{dataset_prefix}_*.jsonl"
            all_files = sorted(glob.glob(pattern))

            # Filter by qid range
            for f in all_files:
                # Extract qid from filename like "aime_2025_014.jsonl"
                basename = os.path.basename(f).replace('.jsonl', '')
                parts = basename.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    file_qid = int(parts[1])
                    if start_qid is not None and file_qid < start_qid:
                        continue
                    if end_qid is not None and file_qid >= end_qid:
                        continue
                    files_to_process.append(f)
        else:
            # Find all matching files
            import glob
            pattern = f"{traces_dir}/{dataset_prefix}_*.jsonl"
            files_to_process = sorted(glob.glob(pattern))

    print(f"[INFO] Files to process: {len(files_to_process)}")
    for f in files_to_process[:5]:
        print(f"  - {f}")
    if len(files_to_process) > 5:
        print(f"  ... and {len(files_to_process) - 5} more")

    # Create verifier
    verifier = Phase2PoolVerifier(model_name=model_name)

    output_files = []

    skipped_existing = 0
    for trace_file in files_to_process:
        if not os.path.exists(trace_file):
            print(f"[WARNING] File not found: {trace_file}, skipping...")
            continue

        # Check if output file already exists - skip if so
        base_name = os.path.basename(trace_file).replace('.jsonl', '')
        exp_suffix = f"_b{budget}_p{int(percentile)}"
        if if_sf:
            exp_suffix += "_SF"
        else:
            exp_suffix += f"_{agg_method}"
        output_filename = f"{base_name}_phase2{exp_suffix}.jsonl"
        output_path = f"{output_dir}/{output_filename}"

        if os.path.exists(output_path):
            skipped_existing += 1
            print(f"[SKIP] Output already exists: {output_filename}")
            output_files.append(output_path)
            continue

        # Load traces
        print(f"\n[INFO] Loading traces from {trace_file}")
        traces = []
        with open(trace_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        traces.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not traces:
            print(f"[WARNING] No traces loaded from {trace_file}, skipping...")
            continue

        # Get question info from first trace
        question = traces[0].get('question', '')
        ground_truth = traces[0].get('ground_truth', '')
        question_id = traces[0].get('question_id', os.path.basename(trace_file).replace('.jsonl', ''))

        # Run verification
        results = verifier.run_pool_verification.remote(
            traces=traces,
            question=question,
            ground_truth=ground_truth,
            question_id=question_id,
            budget=budget,
            num_calibration=num_calibration,
            percentile=percentile,
            top_n=top_n,
            agg_method=agg_method,
            if_sf=if_sf,
            window_size=window_size,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            batch_size=batch_size,
        )

        # Save results (output_path already computed above for skip check)
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')

        results_volume.commit()
        print(f"[SUCCESS] Wrote {len(results)} phase2 results to {output_path}")
        output_files.append(output_path)

    if skipped_existing > 0:
        print(f"\n[INFO] Skipped {skipped_existing} files (already completed)")
    return f"Phase 2 completed for {len(output_files)} files ({skipped_existing} skipped as already done): {output_files}"


@app.local_entrypoint()
def main(
    traces_file: Optional[str] = None,
    traces_dir: str = "/results/traces_jsonl",
    dataset_prefix: Optional[str] = None,
    qid: Optional[int] = None,
    start_qid: Optional[int] = None,
    end_qid: Optional[int] = None,
    budget: int = 256,
    num_calibration: int = 32,
    percentile: float = 80.0,
    top_n: int = 4,
    agg_method: str = "maxc",
    if_sf: bool = False,
    window_size: int = 2048,
    max_tokens: int = 64000,
    temperature: float = 0.6,
    top_p: float = 0.95,
    batch_size: int = 32,
    model: str = "openai/gpt-oss-120b",
    output_dir: str = "/results/phase2_jsonl",
):
    """
    Phase 2: Pool Information Self-Verification

    8 experiment settings by controlling:
    - budget: 128 or 256
    - percentile: 30 or 80
    - if_sf: True (naive self-refinement) or False (pool information)

    Examples:
        # Pool information with budget=256, percentile=80
        modal run modal_phase2.py --traces-dir /results/traces_jsonl --dataset-prefix aime_2025 --budget 256 --percentile 80

        # Naive self-refinement baseline
        modal run modal_phase2.py --traces-dir /results/traces_jsonl --dataset-prefix aime_2025 --if-sf

        # Different aggregation method
        modal run modal_phase2.py --traces-dir /results/traces_jsonl --dataset-prefix aime_2025 --agg-method sumc
    """
    print("=" * 80)
    print("Phase 2: Pool Information Self-Verification")
    print("=" * 80)
    print(f"Traces dir: {traces_dir}")
    print(f"Dataset prefix: {dataset_prefix}")
    print(f"Model: {model}")
    print(f"Budget: {budget}, Num calibration: {num_calibration}")
    print(f"Percentile: {percentile}, Top N: {top_n}, Agg method: {agg_method}")
    print(f"Self-refinement (no pooling): {if_sf}")
    print(f"Max tokens: {max_tokens}")
    print(f"Output dir: {output_dir}")
    print("=" * 80)

    result = run_phase2.remote(
        traces_file=traces_file,
        traces_dir=traces_dir,
        dataset_prefix=dataset_prefix,
        qid=qid,
        start_qid=start_qid,
        end_qid=end_qid,
        budget=budget,
        num_calibration=num_calibration,
        percentile=percentile,
        top_n=top_n,
        agg_method=agg_method,
        if_sf=if_sf,
        window_size=window_size,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size,
        model_name=model,
        output_dir=output_dir,
    )

    print("\n" + "=" * 80)
    print("Result:", result)
    print("=" * 80)
