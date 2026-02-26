"""
Phase 1: Generate 256 traces per question and save to disk

This script:
1. Loads questions from JSONL dataset
2. Generates 256 reasoning traces per question with full logprobs
3. Saves all traces, answers, confidences, and logprobs to disk
4. Minimal GPU usage - only generation, no repeated sampling

Usage:
    modal run modal_runner_phase1.py --dataset data/raw/aime_2024.jsonl --output-dir results/traces
    modal run modal_runner_phase1.py --dataset data/raw/aime_2024.jsonl --start-qid 0 --end-qid 5
"""
import modal
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import time

# Define Modal app
app = modal.App("confidence-pooling-phase1")

# Define GPU image with all dependencies
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.10.2",
        "transformers>=4.46.0",
        "torch>=2.5.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
    )
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
GPU_CONFIG = "H200:1"  # Single H200 for 120B model


@dataclass
class TraceData:
    """Complete data for a single reasoning trace"""
    trace_id: int
    text: str  # Full generated reasoning text
    extracted_answer: str
    group_confidence: List[float]  # DeepConf group confidence (sliding window)
    lowest_group_conf: float  # LGC = min(group_confidence) - key metric from DeepConf
    tokens: int
    token_ids: List[int]  # All token IDs


@dataclass
class QuestionTraces:
    """All 256 traces for a single question"""
    qid: int
    question: str
    ground_truth: str
    traces: List[Dict]  # List of TraceData dicts
    answer_groups: Dict[str, Dict]  # Group confidences by answer (C₁)
    generation_time: float
    total_tokens: int
    timestamp: str


@app.cls(
    image=gpu_image,
    gpu=GPU_CONFIG,
    timeout=20000,
    volumes={"/models": models_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class TraceGenerator:
    """Generate and save 256 traces per question"""

    model_name: str = modal.parameter(default="openai/gpt-oss-120b")

    @modal.enter()
    def setup(self):
        """Initialize vLLM model"""
        import sys
        sys.path.insert(0, "/root")

        from deepconf import DeepThinkLLM

        print(f"Loading model: {self.model_name}")
        start_time = time.time()

        self.deep_llm = DeepThinkLLM(
            model=self.model_name,
            tensor_parallel_size=1,  # Match GPU count (H200:1)
            enable_prefix_caching=True,
            trust_remote_code=True,
            download_dir="/models",
        )

        self.tokenizer = self.deep_llm.tokenizer
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f}s")

    def compute_deepconf_group_confidence(
        self, raw_logprobs: list, window_size: int = 2048
    ) -> List[float]:
        """
        Compute Group Confidence per DeepConf paper Appendix G.

        Logic:
        1. Token Confidence: Exclude the highest logprob, average the rest (negative mean)
        2. Group Confidence: Sliding window mean over token confidences

        Args:
            raw_logprobs: vLLM's trace_output.logprobs (List of dicts)
            window_size: Sliding window size, paper default is 2048

        Returns:
            List[float]: Mean confidence for each complete sliding window
        """
        token_confs = []

        # Step 1: Compute token confidence for each step
        for step_logprobs_dict in raw_logprobs:
            if not step_logprobs_dict:
                continue

            # Extract all candidate token logprobs and sort descending
            all_lps = sorted(
                [lp.logprob for lp in step_logprobs_dict.values()],
                reverse=True
            )

            # Key: Exclude logprobs[0] (highest prob token)
            # Compute negative mean of remaining candidates
            if len(all_lps) > 1:
                new_conf = -sum(all_lps[1:]) / len(all_lps[1:])
            else:
                # No other candidates - assign low confidence
                new_conf = 0.0

            token_confs.append(new_conf)

        # Step 2: Compute sliding window mean (Group Confidence)
        # Paper requires full window length
        if len(token_confs) < window_size:
            # If total length < window, no signal produced (per paper)
            return []

        group_confs = []
        # Initialize first complete window sum
        current_sum = sum(token_confs[:window_size])
        group_confs.append(current_sum / window_size)

        # Incremental sliding window update
        for i in range(len(token_confs) - window_size):
            current_sum = current_sum - token_confs[i] + token_confs[i + window_size]
            group_confs.append(current_sum / window_size)

        return group_confs

    @modal.method()
    def generate_traces(
        self,
        question: str,
        qid: int,
        ground_truth: str,
        num_traces: int = 256,
    ) -> Dict:
        """Generate num_traces reasoning traces for a question"""
        from vllm import SamplingParams
        from datetime import datetime
        import re

        print(f"\n{'='*80}")
        print(f"QID {qid}: Generating {num_traces} traces")
        print(f"Question: {question[:100]}...")
        print(f"{'='*80}\n")

        # Prepare prompt
        messages = [{"role": "user", "content": question}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Sampling parameters for diverse traces
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=64000,  # Allow long reasoning
            logprobs=20,  # Get top-20 logprobs for confidence calculation
        )

        # Generate all traces in one batch
        start_time = time.time()

        # Create batch of prompts (all identical)
        prompts = [prompt] * num_traces

        print(f"Generating {num_traces} traces...")
        outputs = self.deep_llm.llm.generate(prompts, sampling_params)

        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f}s")

        # Extract trace data
        traces = []
        total_tokens = 0

        for trace_id, output in enumerate(outputs):
            result = output.outputs[0]

            # Extract answer using \\boxed{} pattern
            text = result.text
            answer_match = re.search(r'\\boxed\{([^}]+)\}', text)
            extracted_answer = answer_match.group(1) if answer_match else ""

            # Calculate DeepConf group confidence from logprobs
            if result.logprobs:
                group_conf = self.compute_deepconf_group_confidence(
                    result.logprobs, window_size=2048
                )
                # LGC (Lowest Group Confidence) - key metric from DeepConf paper
                lowest_group_conf = min(group_conf) if group_conf else 0.0
            else:
                group_conf = []
                lowest_group_conf = 0.0

            trace_data = TraceData(
                trace_id=trace_id,
                text=text,
                extracted_answer=extracted_answer,
                group_confidence=group_conf,
                lowest_group_conf=lowest_group_conf,
                tokens=len(result.token_ids),
                token_ids=result.token_ids,
            )

            traces.append(asdict(trace_data))
            total_tokens += len(result.token_ids)

        # Calculate group confidences (C₁) - group by answer
        from collections import defaultdict
        answer_groups = defaultdict(lambda: {
            'traces': [],
            'count': 0,
            'lgc_values': [],  # Lowest Group Confidence per trace
        })

        for trace in traces:
            answer = trace['extracted_answer']
            answer_groups[answer]['traces'].append(trace['trace_id'])
            answer_groups[answer]['count'] += 1
            answer_groups[answer]['lgc_values'].append(trace['lowest_group_conf'])

        # Compute aggregate statistics for each answer group
        group_confidences = {}
        for answer, group_data in answer_groups.items():
            lgc_vals = group_data['lgc_values']
            group_confidences[answer] = {
                'count': group_data['count'],
                'trace_ids': group_data['traces'],
                # LGC-based statistics (DeepConf paper metrics)
                'mean_lgc': sum(lgc_vals) / len(lgc_vals),
                'max_lgc': max(lgc_vals),
                'min_lgc': min(lgc_vals),
                # Weighted vote score: sum of LGC values for this answer
                'total_lgc_weight': sum(lgc_vals),
            }

        # Package all data
        question_data = QuestionTraces(
            qid=qid,
            question=question,
            ground_truth=ground_truth,
            traces=traces,
            answer_groups=group_confidences,
            generation_time=generation_time,
            total_tokens=total_tokens,
            timestamp=datetime.now().isoformat(),
        )

        print(f"\n✓ Generated {len(traces)} traces")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Avg tokens/trace: {total_tokens/len(traces):.0f}")
        print(f"  Time: {generation_time:.2f}s")
        print(f"  Throughput: {total_tokens/generation_time:.0f} tokens/s")
        print(f"\n  Answer groups: {len(group_confidences)}")
        for answer, stats in sorted(group_confidences.items(), key=lambda x: x[1]['count'], reverse=True)[:5]:
            print(f"    {answer}: {stats['count']} traces (mean_lgc: {stats['mean_lgc']:.3f}, total_weight: {stats['total_lgc_weight']:.3f})")
        print()

        return asdict(question_data)


@app.function(
    image=gpu_image.pip_install("pandas"),
    timeout=15000,
    volumes={"/results": results_volume},
)
def generate_all_traces(
    dataset_path: str,
    output_dir: str = "results/traces",
    start_qid: int = 0,
    end_qid: Optional[int] = None,
    model_name: str = "openai/gpt-oss-120b",
    num_traces: int = 256,
) -> List[Dict]:
    """Generate traces for all questions in dataset"""
    from datetime import datetime
    import json

    # Load dataset (prepend /root/ for Modal mounts)
    if not dataset_path.startswith('/'):
        dataset_path = f"/root/{dataset_path}"

    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r') as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]

    # Select question range
    if end_qid is None:
        end_qid = len(data)

    questions = data[start_qid:end_qid]
    print(f"\n{'#'*80}")
    print(f"PHASE 1: Generating {num_traces} traces for {len(questions)} questions")
    print(f"QID range: {start_qid}-{end_qid-1}")
    print(f"Model: {model_name}")
    print(f"{'#'*80}\n")

    # Initialize generator
    generator = TraceGenerator(model_name=model_name)

    all_results = []

    # Use volume path for persistent storage
    volume_output_dir = f"/results/{output_dir}"
    Path(volume_output_dir).mkdir(parents=True, exist_ok=True)

    for idx, q_data in enumerate(questions):
        qid = start_qid + idx
        question = q_data['question']
        ground_truth = str(q_data.get('answer', '')).strip()

        # Generate traces
        result = generator.generate_traces.remote(
            question=question,
            qid=qid,
            ground_truth=ground_truth,
            num_traces=num_traces,
        )

        all_results.append(result)

        # Save after each question to persistent volume
        output_file = f"{volume_output_dir}/traces_qid_{qid}.json"

        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        # Commit to volume immediately so data persists even if job fails
        results_volume.commit()

        print(f"✓ Saved traces to {output_file} (committed to volume)")

    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"{volume_output_dir}/phase1_summary_{timestamp}.json"

    summary = {
        "dataset": dataset_path,
        "model": model_name,
        "num_traces_per_question": num_traces,
        "start_qid": start_qid,
        "end_qid": end_qid,
        "total_questions": len(questions),
        "timestamp": timestamp,
        "trace_files": [f"traces_qid_{start_qid + i}.json" for i in range(len(questions))],
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Final commit
    results_volume.commit()

    print(f"\n{'#'*80}")
    print(f"PHASE 1 COMPLETE!")
    print(f"{'#'*80}")
    print(f"Generated {num_traces} traces for {len(questions)} questions")
    print(f"Traces saved to volume: /results/{output_dir}/")
    print(f"Summary: {summary_file}")
    print(f"\nTo download results locally:")
    print(f"  modal volume get results-cache {output_dir} ./local_results/")
    print(f"\nNext: Run Phase 2 to analyze these traces with different methods")
    print(f"{'#'*80}\n")

    return all_results


@app.local_entrypoint()
def main(
    dataset: str = "data/raw/aime_2024.jsonl",
    output_dir: str = "results/traces",
    start_qid: int = 0,
    end_qid: int = None,
    model: str = "openai/gpt-oss-120b",
    num_traces: int = 256,
):
    """
    Phase 1: Generate and save 256 traces per question

    Examples:
        # Generate traces for all questions
        modal run modal_runner_phase1.py --dataset data/raw/aime_2024.jsonl

        # Generate for first 5 questions only
        modal run modal_runner_phase1.py --dataset data/raw/aime_2024.jsonl --end-qid 5

        # Use different model
        modal run modal_runner_phase1.py --dataset data/raw/aime_2024.jsonl --model "Qwen/Qwen2.5-Math-72B"
    """

    results = generate_all_traces.remote(
        dataset_path=dataset,
        output_dir=output_dir,
        start_qid=start_qid,
        end_qid=end_qid,
        model_name=model,
        num_traces=num_traces,
    )

    print(f"\n✓ Phase 1 completed successfully!")
    print(f"✓ Trace files saved to {output_dir}/")


# Web endpoint for deployed app - can be triggered via HTTP
@app.function(
    image=gpu_image.pip_install("pandas"),
    timeout=36000,  # 10 hours for full run
    volumes={"/results": results_volume},
)
@modal.web_endpoint(method="POST")
def run_phase1(
    dataset: str = "data/raw/aime_2024.jsonl",
    output_dir: str = "results/traces",
    start_qid: int = 0,
    end_qid: int = None,
    model: str = "openai/gpt-oss-120b",
    num_traces: int = 256,
):
    """
    Web endpoint to trigger Phase 1 generation.

    After deploying with `modal deploy modal_runner_phase1.py`, trigger via:

    curl -X POST "https://your-app-url.modal.run" \
         -H "Content-Type: application/json" \
         -d '{"start_qid": 14, "end_qid": 30}'
    """
    # Spawn the generation as a background task
    generate_all_traces.spawn(
        dataset_path=dataset,
        output_dir=output_dir,
        start_qid=start_qid,
        end_qid=end_qid,
        model_name=model,
        num_traces=num_traces,
    )

    return {
        "status": "started",
        "message": f"Phase 1 generation started for qid {start_qid} to {end_qid}",
        "check_results": "modal volume ls results-cache results/traces"
    }
