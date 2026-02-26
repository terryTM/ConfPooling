"""
Modal-based runner for comparing MV@256, Online 10%, and Pooling methods
Uses GPT OSS 120B (or other large models) with distributed compute

Run with:
    modal run modal_runner.py --dataset aime_2024.jsonl --method all --output-dir results/

For specific methods:
    modal run modal_runner.py --dataset aime_2024.jsonl --method mv256
    modal run modal_runner.py --dataset aime_2024.jsonl --method online
    modal run modal_runner.py --dataset aime_2024.jsonl --method pooling
"""
import modal
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import time

# Define Modal app
app = modal.App("confidence-pooling-benchmark")

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
)

# Modal Volume for caching models
models_volume = modal.Volume.from_name("models-cache", create_if_missing=True)

# GPU configuration for GPT OSS 120B
# Adjust GPU count/type based on model requirements
GPU_CONFIG = "H200:1"  # 4xA100 80GB for 120B model


@dataclass
class BenchmarkResult:
    """Result for a single question across all methods"""
    qid: int
    question: str
    ground_truth: str

    # MV@256 results
    mv256_answer: Optional[str] = None
    mv256_correct: Optional[bool] = None
    mv256_tokens: Optional[int] = None
    mv256_time: Optional[float] = None
    mv256_traces_count: Optional[int] = None

    # Online 10% results
    online_answer: Optional[str] = None
    online_correct: Optional[bool] = None
    online_tokens: Optional[int] = None
    online_time: Optional[float] = None
    online_survivors: Optional[int] = None
    online_warmup_traces: Optional[int] = None
    online_conf_threshold: Optional[float] = None

    # Pooling results
    pooling_answer: Optional[str] = None
    pooling_correct: Optional[bool] = None
    pooling_tokens: Optional[int] = None
    pooling_time: Optional[float] = None
    pooling_candidates_count: Optional[int] = None
    pooling_followup_tokens: Optional[int] = None

    # Error tracking
    errors: Dict[str, str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = {}


@app.cls(
    image=gpu_image,
    gpu=GPU_CONFIG,
    timeout=3600,  # 1 hour timeout per question
    volumes={"/models": models_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],  # If needed for model access
)
class DeepThinkRunner:
    """Modal class for running deep thinking methods with vLLM"""

    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.model_name = model_name

    @modal.enter()
    def setup(self):
        """Initialize vLLM model on Modal container startup"""
        import sys
        sys.path.insert(0, "/root")

        from deepconf import DeepThinkLLM
        from transformers import AutoTokenizer

        print(f"Loading model: {self.model_name}")
        start_time = time.time()

        # Initialize with tensor parallelism across all GPUs
        self.deep_llm = DeepThinkLLM(
            model=self.model_name,
            tensor_parallel_size=1,  # Match GPU count (H200:1)
            enable_prefix_caching=True,
            trust_remote_code=True,
            download_dir="/models",  # Use persistent volume
        )

        self.tokenizer = self.deep_llm.tokenizer

        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f}s")

    def _prepare_prompt(self, question: str) -> str:
        """Prepare prompt using model's chat template"""
        messages = [
            {"role": "user", "content": question}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt

    def _equal_func(self, answer: str, ground_truth: str) -> bool:
        """Check if answer equals ground truth"""
        # Simple string comparison for now
        # You can integrate dynasor.core.evaluator.math_equal if needed
        answer_clean = str(answer).strip().lower()
        gt_clean = str(ground_truth).strip().lower()
        return answer_clean == gt_clean

    @modal.method()
    def run_mv256(self, question: str, ground_truth: str, qid: int) -> Dict[str, Any]:
        """Run Majority Voting with 256 full traces (offline mode)"""
        from vllm import SamplingParams

        print(f"\n=== Running MV@256 for QID {qid} ===")

        try:
            prompt = self._prepare_prompt(question)

            sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.95,
                top_k=40,
                max_tokens=64000,
                logprobs=20,
            )

            start_time = time.time()

            result = self.deep_llm.deepthink(
                prompt=prompt,
                mode="offline",
                budget=256,
                window_size=2048,
                sampling_params=sampling_params,
                compute_multiple_voting=True
            )

            elapsed_time = time.time() - start_time

            # Get majority vote answer
            final_answer = result.voted_answer
            is_correct = self._equal_func(final_answer, ground_truth) if final_answer else False

            return {
                "answer": final_answer,
                "correct": is_correct,
                "tokens": result.total_tokens,
                "time": elapsed_time,
                "traces_count": result.total_traces_count,
                "voting_results": result.voting_results,
            }

        except Exception as e:
            print(f"Error in MV256: {str(e)}")
            return {"error": str(e)}

    @modal.method()
    def run_online(
        self,
        question: str,
        ground_truth: str,
        qid: int,
        warmup_traces: int = 64,
        total_budget: int = 256,
        confidence_percentile: int = 10
    ) -> Dict[str, Any]:
        """Run Online mode with confidence-based early stopping (10% threshold)"""
        from vllm import SamplingParams

        print(f"\n=== Running Online 10% for QID {qid} ===")

        try:
            prompt = self._prepare_prompt(question)

            sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.95,
                top_k=40,
                max_tokens=64000,
                logprobs=20,
            )

            start_time = time.time()

            result = self.deep_llm.deepthink(
                prompt=prompt,
                mode="online",
                warmup_traces=warmup_traces,
                total_budget=total_budget,
                confidence_percentile=confidence_percentile,
                window_size=2048,
                sampling_params=sampling_params,
                compute_multiple_voting=True
            )

            elapsed_time = time.time() - start_time

            # Get voted answer from survivors
            final_answer = result.voted_answer
            is_correct = self._equal_func(final_answer, ground_truth) if final_answer else False

            # Count survivors (traces that passed threshold)
            survivors = len(result.all_voting_traces) if hasattr(result, 'all_voting_traces') else 0

            return {
                "answer": final_answer,
                "correct": is_correct,
                "tokens": result.total_tokens,
                "time": elapsed_time,
                "survivors": survivors,
                "warmup_traces": warmup_traces,
                "conf_threshold": result.conf_bar,
                "voting_results": result.voting_results,
                "warmup_tokens": result.warmup_tokens,
                "final_tokens": result.final_tokens,
            }

        except Exception as e:
            print(f"Error in Online: {str(e)}")
            return {"error": str(e)}

    @modal.method()
    def run_pooling(
        self,
        question: str,
        ground_truth: str,
        qid: int,
        warmup_traces: int = 64,
        total_budget: int = 256,
        confidence_percentile: int = 10
    ) -> Dict[str, Any]:
        """
        Run Pooling method: Online screening + consensus-aware refinement

        This implements the paper's Algorithm 1:
        1. Run online screening to get survivors
        2. Build information packet with top-5 answers
        3. Perform second-round reflection for each trace
        4. Weighted aggregation with revised answers
        """
        from vllm import SamplingParams
        from collections import Counter, defaultdict
        import re

        print(f"\n=== Running Pooling for QID {qid} ===")

        try:
            prompt = self._prepare_prompt(question)

            sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.95,
                top_k=40,
                max_tokens=64000,
                logprobs=20,
            )

            # Step 1: Run online screening (same as online method)
            start_time = time.time()

            result = self.deep_llm.deepthink(
                prompt=prompt,
                mode="online",
                warmup_traces=warmup_traces,
                total_budget=total_budget,
                confidence_percentile=confidence_percentile,
                window_size=2048,
                sampling_params=sampling_params,
                compute_multiple_voting=False  # We'll do our own pooling
            )

            screening_time = time.time() - start_time

            # Step 2: Build information packet (top-5 answers with support counts)
            pooling_start = time.time()

            traces = result.all_voting_traces if hasattr(result, 'all_voting_traces') else result.all_traces

            # Count answer frequencies
            answer_counts = Counter()
            answer_to_traces = defaultdict(list)

            for trace in traces:
                ans = trace.get('extracted_answer')
                if ans:
                    answer_counts[ans] += 1
                    answer_to_traces[ans].append(trace)

            # Get top-5 answers
            top5_answers = [ans for ans, _ in answer_counts.most_common(5)]

            # Build information packet (simplified version without LLM followup)
            # In full implementation, you'd generate followup reasoning here
            info_packet = []
            for ans in top5_answers:
                count = answer_counts[ans]
                # Sample one representative trace
                sample_trace = answer_to_traces[ans][0]
                reasoning_summary = sample_trace.get('text', '')[:500]  # Truncate for brevity

                info_packet.append({
                    'answer': ans,
                    'count': count,
                    'reasoning': reasoning_summary
                })

            # Step 3: Consensus-conditioned reflection
            # For this baseline, we use confidence-weighted voting with awareness of top answers
            # Full implementation would involve LLM followup prompts as in the paper

            # Simple pooling: weight by confidence and support count
            weighted_votes = defaultdict(float)

            for trace in traces:
                ans = trace.get('extracted_answer')
                if ans:
                    conf = trace.get('min_conf', 1.0)
                    support_weight = answer_counts[ans] / len(traces)  # Consensus boost

                    weighted_votes[ans] += conf * (1 + support_weight)

            # Get final answer
            if weighted_votes:
                final_answer = max(weighted_votes.items(), key=lambda x: x[1])[0]
            else:
                final_answer = None

            pooling_time = time.time() - pooling_start
            total_time = screening_time + pooling_time

            is_correct = self._equal_func(final_answer, ground_truth) if final_answer else False

            return {
                "answer": final_answer,
                "correct": is_correct,
                "tokens": result.total_tokens,
                "time": total_time,
                "screening_time": screening_time,
                "pooling_time": pooling_time,
                "candidates_count": len(top5_answers),
                "info_packet": info_packet,
                "survivors": len(traces),
                "conf_threshold": result.conf_bar,
                "followup_tokens": 0,  # Placeholder for full implementation
            }

        except Exception as e:
            print(f"Error in Pooling: {str(e)}")
            return {"error": str(e)}


@app.function(
    image=gpu_image.pip_install("pandas", "tabulate").add_local_dir("data", "/root/data"),
    timeout=7200,  # 2 hours for full benchmark
)
def run_benchmark(
    dataset_path: str,
    methods: List[str],
    model_name: str = "openai/gpt-oss-120b",
    output_dir: str = "results",
    start_qid: int = 0,
    end_qid: Optional[int] = None,
) -> List[Dict]:
    """
    Run benchmark comparing all specified methods

    Args:
        dataset_path: Path to JSONL dataset file
        methods: List of methods to run ["mv256", "online", "pooling"]
        model_name: Model to use
        output_dir: Directory to save results
        start_qid: Starting question ID
        end_qid: Ending question ID (None = all questions)
    """
    from pathlib import Path
    import json
    from datetime import datetime

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
    print(f"Running benchmark on {len(questions)} questions (QID {start_qid}-{end_qid-1})")
    print(f"Methods: {methods}")
    print(f"Model: {model_name}")

    # Initialize runner
    runner = DeepThinkRunner(model_name=model_name)

    results = []

    for idx, q_data in enumerate(questions):
        qid = start_qid + idx
        question = q_data['question']
        ground_truth = str(q_data.get('answer', '')).strip()

        print(f"\n{'='*80}")
        print(f"Question {qid}/{end_qid-1}: {question[:100]}...")
        print(f"Ground truth: {ground_truth}")
        print(f"{'='*80}")

        result = BenchmarkResult(
            qid=qid,
            question=question,
            ground_truth=ground_truth
        )

        # Run MV@256
        if "mv256" in methods:
            print(f"\n[{qid}] Running MV@256...")
            try:
                mv_result = runner.run_mv256.remote(question, ground_truth, qid)

                if "error" in mv_result:
                    result.errors["mv256"] = mv_result["error"]
                else:
                    result.mv256_answer = mv_result["answer"]
                    result.mv256_correct = mv_result["correct"]
                    result.mv256_tokens = mv_result["tokens"]
                    result.mv256_time = mv_result["time"]
                    result.mv256_traces_count = mv_result["traces_count"]

                    print(f"  Answer: {result.mv256_answer}")
                    print(f"  Correct: {result.mv256_correct}")
                    print(f"  Tokens: {result.mv256_tokens:,}")
                    print(f"  Time: {result.mv256_time:.2f}s")

            except Exception as e:
                result.errors["mv256"] = str(e)
                print(f"  Error: {str(e)}")

        # Run Online 10%
        if "online" in methods:
            print(f"\n[{qid}] Running Online 10%...")
            try:
                online_result = runner.run_online.remote(question, ground_truth, qid)

                if "error" in online_result:
                    result.errors["online"] = online_result["error"]
                else:
                    result.online_answer = online_result["answer"]
                    result.online_correct = online_result["correct"]
                    result.online_tokens = online_result["tokens"]
                    result.online_time = online_result["time"]
                    result.online_survivors = online_result["survivors"]
                    result.online_warmup_traces = online_result["warmup_traces"]
                    result.online_conf_threshold = online_result["conf_threshold"]

                    print(f"  Answer: {result.online_answer}")
                    print(f"  Correct: {result.online_correct}")
                    print(f"  Tokens: {result.online_tokens:,}")
                    print(f"  Survivors: {result.online_survivors}")
                    print(f"  Time: {result.online_time:.2f}s")

            except Exception as e:
                result.errors["online"] = str(e)
                print(f"  Error: {str(e)}")

        # Run Pooling
        if "pooling" in methods:
            print(f"\n[{qid}] Running Pooling...")
            try:
                pooling_result = runner.run_pooling.remote(question, ground_truth, qid)

                if "error" in pooling_result:
                    result.errors["pooling"] = pooling_result["error"]
                else:
                    result.pooling_answer = pooling_result["answer"]
                    result.pooling_correct = pooling_result["correct"]
                    result.pooling_tokens = pooling_result["tokens"]
                    result.pooling_time = pooling_result["time"]
                    result.pooling_candidates_count = pooling_result["candidates_count"]
                    result.pooling_followup_tokens = pooling_result["followup_tokens"]

                    print(f"  Answer: {result.pooling_answer}")
                    print(f"  Correct: {result.pooling_correct}")
                    print(f"  Tokens: {result.pooling_tokens:,}")
                    print(f"  Candidates: {result.pooling_candidates_count}")
                    print(f"  Time: {result.pooling_time:.2f}s")

            except Exception as e:
                result.errors["pooling"] = str(e)
                print(f"  Error: {str(e)}")

        results.append(asdict(result))

        # Save intermediate results
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        intermediate_file = f"{output_dir}/intermediate_{timestamp}.jsonl"

        with open(intermediate_file, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')

    # Save final results
    final_file = f"{output_dir}/benchmark_results_{timestamp}.jsonl"
    with open(final_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    print(f"\n{'='*80}")
    print(f"Benchmark complete! Results saved to {final_file}")
    print(f"{'='*80}")

    # Print summary
    print_summary(results, methods)

    return results


def print_summary(results: List[Dict], methods: List[str]):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    total = len(results)

    for method in methods:
        if method == "mv256":
            prefix = "mv256"
        elif method == "online":
            prefix = "online"
        elif method == "pooling":
            prefix = "pooling"
        else:
            continue

        correct_key = f"{prefix}_correct"
        tokens_key = f"{prefix}_tokens"
        time_key = f"{prefix}_time"

        valid_results = [r for r in results if r.get(correct_key) is not None]

        if not valid_results:
            print(f"\n{method.upper()}: No valid results")
            continue

        accuracy = sum(1 for r in valid_results if r[correct_key]) / len(valid_results)
        avg_tokens = sum(r[tokens_key] for r in valid_results if r.get(tokens_key)) / len(valid_results)
        avg_time = sum(r[time_key] for r in valid_results if r.get(time_key)) / len(valid_results)

        print(f"\n{method.upper()}:")
        print(f"  Accuracy: {accuracy:.1%} ({sum(1 for r in valid_results if r[correct_key])}/{len(valid_results)})")
        print(f"  Avg Tokens: {avg_tokens:,.0f}")
        print(f"  Avg Time: {avg_time:.2f}s")

        if method == "online" and valid_results:
            avg_survivors = sum(r.get('online_survivors', 0) for r in valid_results) / len(valid_results)
            print(f"  Avg Survivors: {avg_survivors:.1f}")

            # Token efficiency vs MV256
            mv_tokens = [r.get('mv256_tokens') for r in valid_results if r.get('mv256_tokens')]
            if mv_tokens:
                avg_mv_tokens = sum(mv_tokens) / len(mv_tokens)
                efficiency = (1 - avg_tokens / avg_mv_tokens) * 100
                print(f"  Token Savings vs MV256: {efficiency:.1f}%")

        if method == "pooling" and valid_results:
            # Compare to both MV256 and Online
            mv_tokens = [r.get('mv256_tokens') for r in valid_results if r.get('mv256_tokens')]
            if mv_tokens:
                avg_mv_tokens = sum(mv_tokens) / len(mv_tokens)
                efficiency = (1 - avg_tokens / avg_mv_tokens) * 100
                print(f"  Token Savings vs MV256: {efficiency:.1f}%")


@app.local_entrypoint()
def main(
    dataset: str = "data/raw/aime_2024.jsonl",
    method: str = "all",
    model: str = "openai/gpt-oss-120b",
    output_dir: str = "results",
    start_qid: int = 0,
    end_qid: int = None,
):
    """
    Main entry point for Modal benchmark

    Examples:
        # Run all methods on full dataset
        modal run modal_runner.py --dataset aime_2024.jsonl --method all

        # Run only MV256 on first 10 questions
        modal run modal_runner.py --dataset aime_2024.jsonl --method mv256 --end-qid 10

        # Run online and pooling methods
        modal run modal_runner.py --dataset aime_2024.jsonl --method online,pooling
    """

    # Parse methods
    if method == "all":
        methods = ["mv256", "online", "pooling"]
    else:
        methods = [m.strip() for m in method.split(",")]

    # Validate methods
    valid_methods = {"mv256", "online", "pooling"}
    for m in methods:
        if m not in valid_methods:
            raise ValueError(f"Invalid method: {m}. Valid methods: {valid_methods}")

    print(f"Starting benchmark with methods: {methods}")

    # Run benchmark
    results = run_benchmark.remote(
        dataset_path=dataset,
        methods=methods,
        model_name=model,
        output_dir=output_dir,
        start_qid=start_qid,
        end_qid=end_qid
    )

    print("\nBenchmark completed successfully!")
    print(f"Results saved to {output_dir}/")
