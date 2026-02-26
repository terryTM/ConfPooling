"""
Enhanced Modal runner that saves full trace outputs for detailed analysis

This version saves:
- All individual trace reasoning texts
- Per-trace answers and confidences
- Complete voting breakdowns
- All intermediate data for reproducibility

Usage:
    modal run modal_runner_with_traces.py --dataset aime_2024.jsonl --method all
"""
import modal
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
import time

# Define Modal app
app = modal.App("confidence-pooling-benchmark-traces")

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

models_volume = modal.Volume.from_name("models-cache", create_if_missing=True)
GPU_CONFIG = "H200:1"


@dataclass
class TraceInfo:
    """Information about a single reasoning trace"""
    trace_id: int
    text: str  # Full generated reasoning
    extracted_answer: str
    min_conf: float
    avg_conf: float
    tokens: int
    stop_reason: str = "length"

    def to_dict(self):
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Enhanced result with full trace information"""
    qid: int
    question: str
    ground_truth: str

    # MV@256 results
    mv256_answer: Optional[str] = None
    mv256_correct: Optional[bool] = None
    mv256_tokens: Optional[int] = None
    mv256_time: Optional[float] = None
    mv256_traces_count: Optional[int] = None
    mv256_traces: Optional[List[Dict]] = None  # NEW: All traces
    mv256_voting_breakdown: Optional[Dict] = None  # NEW: Answer frequencies

    # Online 10% results
    online_answer: Optional[str] = None
    online_correct: Optional[bool] = None
    online_tokens: Optional[int] = None
    online_time: Optional[float] = None
    online_survivors: Optional[int] = None
    online_warmup_traces: Optional[int] = None
    online_conf_threshold: Optional[float] = None
    online_survivor_traces: Optional[List[Dict]] = None  # NEW: Survivor traces only
    online_all_traces: Optional[List[Dict]] = None  # NEW: All attempted traces
    online_voting_breakdown: Optional[Dict] = None  # NEW

    # Pooling results
    pooling_answer: Optional[str] = None
    pooling_correct: Optional[bool] = None
    pooling_tokens: Optional[int] = None
    pooling_time: Optional[float] = None
    pooling_candidates_count: Optional[int] = None
    pooling_followup_tokens: Optional[int] = None
    pooling_info_packet: Optional[List[Dict]] = None  # NEW: Consensus info
    pooling_traces: Optional[List[Dict]] = None  # NEW
    pooling_voting_breakdown: Optional[Dict] = None  # NEW

    # Error tracking
    errors: Dict[str, str] = field(default_factory=dict)

    def to_dict(self):
        """Convert to dict, handling nested dataclasses"""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                # Already dicts
                result[key] = value
            else:
                result[key] = value
        return result


@app.cls(
    image=gpu_image,
    gpu=GPU_CONFIG,
    timeout=3600,
    volumes={"/models": models_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class DeepThinkRunner:
    """Modal class for running deep thinking methods with full trace saving"""

    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.model_name = model_name

    @modal.enter()
    def setup(self):
        """Initialize vLLM model on Modal container startup"""
        import sys
        sys.path.insert(0, "/root")

        from deepconf import DeepThinkLLM

        print(f"Loading model: {self.model_name}")
        start_time = time.time()

        self.deep_llm = DeepThinkLLM(
            model=self.model_name,
            tensor_parallel_size=4,
            enable_prefix_caching=True,
            trust_remote_code=True,
            download_dir="/models",
        )

        self.tokenizer = self.deep_llm.tokenizer

        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f}s")

    def _prepare_prompt(self, question: str) -> str:
        """Prepare prompt using model's chat template"""
        messages = [{"role": "user", "content": question}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt

    def _equal_func(self, answer: str, ground_truth: str) -> bool:
        """Check if answer equals ground truth"""
        answer_clean = str(answer).strip().lower()
        gt_clean = str(ground_truth).strip().lower()
        return answer_clean == gt_clean

    def _extract_trace_info(self, traces: List[Dict], save_full_text: bool = True) -> List[Dict]:
        """Extract trace information with optional full text"""
        trace_infos = []

        for idx, trace in enumerate(traces):
            info = {
                'trace_id': idx,
                'extracted_answer': trace.get('extracted_answer', ''),
                'min_conf': trace.get('min_conf', 0.0),
                'avg_conf': trace.get('avg_conf', 0.0) if 'avg_conf' in trace else
                           (sum(trace.get('confs', [0])) / len(trace.get('confs', [1])) if trace.get('confs') else 0.0),
                'tokens': len(trace.get('token_ids', [])),
                'stop_reason': trace.get('stop_reason', 'unknown'),
            }

            # Optionally include full text (can be very large!)
            if save_full_text:
                info['text'] = trace.get('text', '')
            else:
                # Save only first 500 chars as preview
                info['text_preview'] = trace.get('text', '')[:500]

            trace_infos.append(info)

        return trace_infos

    def _compute_voting_breakdown(self, traces: List[Dict]) -> Dict:
        """Compute voting breakdown statistics"""
        from collections import Counter

        answers = [t.get('extracted_answer', '') for t in traces if t.get('extracted_answer')]
        answer_counts = Counter(answers)

        breakdown = {
            'total_traces': len(traces),
            'unique_answers': len(answer_counts),
            'answer_distribution': dict(answer_counts.most_common()),
            'majority_answer': answer_counts.most_common(1)[0] if answer_counts else None,
            'majority_count': answer_counts.most_common(1)[0][1] if answer_counts else 0,
            'majority_fraction': answer_counts.most_common(1)[0][1] / len(traces) if answer_counts and traces else 0,
        }

        return breakdown

    @modal.method()
    def run_mv256(
        self,
        question: str,
        ground_truth: str,
        qid: int,
        save_full_traces: bool = True
    ) -> Dict[str, Any]:
        """Run Majority Voting with 256 full traces"""
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

            final_answer = result.voted_answer
            is_correct = self._equal_func(final_answer, ground_truth) if final_answer else False

            # Extract trace information
            traces_info = self._extract_trace_info(result.all_traces, save_full_text=save_full_traces)
            voting_breakdown = self._compute_voting_breakdown(result.all_traces)

            return {
                "answer": final_answer,
                "correct": is_correct,
                "tokens": result.total_tokens,
                "time": elapsed_time,
                "traces_count": result.total_traces_count,
                "traces": traces_info,
                "voting_breakdown": voting_breakdown,
                "voting_results": result.voting_results,
            }

        except Exception as e:
            print(f"Error in MV256: {str(e)}")
            import traceback
            return {"error": str(e), "traceback": traceback.format_exc()}

    @modal.method()
    def run_online(
        self,
        question: str,
        ground_truth: str,
        qid: int,
        warmup_traces: int = 64,
        total_budget: int = 256,
        confidence_percentile: int = 10,
        save_full_traces: bool = True
    ) -> Dict[str, Any]:
        """Run Online mode with confidence-based early stopping"""
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

            final_answer = result.voted_answer
            is_correct = self._equal_func(final_answer, ground_truth) if final_answer else False

            survivors = len(result.all_voting_traces) if hasattr(result, 'all_voting_traces') else 0

            # Extract survivor traces and all traces
            survivor_traces_info = self._extract_trace_info(
                result.all_voting_traces if hasattr(result, 'all_voting_traces') else [],
                save_full_text=save_full_traces
            )

            all_traces_info = self._extract_trace_info(
                result.all_traces,
                save_full_text=save_full_traces
            )

            voting_breakdown = self._compute_voting_breakdown(
                result.all_voting_traces if hasattr(result, 'all_voting_traces') else []
            )

            return {
                "answer": final_answer,
                "correct": is_correct,
                "tokens": result.total_tokens,
                "time": elapsed_time,
                "survivors": survivors,
                "warmup_traces": warmup_traces,
                "conf_threshold": result.conf_bar,
                "survivor_traces": survivor_traces_info,
                "all_traces": all_traces_info,
                "voting_breakdown": voting_breakdown,
                "voting_results": result.voting_results,
                "warmup_tokens": result.warmup_tokens,
                "final_tokens": result.final_tokens,
            }

        except Exception as e:
            print(f"Error in Online: {str(e)}")
            import traceback
            return {"error": str(e), "traceback": traceback.format_exc()}

    @modal.method()
    def run_pooling(
        self,
        question: str,
        ground_truth: str,
        qid: int,
        warmup_traces: int = 64,
        total_budget: int = 256,
        confidence_percentile: int = 10,
        save_full_traces: bool = True
    ) -> Dict[str, Any]:
        """Run Pooling method with full info packet saved"""
        from vllm import SamplingParams
        from collections import Counter, defaultdict

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

            start_time = time.time()

            result = self.deep_llm.deepthink(
                prompt=prompt,
                mode="online",
                warmup_traces=warmup_traces,
                total_budget=total_budget,
                confidence_percentile=confidence_percentile,
                window_size=2048,
                sampling_params=sampling_params,
                compute_multiple_voting=False
            )

            screening_time = time.time() - start_time
            pooling_start = time.time()

            traces = result.all_voting_traces if hasattr(result, 'all_voting_traces') else result.all_traces

            # Build information packet
            answer_counts = Counter()
            answer_to_traces = defaultdict(list)

            for trace in traces:
                ans = trace.get('extracted_answer')
                if ans:
                    answer_counts[ans] += 1
                    answer_to_traces[ans].append(trace)

            top5_answers = [ans for ans, _ in answer_counts.most_common(5)]

            # Full info packet with sample reasoning
            info_packet = []
            for ans in top5_answers:
                count = answer_counts[ans]
                sample_trace = answer_to_traces[ans][0]

                packet_entry = {
                    'answer': ans,
                    'count': count,
                    'support_fraction': count / len(traces),
                    'reasoning_preview': sample_trace.get('text', '')[:500],
                    'sample_trace_conf': sample_trace.get('min_conf', 0.0),
                }

                if save_full_traces:
                    packet_entry['full_reasoning'] = sample_trace.get('text', '')

                info_packet.append(packet_entry)

            # Consensus-conditioned voting
            weighted_votes = defaultdict(float)

            for trace in traces:
                ans = trace.get('extracted_answer')
                if ans:
                    conf = trace.get('min_conf', 1.0)
                    support_weight = answer_counts[ans] / len(traces)
                    weighted_votes[ans] += conf * (1 + support_weight)

            if weighted_votes:
                final_answer = max(weighted_votes.items(), key=lambda x: x[1])[0]
            else:
                final_answer = None

            pooling_time = time.time() - pooling_start
            total_time = screening_time + pooling_time

            is_correct = self._equal_func(final_answer, ground_truth) if final_answer else False

            # Extract trace info
            traces_info = self._extract_trace_info(traces, save_full_text=save_full_traces)
            voting_breakdown = self._compute_voting_breakdown(traces)

            return {
                "answer": final_answer,
                "correct": is_correct,
                "tokens": result.total_tokens,
                "time": total_time,
                "screening_time": screening_time,
                "pooling_time": pooling_time,
                "candidates_count": len(top5_answers),
                "info_packet": info_packet,
                "traces": traces_info,
                "voting_breakdown": voting_breakdown,
                "survivors": len(traces),
                "conf_threshold": result.conf_bar,
                "followup_tokens": 0,
            }

        except Exception as e:
            print(f"Error in Pooling: {str(e)}")
            import traceback
            return {"error": str(e), "traceback": traceback.format_exc()}


@app.function(
    image=gpu_image.pip_install("pandas", "tabulate"),
    timeout=7200,
)
def run_benchmark(
    dataset_path: str,
    methods: List[str],
    model_name: str = "openai/gpt-oss-120b",
    output_dir: str = "results",
    start_qid: int = 0,
    end_qid: Optional[int] = None,
    save_full_traces: bool = False,  # NEW: Option to save full text (can be huge!)
) -> List[Dict]:
    """Run benchmark with full trace saving"""
    from pathlib import Path
    import json
    from datetime import datetime

    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r') as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]

    if end_qid is None:
        end_qid = len(data)

    questions = data[start_qid:end_qid]
    print(f"Running benchmark on {len(questions)} questions (QID {start_qid}-{end_qid-1})")
    print(f"Methods: {methods}")
    print(f"Model: {model_name}")
    print(f"Save full traces: {save_full_traces}")

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
                mv_result = runner.run_mv256.remote(question, ground_truth, qid, save_full_traces)

                if "error" in mv_result:
                    result.errors["mv256"] = mv_result["error"]
                else:
                    result.mv256_answer = mv_result["answer"]
                    result.mv256_correct = mv_result["correct"]
                    result.mv256_tokens = mv_result["tokens"]
                    result.mv256_time = mv_result["time"]
                    result.mv256_traces_count = mv_result["traces_count"]
                    result.mv256_traces = mv_result.get("traces", [])
                    result.mv256_voting_breakdown = mv_result.get("voting_breakdown", {})

                    print(f"  Answer: {result.mv256_answer}")
                    print(f"  Correct: {result.mv256_correct}")
                    print(f"  Tokens: {result.mv256_tokens:,}")
                    print(f"  Voting: {result.mv256_voting_breakdown.get('majority_answer')}")

            except Exception as e:
                result.errors["mv256"] = str(e)
                print(f"  Error: {str(e)}")

        # Run Online 10%
        if "online" in methods:
            print(f"\n[{qid}] Running Online 10%...")
            try:
                online_result = runner.run_online.remote(question, ground_truth, qid, save_full_traces=save_full_traces)

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
                    result.online_survivor_traces = online_result.get("survivor_traces", [])
                    result.online_all_traces = online_result.get("all_traces", [])
                    result.online_voting_breakdown = online_result.get("voting_breakdown", {})

                    print(f"  Answer: {result.online_answer}")
                    print(f"  Correct: {result.online_correct}")
                    print(f"  Survivors: {result.online_survivors}")

            except Exception as e:
                result.errors["online"] = str(e)
                print(f"  Error: {str(e)}")

        # Run Pooling
        if "pooling" in methods:
            print(f"\n[{qid}] Running Pooling...")
            try:
                pooling_result = runner.run_pooling.remote(question, ground_truth, qid, save_full_traces=save_full_traces)

                if "error" in pooling_result:
                    result.errors["pooling"] = pooling_result["error"]
                else:
                    result.pooling_answer = pooling_result["answer"]
                    result.pooling_correct = pooling_result["correct"]
                    result.pooling_tokens = pooling_result["tokens"]
                    result.pooling_time = pooling_result["time"]
                    result.pooling_candidates_count = pooling_result["candidates_count"]
                    result.pooling_followup_tokens = pooling_result["followup_tokens"]
                    result.pooling_info_packet = pooling_result.get("info_packet", [])
                    result.pooling_traces = pooling_result.get("traces", [])
                    result.pooling_voting_breakdown = pooling_result.get("voting_breakdown", {})

                    print(f"  Answer: {result.pooling_answer}")
                    print(f"  Correct: {result.pooling_correct}")
                    print(f"  Candidates: {result.pooling_candidates_count}")

            except Exception as e:
                result.errors["pooling"] = str(e)
                print(f"  Error: {str(e)}")

        results.append(result.to_dict())

        # Save intermediate results
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        intermediate_file = f"{output_dir}/intermediate_{timestamp}.jsonl"

        with open(intermediate_file, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')

    # Save final results
    final_file = f"{output_dir}/benchmark_full_traces_{timestamp}.jsonl"
    with open(final_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    print(f"\n{'='*80}")
    print(f"Benchmark complete! Results saved to {final_file}")
    print(f"{'='*80}")

    return results


@app.local_entrypoint()
def main(
    dataset: str = "data/raw/aime_2024.jsonl",
    method: str = "all",
    model: str = "openai/gpt-oss-120b",
    output_dir: str = "results",
    start_qid: int = 0,
    end_qid: int = None,
    save_full_traces: bool = False,
):
    """
    Main entry point with trace saving option

    Examples:
        # Save metadata only (lightweight)
        modal run modal_runner_with_traces.py --dataset aime_2024.jsonl

        # Save full reasoning text (LARGE files!)
        modal run modal_runner_with_traces.py --dataset aime_2024.jsonl --save-full-traces
    """

    if method == "all":
        methods = ["mv256", "online", "pooling"]
    else:
        methods = [m.strip() for m in method.split(",")]

    print(f"Starting benchmark with methods: {methods}")
    print(f"Save full traces: {save_full_traces}")

    if save_full_traces:
        print("\n⚠️  WARNING: Saving full traces will create VERY large output files!")
        print("   For 256 traces × 2000 tokens each ≈ 100MB+ per question")

    results = run_benchmark.remote(
        dataset_path=dataset,
        methods=methods,
        model_name=model,
        output_dir=output_dir,
        start_qid=start_qid,
        end_qid=end_qid,
        save_full_traces=save_full_traces
    )

    print("\nBenchmark completed successfully!")
