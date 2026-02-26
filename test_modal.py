"""
Quick test script to validate Modal setup before running full benchmark

Usage:
    modal run test_modal.py                    # Test with default settings
    modal run test_modal.py --model "Qwen/Qwen2.5-Math-32B"  # Test different model
"""
import modal
from pathlib import Path

app = modal.App("confidence-pooling-test")

# Same image as main runner
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.10.2",
        "transformers>=4.46.0",
        "torch>=2.5.0",
        "numpy>=1.24.0",
    )
    .add_local_dir(
        "deepconf",
        "/root/deepconf"
    )
)

models_volume = modal.Volume.from_name("models-cache", create_if_missing=True)


@app.cls(
    image=gpu_image,
    gpu="A100-40GB:2",  # Smaller for testing
    timeout=600,  # 10 minutes
    volumes={"/models": models_volume},
)
class TestRunner:
    """Test runner for validating setup"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct"):
        self.model_name = model_name

    @modal.enter()
    def setup(self):
        """Initialize model"""
        import sys
        sys.path.insert(0, "/root")

        from deepconf import DeepThinkLLM
        import time

        print(f"\n{'='*80}")
        print(f"SETUP TEST: Loading {self.model_name}")
        print(f"{'='*80}\n")

        start = time.time()

        self.deep_llm = DeepThinkLLM(
            model=self.model_name,
            tensor_parallel_size=2,
            enable_prefix_caching=True,
            trust_remote_code=True,
            download_dir="/models",
        )

        elapsed = time.time() - start
        print(f"\n✓ Model loaded successfully in {elapsed:.2f}s\n")

    @modal.method()
    def test_generation(self) -> dict:
        """Test basic generation"""
        from vllm import SamplingParams
        import time

        print(f"\n{'='*80}")
        print("GENERATION TEST: Simple math question")
        print(f"{'='*80}\n")

        question = "What is 15 + 27? Put your final answer in \\boxed{}."

        messages = [{"role": "user", "content": question}]
        prompt = self.deep_llm.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=1000,
            logprobs=20,
        )

        print(f"Question: {question}")
        print(f"Generating...")

        start = time.time()
        outputs = self.deep_llm.llm.generate([prompt], sampling_params)
        elapsed = time.time() - start

        text = outputs[0].outputs[0].text
        print(f"\nGenerated in {elapsed:.2f}s:")
        print(f"Response: {text}\n")

        return {
            "question": question,
            "response": text,
            "time": elapsed,
            "tokens": len(outputs[0].outputs[0].token_ids)
        }

    @modal.method()
    def test_offline_mode(self) -> dict:
        """Test offline mode with multiple traces"""
        from vllm import SamplingParams
        import time

        print(f"\n{'='*80}")
        print("OFFLINE MODE TEST: 32 traces")
        print(f"{'='*80}\n")

        question = "What is the square root of 144? Put your final answer in \\boxed{}."

        messages = [{"role": "user", "content": question}]
        prompt = self.deep_llm.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=2000,
            logprobs=20,
        )

        print(f"Question: {question}")
        print(f"Generating 32 traces...")

        start = time.time()
        result = self.deep_llm.deepthink(
            prompt=prompt,
            mode="offline",
            budget=32,
            window_size=512,
            sampling_params=sampling_params,
            compute_multiple_voting=True
        )
        elapsed = time.time() - start

        print(f"\n✓ Offline mode completed in {elapsed:.2f}s")
        print(f"  Traces: {result.total_traces_count}")
        print(f"  Tokens: {result.total_tokens:,}")
        print(f"  Voted answer: {result.voted_answer}")
        print(f"  Avg tokens/trace: {result.avg_tokens_per_trace:.0f}\n")

        return {
            "mode": "offline",
            "question": question,
            "answer": result.voted_answer,
            "traces_count": result.total_traces_count,
            "total_tokens": result.total_tokens,
            "time": elapsed,
        }

    @modal.method()
    def test_online_mode(self) -> dict:
        """Test online mode with early stopping"""
        from vllm import SamplingParams
        import time

        print(f"\n{'='*80}")
        print("ONLINE MODE TEST: 16 warmup + 16 online traces (32 total budget)")
        print(f"{'='*80}\n")

        question = "What is the square root of 144? Put your final answer in \\boxed{}."

        messages = [{"role": "user", "content": question}]
        prompt = self.deep_llm.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=2000,
            logprobs=20,
        )

        print(f"Question: {question}")
        print(f"Generating with confidence-based early stopping...")

        start = time.time()
        result = self.deep_llm.deepthink(
            prompt=prompt,
            mode="online",
            warmup_traces=16,
            total_budget=32,
            confidence_percentile=10,
            window_size=512,
            sampling_params=sampling_params,
            compute_multiple_voting=True
        )
        elapsed = time.time() - start

        survivors = len(result.all_voting_traces) if hasattr(result, 'all_voting_traces') else 0

        print(f"\n✓ Online mode completed in {elapsed:.2f}s")
        print(f"  Confidence threshold: {result.conf_bar:.3f}")
        print(f"  Warmup traces: {len(result.warmup_traces)}")
        print(f"  Final traces: {len(result.final_traces)}")
        print(f"  Survivors: {survivors}")
        print(f"  Total tokens: {result.total_tokens:,}")
        print(f"  Voted answer: {result.voted_answer}\n")

        return {
            "mode": "online",
            "question": question,
            "answer": result.voted_answer,
            "survivors": survivors,
            "conf_threshold": result.conf_bar,
            "total_tokens": result.total_tokens,
            "warmup_tokens": result.warmup_tokens,
            "final_tokens": result.final_tokens,
            "time": elapsed,
        }


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-Math-7B-Instruct"):
    """
    Run all tests

    Examples:
        modal run test_modal.py
        modal run test_modal.py --model "Qwen/Qwen2.5-Math-32B-Instruct"
    """
    print(f"\n{'#'*80}")
    print(f"{'#'*80}")
    print(f"  MODAL SETUP TEST")
    print(f"  Model: {model}")
    print(f"{'#'*80}")
    print(f"{'#'*80}\n")

    runner = TestRunner(model_name=model)

    # Test 1: Basic generation
    try:
        print("\n[1/3] Testing basic generation...")
        gen_result = runner.test_generation.remote()
        print(f"✓ Basic generation test passed")
        print(f"  Generated {gen_result['tokens']} tokens in {gen_result['time']:.2f}s")
    except Exception as e:
        print(f"✗ Basic generation test failed: {e}")
        return

    # Test 2: Offline mode
    try:
        print("\n[2/3] Testing offline mode...")
        offline_result = runner.test_offline_mode.remote()
        print(f"✓ Offline mode test passed")
        print(f"  Answer: {offline_result['answer']}")
        print(f"  Tokens: {offline_result['total_tokens']:,}")
    except Exception as e:
        print(f"✗ Offline mode test failed: {e}")
        return

    # Test 3: Online mode
    try:
        print("\n[3/3] Testing online mode...")
        online_result = runner.test_online_mode.remote()
        print(f"✓ Online mode test passed")
        print(f"  Answer: {online_result['answer']}")
        print(f"  Survivors: {online_result['survivors']}")
        print(f"  Tokens: {online_result['total_tokens']:,}")
    except Exception as e:
        print(f"✗ Online mode test failed: {e}")
        return

    # Summary
    print(f"\n{'#'*80}")
    print("ALL TESTS PASSED ✓")
    print(f"{'#'*80}\n")

    print("Summary:")
    print(f"  Basic Generation: {gen_result['time']:.2f}s, {gen_result['tokens']} tokens")
    print(f"  Offline Mode:     {offline_result['time']:.2f}s, {offline_result['total_tokens']:,} tokens, {offline_result['traces_count']} traces")
    print(f"  Online Mode:      {online_result['time']:.2f}s, {online_result['total_tokens']:,} tokens, {online_result['survivors']} survivors")

    # Calculate token savings
    if offline_result['total_tokens'] > 0:
        savings = (1 - online_result['total_tokens'] / offline_result['total_tokens']) * 100
        print(f"\nOnline mode token savings: {savings:.1f}% vs offline")

    print(f"\n{'#'*80}")
    print("Ready to run full benchmark!")
    print("Use: modal run modal_runner.py --dataset <path> --method all")
    print(f"{'#'*80}\n")
