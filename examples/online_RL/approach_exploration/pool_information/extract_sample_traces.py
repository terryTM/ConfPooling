import os
import re
import json
import random
from deepconf import DeepThinkLLM

# 初始化 tokenizer
deep_llm = DeepThinkLLM(model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")

# --- 清洗函数 ---
def split_thinking_and_answer(text):
    """
    拆分 <think> ... </think> 结构，返回 (reasoning, answer)，
    并去掉末尾的 <｜end▁of▁sentence｜>。
    """
    match = re.search(r"<think>(.*?)</think>(.*)", text, flags=re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
        answer = match.group(2).strip()
    else:
        reasoning = ""
        answer = text.strip()

    answer = re.sub(r"<\s*[\|｜]\s*end▁of▁sentence\s*[\|｜]\s*>", "", answer, flags=re.IGNORECASE).strip()
    return reasoning, answer


# --- JSON 读取函数 ---
def load_concatenated_json(file_path):
    """读取拼接 JSONL 文件"""
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

    final_data = {k: v for k, v in data_parts[0].items() if k != 'traces'}
    all_traces = [trace for part in data_parts for trace in part.get('traces', [])]
    final_data['traces'] = all_traces
    final_data['num_traces'] = len(all_traces)

    return final_data


# --- 主逻辑 ---
def extract_random_traces(input_dir, output_txt, num_samples=2):
    """
    对每个 JSONL 问题文件，随机抽取若干条 trace 并导出为结构化文本
    """
    random.seed(42)
    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jsonl")])
    print(f"Found {len(files)} problem files.")

    with open(output_txt, "w", encoding="utf-8") as out:
        for i, filename in enumerate(files):
            file_path = os.path.join(input_dir, filename)
            print(f"[{i+1}/{len(files)}] Processing {filename} ...")

            data = load_concatenated_json(file_path)
            if not data or not data.get("traces"):
                print(f"  ⚠️  No valid traces found in {filename}")
                continue

            question = data.get("question", "").strip()
            ground_truth = data.get("ground_truth", "N/A")
            traces = data["traces"]
            samples = random.sample(traces, min(num_samples, len(traces)))

            out.write(f"# Question {i+1}\n")
            out.write("## question and answer\n")
            out.write(f"{question}\n")
            out.write(f"Correct Answer: {ground_truth}\n")
            out.write("## extracted trace\n")

            for j, t in enumerate(samples):
                # ✅ 如果没有 text，就从 tokens 解码
                if "text" in t and t["text"]:
                    trace_text = t["text"]
                elif "tokens" in t and t["tokens"]:
                    try:
                        trace_text = deep_llm.tokenizer.convert_tokens_to_string(t["tokens"])
                    except Exception:
                        trace_text = "⚠️ Failed to decode tokens."
                else:
                    trace_text = "⚠️ Empty trace."

                _, answer = split_thinking_and_answer(trace_text)
                out.write(f"\n### Trace {j+1}\n")
                out.write(answer.strip() + "\n")

            out.write("\n" + "=" * 80 + "\n\n")

    print(f"\n✅ All done. Output written to: {output_txt}")


# --- 示例调用 ---
if __name__ == "__main__":
    INPUT_DIR = "/home/yz54720/Projects/Method/deepconf/trace_data"
    OUTPUT_TXT = "/home/yz54720/Projects/Method/deepconf/trace_data/pool_information_v3/trace_samples.txt"
    extract_random_traces(INPUT_DIR, OUTPUT_TXT, num_samples=2)
