import os
import re
import json
import pandas as pd
from pathlib import Path

# ========== 配置部分 ==========
DATA_NAME = "brumo_2025"
BASE_DIR = Path(f"/home/yz54720/Projects/Method/deepconf/data/processed/{DATA_NAME}/pool_information_v2")
AIME_DATA_PATH = Path(f"/home/yz54720/Projects/Method/deepconf/data/raw/{DATA_NAME}.jsonl")
OUTPUT_CSV = BASE_DIR / "followup_evaluation_summary.csv"
FILE_PATTERN = f"{DATA_NAME}_*_deepconflow_self_check.jsonl"

# ========== 工具函数 ==========
import re
import regex  # pip install regex



def clean_latex_answer(ans: str) -> str:
    """清洗 LaTeX 表达式：去空格、去 a= 等、标准化 \\dfrac"""
    if not ans:
        return ans

    # 去掉 LaTeX 空格命令（\ , \quad, \qquad 等）
    ans = re.sub(r"\\(?:,|;|:|!|quad|qquad|enspace|,| )", "", ans)
    
    # 去掉所有普通空格
    ans = re.sub(r"\s+", "", ans)
    
    # 将 \dfrac 转为 \frac
    ans = ans.replace(r"\dfrac", r"\frac")
    
    # 去掉形如 a=、x=、y= 等赋值
    ans = re.sub(r"\b[a-zA-Z]\s*=", "", ans)
    
    # 去掉可能的多余逗号空位，比如 "2,,3" → "2,3"
    ans = re.sub(r",+", ",", ans)
    
    # 去掉首尾逗号
    ans = ans.strip(",")
    
    return ans
def extract_boxed_answer(text: str):
    """提取文本中最后一个 \\boxed{...} 的内容（支持任意嵌套 {}）并清洗"""
    if not text:
        return None

    results = []
    pos = 0
    while True:
        start = text.find(r'\boxed{', pos)
        if start == -1:
            break  # 没有更多了

        i = start + len(r'\boxed{')
        depth = 1
        content_chars = []

        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == '{':
                depth += 1
                content_chars.append(ch)
            elif ch == '}':
                depth -= 1
                if depth > 0:
                    content_chars.append(ch)
            else:
                content_chars.append(ch)
            i += 1

        if depth == 0:
            results.append(''.join(content_chars))
            pos = i  # 从上次结束后继续查找
        else:
            break  # 不完整，结束循环

    if not results:
        return None

    # 返回最后一个
    return clean_latex_answer(results[-1])
# def extract_boxed_answer(text: str):
#     """从模型输出中提取 \\boxed{} 内的最终答案"""
#     if not text:
#         return None
#     match = re.search(r"\\boxed\{([^}]*)\}", text)
#     if match:
#         return match.group(1).strip()
#     return None


def parse_jsonl(file_path):
    """逐行读取 JSONL 文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def extract_followup_answers(jsonl_path, qid):
    """
    从每个 follow-up JSONL 文件中提取：
    - 原始答案
    - follow-up 后答案
    """
    try:
        records = list(parse_jsonl(jsonl_path))
    except Exception as e:
        print(f"⚠️ Error reading {jsonl_path}: {e}")
        return None

    results = []
    for rec in records:
        original = rec.get("base_answer", None)
        followup_text = rec.get("trace_2", "")
        followup_ans = extract_boxed_answer(followup_text)
        original = clean_latex_answer(original)
        results.append({
            "question_id": qid,
            "original_answer": original,
            "after_followup": followup_ans,
            "changed": (followup_ans is not None and original is not None and followup_ans != original)
        })
    return results


def load_ground_truth(aime_jsonl):
    """加载 ground truth 答案字典 {qid: correct_answer}"""
    gt_dict = {}
    for i, item in enumerate(parse_jsonl(aime_jsonl)):
        gt_dict[i] = str(item.get("answer", "")).strip()
    return gt_dict


# ========== 主逻辑 ==========

def main():
    all_results = []
    files = list(BASE_DIR.glob(FILE_PATTERN))
    if not files:
        print("⚠️ No follow-up result files found.")
        return

    print(f"Found {len(files)} follow-up files.")
    gt_dict = load_ground_truth(AIME_DATA_PATH)

    for f in sorted(files):
        qid_match = re.search(r".*_(\d+)_deepconflow_self_check", f.name)
        qid = int(qid_match.group(1)) if qid_match else None

        file_results = extract_followup_answers(f, qid)
        if not file_results:
            print(f"⚠️ No valid entries in {f}")
            continue

        for r in file_results:
            r["question_id"] = qid
            r["ground_truth"] = gt_dict.get(qid, "N/A")

            # correctness check
            r["original_correct"] = (r["original_answer"] == r["ground_truth"])
            r["followup_correct"] = (r["after_followup"] == r["ground_truth"])

            # textual summary
            if r["original_correct"] and r["followup_correct"]:
                status = "✅ Correct → Correct"
            elif r["original_correct"] and not r["followup_correct"]:
                status = "❌ Correct → Wrong"
            elif not r["original_correct"] and r["followup_correct"]:
                status = "✅ Wrong → Correct"
            else:
                status = "❌ Wrong → Wrong"

            if r["changed"]:
                status += " (🔁 changed)"
            else:
                status += " (no change)"

            r["status_summary"] = status
            all_results.append(r)

    if not all_results:
        print("⚠️ No results extracted.")
        return

    df = pd.DataFrame(all_results)
    df = df.sort_values("question_id").reset_index(drop=True)

    print("\n=== 🧩 Follow-up Evaluation Summary ===")
    print(df[[
        "question_id",
        "ground_truth",
        "original_answer",
        "after_followup",
        "changed",
        "original_correct",
        "followup_correct",
        "status_summary"
    ]].to_string(index=False))

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Summary saved to: {OUTPUT_CSV}")

    # 汇总统计
    total = len(df)
    changed = df["changed"].sum()
    orig_correct = df["original_correct"].sum()
    follow_correct = df["followup_correct"].sum()

    print("\n=== 📊 Statistics ===")
    print(f"Total questions: {total}")
    print(f"Changed answers: {changed} ({changed/total:.2%})")
    print(f"Original correct: {orig_correct}/{total} = {orig_correct/total:.2%}")
    print(f"After follow-up correct: {follow_correct}/{total} = {follow_correct/total:.2%}")
    delta = follow_correct - orig_correct
    print(f"Δ Improvement: {delta} ({delta/total:.2%})")


if __name__ == "__main__":
    main()
